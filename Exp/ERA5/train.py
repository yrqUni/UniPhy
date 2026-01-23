import argparse
import contextlib
import datetime
import glob
import logging
import os
import gc
import random
import sys
import time
import warnings
import yaml
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")

custom_theme = Theme({
    "metric": "bold cyan",
    "value": "bold white",
    "danger": "bold red",
    "param": "dim yellow"
})
console = Console(theme=custom_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)]
)
logger = logging.getLogger("rich")

def crps_ensemble_loss(pred_ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1)
    mae = torch.mean(torch.abs(pred_ensemble - target), dim=1)
    M = pred_ensemble.shape[1]
    pred_sorted, _ = torch.sort(pred_ensemble, dim=1)
    indices = torch.arange(1, M + 1, device=pred_ensemble.device).view(1, M, 1, 1, 1)
    spread = torch.mean(pred_sorted * (2 * indices - M - 1), dim=1) / M
    return (mae - spread).mean()

def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

set_random_seed(1017, deterministic=False)

def format_params(num):
    if num >= 1e9: readable = f"{num / 1e9:.2f}B"
    elif num >= 1e6: readable = f"{num / 1e6:.2f}M"
    elif num >= 1e3: readable = f"{num / 1e3:.2f}K"
    else: readable = f"{num}"
    return f"{num:,} ({readable})"

def log_model_stats(model: torch.nn.Module, rank: int) -> None:
    if rank != 0: return
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right")
    
    stats_table = Table(box=None, show_header=False, pad_edge=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="bold white", justify="right")
    stats_table.add_row("Total Params", format_params(total_params))
    stats_table.add_row("Trainable", format_params(trainable_params))
    
    console.print(Panel(stats_table, title="[bold blue]Model Statistics[/]", border_style="blue", expand=False))

def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str, local_rank: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(local_rank)

def cleanup_ddp() -> None:
    if dist.is_initialized(): dist.destroy_process_group()

def setup_file_logging(cfg: dict, rank: int) -> None:
    if rank != 0: return
    os.makedirs(cfg['logging']['log_path'], exist_ok=True)
    log_filename = os.path.join(cfg['logging']['log_path'], f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(file_formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

def setup_wandb(rank, cfg):
    if rank != 0 or not cfg['logging'].get('use_wandb', False): return
    wandb_settings = wandb.Settings(
        start_method="spawn",
        init_timeout=600,
        _disable_stats=True,
        _disable_meta=True
    )
    wandb.init(
        project=cfg['logging']['wandb_project'],
        entity=cfg['logging']['wandb_entity'],
        name=cfg['logging']['wandb_run_name'],
        config=cfg,
        settings=wandb_settings,
        resume="allow"
    )

def save_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, metric: float, cfg: dict, scheduler: Optional[Any] = None) -> None:
    if dist.get_rank() != 0: return
    ckpt_dir = cfg['logging']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    
    state = {
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "step": step,
        "metric": metric,
        "config": cfg,
        "model_args": cfg['model'],
    }
    if scheduler: state["scheduler"] = scheduler.state_dict()
    ckpt_path = os.path.join(ckpt_dir, f"ep{epoch}_step{step}_crps{metric:.4f}.pth")
    torch.save(state, ckpt_path)
    
    files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    files.sort(key=os.path.getmtime)
    if len(files) > 3:
        for f in files[:-3]: 
            try: os.remove(f)
            except: pass

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu") -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
        if dist.get_rank() == 0:
            console.print(f"[yellow]No checkpoint found at {ckpt_path}. Starting fresh.[/yellow]")
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])
    ep = checkpoint.get("epoch", 0)
    st = checkpoint.get("step", 0)
    
    if dist.get_rank() == 0:
        console.print(Panel(f"RESUMED\nSrc: {ckpt_path}\nEp: {ep} | Step: {st}", border_style="green"))
    return ep, st

def get_grad_stats(model: torch.nn.Module) -> Tuple[float, float, float]:
    total_norm_sq = 0.0
    max_abs = 0.0
    param_norm_sq = 0.0
    for p in model.parameters():
        param_norm_sq += p.data.norm(2).item() ** 2
        if p.grad is None: continue
        g = p.grad.data
        total_norm_sq += g.norm(2).item() ** 2
        max_abs = max(max_abs, g.abs().max().item())
    return float(total_norm_sq**0.5), float(max_abs), float(param_norm_sq**0.5)

def run_ddp(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, cfg: dict) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0:
        setup_file_logging(cfg, rank)
        setup_wandb(rank, cfg)

    if cfg['train']['use_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(**cfg['model']).cuda(local_rank)
    
    log_model_stats(model, rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    train_ds = ERA5_Dataset(is_train=True, **cfg['data'])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=False, drop_last=True)
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=cfg['train']['batch_size'],
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [p for p in param_dict.values() if p.dim() >= 2], "weight_decay": cfg['train']['weight_decay']},
        {"params": [p for p in param_dict.values() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=float(cfg['train']['lr']))
    
    grad_accum_steps = cfg['train']['grad_accum_steps']
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    epochs = cfg['train']['epochs']
    
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=float(cfg['train']['lr']), steps_per_epoch=steps_per_epoch, epochs=epochs) if cfg['train']['use_scheduler'] else None

    start_ep, global_step = 0, 0
    if cfg['logging']['ckpt']:
        start_ep, global_step = load_ckpt(model, opt, cfg['logging']['ckpt'], scheduler, map_location=f"cuda:{local_rank}")

    save_interval = max(1, int(len(train_loader) * cfg['logging']['ckpt_step']))
    log_every = cfg['logging']['log_every']
    start_time = time.time()
    ensemble_size = cfg['train']['ensemble_size']
    grad_clip = cfg['train']['grad_clip']
    
    for ep in range(start_ep, epochs):
        train_sampler.set_epoch(ep)
        
        with Progress(
            TextColumn("[bold blue]Epoch {task.fields[ep]}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=(rank != 0)
        ) as progress:
            task = progress.add_task("Train", total=len(train_loader), ep=ep+1)
            
            for train_step, (data, dt) in enumerate(train_loader, start=1):
                model.train()
                
                data = data.to(f"cuda:{local_rank}", non_blocking=True).float()
                dt = dt.to(f"cuda:{local_rank}", non_blocking=True)
                
                x_in = data[:, :-1]
                target = data[:, 1:]
                
                is_accum = (train_step % grad_accum_steps != 0)
                sync_ctx = model.no_sync() if is_accum else contextlib.nullcontext()

                with sync_ctx:
                    ensemble_preds = []
                    
                    for _ in range(ensemble_size):
                        pred = model(x_in, dt)
                        B_seq, T_seq, C, H, W = target.shape
                        ensemble_preds.append(pred.view(B_seq * T_seq, C, H, W))
                        del pred
                    
                    pred_ensemble = torch.stack(ensemble_preds, dim=1)
                    target_flat = target.view(B_seq * T_seq, C, H, W)
                    
                    loss = crps_ensemble_loss(pred_ensemble, target_flat)
                    
                    with torch.no_grad():
                        pred_detach = pred_ensemble.detach()
                        pred_mean = torch.mean(pred_detach, dim=1)
                        l1_val = F.l1_loss(pred_mean, target_flat)
                        mse_val = F.mse_loss(pred_mean, target_flat)
                        spread_val = torch.std(pred_detach, dim=1).mean()
                        del pred_detach

                    if torch.isnan(loss) or torch.isinf(loss):
                        opt.zero_grad(set_to_none=True)
                        continue

                    (loss / grad_accum_steps).backward()

                del pred_ensemble, ensemble_preds
                
                if not is_accum:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    gn, max_g, pn = get_grad_stats(model)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    global_step += 1

                    with torch.no_grad():
                        metrics = torch.tensor([loss.item(), l1_val.item(), mse_val.item(), spread_val.item()], device=local_rank)
                        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                        metrics /= world_size
                        avg_crps, avg_l1, avg_mse, avg_spread = metrics.tolist()
                        avg_rmse = avg_mse ** 0.5
                    
                    if rank == 0:
                        progress.advance(task)
                        
                        if train_step % log_every == 0:
                            elapsed = time.time() - start_time
                            start_time = time.time()
                            logger.info(
                                f"Ep [param]{ep+1:03d}[/] | Step [param]{global_step:06d}[/] | "
                                f"[metric]CRPS[/]: [value]{avg_crps:.4f}[/] | [metric]MAE[/]: [value]{avg_l1:.4f}[/] | "
                                f"[metric]RMSE[/]: [value]{avg_rmse:.4f}[/] | "
                                f"[metric]Spread[/]: [value]{avg_spread:.4f}[/] | [metric]GN[/]: [value]{gn:.2f}[/]"
                            )
                            
                            file_msg = (f"Ep {ep+1:03d} | Step {global_step:06d} | CRPS {avg_crps:.4f} | "
                                      f"MAE {avg_l1:.4f} | RMSE {avg_rmse:.4f} | Spread {avg_spread:.4f} | GN {gn:.2f}")
                            logging.getLogger().handlers[1].handle(logging.LogRecord(
                                name="file", level=logging.INFO, pathname="", lineno=0,
                                msg=file_msg, args=(), exc_info=None))

                        if cfg['logging']['use_wandb'] and global_step % cfg['logging']['wandb_every'] == 0:
                            wandb.log({
                                "train/crps": avg_crps,
                                "train/l1": avg_l1,
                                "train/rmse": avg_rmse,
                                "train/spread": avg_spread,
                                "train/spread_ratio": avg_spread / (avg_rmse + 1e-6),
                                "train/lr": opt.param_groups[0]["lr"],
                                "train/grad_norm": gn,
                                "train/step": global_step,
                                "perf/sec_per_step": elapsed / log_every
                            })

                    if train_step % save_interval == 0:
                        save_ckpt(model, opt, ep+1, train_step, avg_crps, cfg, scheduler)

                del loss
                if train_step % 100 == 0: gc.collect()

        dist.barrier()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if rank == 0 and cfg['logging']['use_wandb']: wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_ddp(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=os.environ.get("MASTER_PORT", "12355"),
        cfg=cfg
    )
    