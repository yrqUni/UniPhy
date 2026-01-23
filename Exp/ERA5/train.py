import argparse
import contextlib
import datetime
import glob
import logging
import os
import random
import sys
import time
import warnings
import yaml
from typing import Any, Optional, Tuple

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

def crps_ensemble_loss(pred_ensemble: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    target = target.unsqueeze(1)
    mae = torch.mean(torch.abs(pred_ensemble - target), dim=1)
    M = pred_ensemble.shape[1]
    pred_sorted, _ = torch.sort(pred_ensemble, dim=1)
    indices = torch.arange(1, M + 1, device=pred_ensemble.device).view(1, M, 1, 1, 1)
    spread = torch.mean(pred_sorted * (2 * indices - M - 1), dim=1) / M
    return (mae - spread).mean(), spread.mean()

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
    file_formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(file_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

def setup_wandb(rank, cfg):
    if rank != 0 or not cfg['logging'].get('use_wandb', False): return
    wandb.init(project=cfg['logging']['wandb_project'], entity=cfg['logging']['wandb_entity'], name=cfg['logging']['wandb_run_name'], config=cfg)

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
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_ds = ERA5_Dataset(is_train=True, **cfg['data'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=False, drop_last=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=cfg['train']['batch_size'], num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=cfg['train']['weight_decay'])
    grad_accum_steps = cfg['train']['grad_accum_steps']
    epochs = cfg['train']['epochs']
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=float(cfg['train']['lr']), steps_per_epoch=len(train_loader)//grad_accum_steps, epochs=epochs) if cfg['train']['use_scheduler'] else None

    global_step = 0
    save_interval = max(1, int(len(train_loader) * cfg['logging']['ckpt_step']))
    log_every = cfg['logging']['log_every']
    start_time = time.time()
    ensemble_size = cfg['train']['ensemble_size']

    for ep in range(epochs):
        train_sampler.set_epoch(ep)
        with Progress(TextColumn("[bold blue]Epoch {task.fields[ep]}"), BarColumn(bar_width=40), MofNCompleteColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console, disable=(rank != 0)) as progress:
            task = progress.add_task("Train", total=len(train_loader), ep=ep+1)
            for train_step, (data, dt) in enumerate(train_loader, start=1):
                model.train()
                data, dt = data.to(f"cuda:{local_rank}").float(), dt.to(f"cuda:{local_rank}")
                x_in, target = data[:, :-1], data[:, 1:]
                
                is_accum = (train_step % grad_accum_steps != 0)
                sync_ctx = model.no_sync() if is_accum else contextlib.nullcontext()

                with sync_ctx:
                    ensemble_preds = [model(x_in, dt).view(-1, target.shape[2], target.shape[3], target.shape[4]) for _ in range(ensemble_size)]
                    pred_ensemble = torch.stack(ensemble_preds, dim=1)
                    target_flat = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])
                    
                    crps, avg_spread = crps_ensemble_loss(pred_ensemble, target_flat)
                    
                    with torch.no_grad():
                        pred_mean = pred_ensemble.mean(dim=1)
                        mae_val = F.l1_loss(pred_mean, target_flat)
                        rmse_val = F.mse_loss(pred_mean, target_flat) ** 0.5
                        raw_spread = pred_ensemble.std(dim=1).mean()
                    
                    loss_collapse = torch.clamp(0.4 * rmse_val - raw_spread, min=0)
                    loss = crps + 2.0 * loss_collapse
                    (loss / grad_accum_steps).backward()

                if not is_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
                    gn, _, _ = get_grad_stats(model)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    global_step += 1

                    if rank == 0 and train_step % log_every == 0:
                        cur_noise = F.softplus(model.module.blocks[0].prop.raw_noise_param).item()
                        logger.info(f"CRPS: {crps:.4f} | MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f} | Spread: {raw_spread:.4f} | Noise: {cur_noise:.4f}")
                        if cfg['logging']['use_wandb']:
                            wandb.log({"train/crps": crps, "train/mae": mae_val, "train/rmse": rmse_val, "train/spread": raw_spread, "train/noise_scale": cur_noise, "train/gn": gn})

                    if train_step % save_interval == 0:
                        save_ckpt(model, opt, ep+1, train_step, crps.item(), cfg, scheduler)
                progress.advance(task)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f: cfg = yaml.safe_load(f)
    run_ddp(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"]), os.environ.get("MASTER_ADDR", "127.0.0.1"), os.environ.get("MASTER_PORT", "12355"), cfg)
