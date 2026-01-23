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
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

MODEL_ARG_KEYS = [
    "in_channels",
    "out_channels",
    "embed_dim",
    "expand",
    "num_experts",
    "depth",
    "patch_size",
    "img_height",
    "img_width",
]

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
    table = Table(title="Model Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Total Parameters", format_params(total_params))
    table.add_row("Trainable Parameters", format_params(trainable_params))
    console.print(Panel(table, expand=False))

class Args:
    def __init__(self) -> None:
        self.img_height = 721
        self.img_width = 1440
        self.in_channels = 30
        self.out_channels = 30
        self.embed_dim = 768
        self.expand = 4
        self.num_experts = 8
        self.patch_size = 32
        self.depth = 12
        self.ensemble_size = 2
        self.dt_ref = 6.0
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.use_scheduler = False
        self.lr = 3e-5
        self.weight_decay = 0.05
        self.epochs = 100
        self.grad_clip = 1.0
        self.grad_accum_steps = 1
        self.log_every = 1
        self.wandb_every = 1
        self.ckpt_step = 0.5
        self.log_path = "./uniphy/logs"
        self.ckpt_dir = "./uniphy/ckpt"
        self.ckpt = ""
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 20
        self.sample_k = 5
        self.eval_sample_num = 1
        self.use_tf32 = True
        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "UniPhy"
        self.wandb_run_name = "UniPhy-A800-Base-K8"

def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str, local_rank: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(local_rank)

def cleanup_ddp() -> None:
    if dist.is_initialized(): dist.destroy_process_group()

def setup_logging(args: Args) -> None:
    if not dist.is_initialized() or dist.get_rank() != 0: return
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    logging.getLogger().addHandler(file_handler)

def setup_wandb(rank, args):
    if rank != 0 or not bool(getattr(args, "use_wandb", False)): return
    wandb_settings = wandb.Settings(
        start_method="spawn",
        init_timeout=600,
        _disable_stats=True,
        _disable_meta=True
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        settings=wandb_settings,
        force=True
    )

def extract_model_args(args_obj: Any) -> Dict[str, Any]:
    return {k: getattr(args_obj, k) for k in MODEL_ARG_KEYS if hasattr(args_obj, k)}

def save_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, loss: float, args: Args, scheduler: Optional[Any] = None) -> None:
    if dist.get_rank() != 0: return
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "args": dict(vars(args)),
        "model_args": extract_model_args(args),
    }
    if scheduler: state["scheduler"] = scheduler.state_dict()
    ckpt_path = os.path.join(args.ckpt_dir, f"e{epoch}_s{step}_l{loss:.4f}.pth")
    torch.save(state, ckpt_path)
    files = glob.glob(os.path.join(args.ckpt_dir, "*.pth"))
    files.sort(key=os.path.getmtime)
    if len(files) > 5:
        for f in files[:-5]: os.remove(f)

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu") -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
        if dist.get_rank() == 0:
            console.print(f"[yellow][INFO] No checkpoint found at {ckpt_path}. Starting from scratch.[/yellow]")
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])
    ep = checkpoint.get("epoch", 0)
    st = checkpoint.get("step", 0)
    ls = checkpoint.get("loss", 0.0)
    if dist.get_rank() == 0:
        console.print(Panel(f"RESUMING TRAINING\nSource: {ckpt_path}\nEpoch: {ep}\nStep: {st}\nLoss: {ls:.6f}", title="Checkpoint Loaded", border_style="green"))
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

def run_ddp(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, args: Args) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0:
        setup_logging(args)
        setup_wandb(rank, args)
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = UniPhyModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        embed_dim=args.embed_dim,
        expand=args.expand,
        num_experts=args.num_experts,
        depth=args.depth,
        patch_size=args.patch_size,
        img_height=args.img_height,
        img_width=args.img_width
    ).cuda(local_rank)
    log_model_stats(model, rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    train_ds = ERA5_Dataset(
        input_dir=args.data_root,
        year_range=args.year_range,
        window_size=args.train_data_n_frames,
        sample_k=args.sample_k,
        look_ahead=2,
        is_train=True,
        dt_ref=args.dt_ref
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        shuffle=False, 
        drop_last=True)
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [p for p in param_dict.values() if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in param_dict.values() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs) if args.use_scheduler else None
    start_ep, global_step = 0, 0
    if args.ckpt:
        start_ep, global_step = load_ckpt(model, opt, args.ckpt, scheduler, map_location=f"cuda:{local_rank}")
    save_interval = max(1, int(len(train_loader) * args.ckpt_step))
    
    start_time = time.time()
    for ep in range(start_ep, args.epochs):
        train_sampler.set_epoch(ep)
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=(rank != 0)
        ) as progress:
            task = progress.add_task(f"Epoch {ep+1}", total=len(train_loader))
            for train_step, (data, dt) in enumerate(train_loader, start=1):
                model.train()
                data = data.to(f"cuda:{local_rank}", non_blocking=True).float()
                dt = dt.to(f"cuda:{local_rank}", non_blocking=True)
                
                x_in = data[:, :-1]
                target = data[:, 1:]
                B_seq, T_seq, C, H, W = target.shape
                
                is_accum = (train_step % args.grad_accum_steps != 0)
                sync_ctx = model.no_sync() if is_accum else contextlib.nullcontext()
                with sync_ctx:
                    ensemble_preds = []
                    for _ in range(args.ensemble_size):
                        pred = model(x_in, dt)
                        pred_flat = pred.reshape(B_seq * T_seq, C, H, W)
                        ensemble_preds.append(pred_flat)
                        del pred
                    pred_ensemble = torch.stack(ensemble_preds, dim=1)
                    target_flat = target.reshape(B_seq * T_seq, C, H, W)
                    
                    loss = crps_ensemble_loss(pred_ensemble, target_flat)
                    
                    with torch.no_grad():
                        pred_mean = torch.mean(pred_ensemble, dim=1)
                        l1_val = F.l1_loss(pred_mean, target_flat)
                        mse_val = F.mse_loss(pred_mean, target_flat)
                        spread_val = torch.std(pred_ensemble, dim=1).mean()

                    if torch.isnan(loss) or torch.isinf(loss):
                        opt.zero_grad(set_to_none=True)
                        continue
                    (loss / args.grad_accum_steps).backward()
                
                del pred_ensemble, ensemble_preds
                
                if not is_accum:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    gn, max_g, pn = get_grad_stats(model)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    global_step += 1
                    
                    with torch.no_grad():
                        metrics_tensor = torch.tensor([loss.item(), l1_val.item(), mse_val.item(), spread_val.item()], device=local_rank)
                        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                        metrics_tensor /= world_size
                        avg_crps, avg_l1, avg_mse, avg_spread = metrics_tensor.tolist()
                        avg_rmse = avg_mse ** 0.5

                    if rank == 0:
                        progress.advance(task)
                        if train_step % args.log_every == 0:
                            elapsed = time.time() - start_time
                            start_time = time.time()
                            logger.info(f"Ep {ep+1} | Step {global_step} | CRPS: {avg_crps:.4f} | RMSE: {avg_rmse:.4f} | Spread: {avg_spread:.4f} | GN: {gn:.2f}")
                        
                        if args.use_wandb:
                            if global_step % args.wandb_every == 0:
                                wandb.log({
                                    "train/crps": avg_crps,
                                    "train/l1": avg_l1,
                                    "train/rmse": avg_rmse,
                                    "train/spread": avg_spread,
                                    "train/spread_rmse_ratio": avg_spread / (avg_rmse + 1e-6),
                                    "train/lr": opt.param_groups[0]["lr"],
                                    "train/grad_norm": gn,
                                    "train/param_norm": pn,
                                    "train/epoch": ep + 1,
                                    "train/step": global_step,
                                    "perf/step_time": elapsed / args.log_every
                                })
                    if train_step % save_interval == 0:
                        save_ckpt(model, opt, ep+1, train_step, avg_crps, args, scheduler)
                
                del loss
                if train_step % 100 == 0:
                    gc.collect()
        dist.barrier()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if rank == 0 and args.use_wandb:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    a = Args()
    run_ddp(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=os.environ.get("MASTER_PORT", "12355"),
        args=a
    )
