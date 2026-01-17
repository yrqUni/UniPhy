import contextlib
import datetime
import gc
import glob
import logging
import os
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")

MODEL_ARG_KEYS = [
    "in_channels",
    "out_channels",
    "embed_dim",
    "depth",
    "patch_size",
    "img_height",
    "img_width",
    "dropout"
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
    msg = f"\n{'='*40}\nModel Stats:\nTotal: {format_params(total_params)}\nTrainable: {format_params(trainable_params)}\n{'='*40}\n"
    print(msg)
    logging.info(msg)

class Args:
    def __init__(self) -> None:
        self.img_height = 721
        self.img_width = 1440
        self.in_channels = 2
        self.out_channels = 2
        self.embed_dim = 300
        self.patch_size = 16
        self.depth = 6
        self.ensemble_size = 4 
        self.dropout = 0.0
        
        self.dt_ref = 6.0
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.use_scheduler = True
        self.lr = 1e-4 
        self.weight_decay = 0.05
        self.epochs = 100
        self.grad_clip = 1.0
        self.grad_accum_steps = 1
        self.log_every = 1
        self.wandb_every = 1
        self.ckpt_step = 0.5
        self.log_path = "./uniphy_logs"
        self.ckpt_dir = "./uniphy_ckpt"
        self.ckpt = ""
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 8
        self.sample_k = 4
        self.eval_sample_num = 1
        self.use_tf32 = True
        self.use_wandb = True
        self.wandb_project = "ERA5_SDE"
        self.wandb_entity = "UniPhy"
        self.wandb_run_name = self.ckpt
        self.wandb_mode = "online"

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
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")

def setup_wandb(rank: int, args: Args) -> None:
    if rank != 0 or not bool(getattr(args, "use_wandb", False)): return
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

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
    if len(files) > 5:
        files.sort(key=os.path.getmtime)
        for f in files[:-5]: os.remove(f)

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu") -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path): return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def get_grad_stats(model: torch.nn.Module) -> Tuple[float, float]:
    total_norm_sq = 0.0
    max_abs = 0.0
    for p in model.parameters():
        if p.grad is None: continue
        g = p.grad.data
        total_norm_sq += g.norm(2).item() ** 2
        max_abs = max(max_abs, g.abs().max().item())
    return float(total_norm_sq**0.5), float(max_abs)

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
        depth=args.depth,
        patch_size=args.patch_size,
        img_height=args.img_height,
        img_width=args.img_width,
        dropout=args.dropout
    ).cuda(local_rank)

    log_model_stats(model, rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_ds = ERA5_Dataset(
        input_dir=args.data_root, year_range=args.year_range, is_train=True,
        sample_len=args.train_data_n_frames, eval_sample=args.eval_sample_num,
        max_cache_size=8, rank=dist.get_rank(), gpus=dist.get_world_size()
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=False, drop_last=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {"params": [p for p in param_dict.values() if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in param_dict.values() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs) if args.use_scheduler else None

    start_ep, global_step = 0, 0
    if args.ckpt: start_ep, _ = load_ckpt(model, opt, args.ckpt, scheduler, map_location=f"cuda:{local_rank}")

    save_interval = max(1, int(len(train_loader) * args.ckpt_step))

    for ep in range(start_ep, args.epochs):
        train_sampler.set_epoch(ep)
        train_iter = tqdm(train_loader, desc=f"Ep {ep+1}") if rank == 0 else train_loader

        for train_step, data in enumerate(train_iter, start=1):
            model.train()
            data = data.to(f"cuda:{local_rank}", non_blocking=True).float()
            B, T_tot, C, H, W = data.shape

            indices = torch.tensor([sorted(random.sample(range(T_tot), args.sample_k)) for _ in range(B)], device=data.device)
            sampled = data[torch.arange(B, device=data.device).unsqueeze(1), indices]
            x_in, target = sampled[:, :-1], sampled[:, 1:]
            
            dt = (indices[:, 1:] - indices[:, :-1]).float() * args.dt_ref
            
            B_seq, T_seq, C, H, W = target.shape
            
            is_accum = (train_step % args.grad_accum_steps != 0)
            sync_ctx = model.no_sync() if is_accum else contextlib.nullcontext()

            with sync_ctx:
                ensemble_preds = []
                
                m_base = model.module if isinstance(model, DDP) else model
                
                for _ in range(args.ensemble_size):
                    pred = model(x_in, dt)
                    
                    pred_flat = pred.reshape(B_seq * T_seq, C, H, W)
                    ensemble_preds.append(pred_flat)
                
                pred_ensemble = torch.stack(ensemble_preds, dim=1)
                target_flat = target.reshape(B_seq * T_seq, C, H, W)
                
                loss = crps_ensemble_loss(pred_ensemble, target_flat)

                if torch.isnan(loss) or torch.isinf(loss):
                    opt.zero_grad(set_to_none=True)
                    continue

                (loss / args.grad_accum_steps).backward()

            if not is_accum:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                gn, _ = get_grad_stats(model)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
                global_step += 1

                if rank == 0:
                    if train_step % args.log_every == 0:
                        msg = f"Ep {ep+1} | Loss: {loss.item():.4e} | GN: {gn:.2f}"
                        if isinstance(train_iter, tqdm): train_iter.set_description(msg)
                        logging.info(msg)

                    if args.use_wandb and (train_step % args.wandb_every == 0):
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": opt.param_groups[0]["lr"],
                            "train/gn": gn,
                            "train/step": global_step
                        })

                    if train_step % save_interval == 0:
                        save_ckpt(model, opt, ep+1, train_step, loss.item(), args, scheduler)

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

