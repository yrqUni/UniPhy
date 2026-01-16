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
    "input_shape",
    "in_channels",
    "dim",
    "patch_size",
    "num_layers",
    "para_pool_expansion",
    "conserve_energy"
]

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
    if num >= 1e9:
        readable = f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        readable = f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        readable = f"{num / 1e3:.2f}K"
    else:
        readable = f"{num}"
    return f"{num:,} ({readable})"

def log_model_stats(model: torch.nn.Module, rank: int) -> None:
    if rank != 0:
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    msg = (
        f"\n{'='*40}\n"
        f"Model Statistics:\n"
        f"Total Parameters:     {format_params(total_params)}\n"
        f"Trainable Parameters: {format_params(trainable_params)}\n"
        f"{'='*40}\n"
    )
    print(msg)
    logging.info(msg)

class Args:
    def __init__(self) -> None:
        self.input_shape = (721, 1440)
        self.in_channels = 30
        self.dim = 1536
        self.patch_size = 4
        self.num_layers = 24
        self.para_pool_expansion = 4
        self.conserve_energy = True
        
        self.dt_ref = 1.0
        self.train_batch_size = 1
        self.eval_batch_size = 1
        
        self.use_scheduler = True
        self.lr = 1e-4
        self.weight_decay = 0.05
        self.epochs = 100
        
        self.grad_clip = 1.0
        self.grad_accum_steps = 1
        
        self.log_every = 10
        self.wandb_every = 10
        self.ckpt_step = 0.5
        
        self.log_path = "./uniphy_logs"
        self.ckpt_dir = "./uniphy_ckpt"
        self.ckpt = ""
        
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 4
        self.eval_sample_num = 1
        
        self.use_tf32 = True

        self.use_wandb = True
        self.wandb_project = "UniPhy-ERA5"
        self.wandb_entity = "UniPhy"
        self.wandb_run_name = "UniPhy_Large_A100"
        self.wandb_mode = "online"

        self.check_args()
        
    def check_args(self) -> None:
        if int(self.grad_accum_steps) < 1:
            raise ValueError("grad_accum_steps must be >= 1")

def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str, local_rank: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(local_rank)

def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(args: Args) -> None:
    if not dist.is_initialized() or dist.get_rank() != 0:
        return
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def setup_wandb(rank: int, args: Args) -> None:
    if rank != 0 or not bool(getattr(args, "use_wandb", False)):
        return
    wandb_kwargs: Dict[str, Any] = {"project": args.wandb_project, "config": vars(args)}
    if args.wandb_entity: wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name: wandb_kwargs["name"] = args.wandb_run_name
    wandb.init(**wandb_kwargs)

def keep_latest_ckpts(ckpt_dir: str) -> None:
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(ckpt_files) <= 5: return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    for file_path in ckpt_files[5:]:
        try: os.remove(file_path)
        except Exception: pass

def extract_model_args(args_obj: Any) -> Dict[str, Any]:
    return {k: getattr(args_obj, k) for k in MODEL_ARG_KEYS if hasattr(args_obj, k)}

def save_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, loss: float, args: Args, scheduler: Optional[Any] = None) -> None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state: Dict[str, Any] = {
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
    keep_latest_ckpts(args.ckpt_dir)

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu") -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path): return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def get_grad_stats(model: torch.nn.Module) -> Tuple[float, float]:
    total_norm_sq = 0.0
    max_abs = 0.0
    for p in model.parameters():
        if p.grad is None: continue
        g = p.grad.data
        n = g.norm(2).item()
        total_norm_sq += n * n
        max_abs = max(max_abs, g.abs().max().item())
    return float(total_norm_sq**0.5), float(max_abs)

def run_ddp(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, args: Args) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0: setup_logging(args)

    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(
        input_shape=args.input_shape,
        in_channels=args.in_channels,
        dim=args.dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        para_pool_expansion=args.para_pool_expansion,
        conserve_energy=args.conserve_energy
    ).cuda(local_rank)

    log_model_stats(model, rank)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    setup_wandb(rank, args)

    train_dataset = ERA5_Dataset(
        input_dir=args.data_root,
        year_range=args.year_range,
        is_train=True,
        sample_len=args.train_data_n_frames,
        eval_sample=args.eval_sample_num,
        max_cache_size=8,
        rank=dist.get_rank(),
        gpus=dist.get_world_size(),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr)

    scheduler = None
    if args.use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(
            opt,
            max_lr=args.lr,
            steps_per_epoch=len(train_dataloader) // args.grad_accum_steps,
            epochs=args.epochs,
        )

    start_epoch = 0
    if args.ckpt:
        start_epoch, _ = load_ckpt(
            model, opt, args.ckpt, scheduler, map_location=f"cuda:{local_rank}"
        )

    T_diffusion = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, T_diffusion).cuda(local_rank)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    global_step = int(start_epoch) * len(train_dataloader)
    
    for ep in range(start_epoch, args.epochs):
        train_sampler.set_epoch(ep)
        train_iter = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{args.epochs}") if rank == 0 else train_dataloader
        
        for train_step, data in enumerate(train_iter, start=1):
            model.train()
            
            x = data[:, :-1].to(f"cuda:{local_rank}", non_blocking=True).float()
            target = data[:, 1:].to(f"cuda:{local_rank}", non_blocking=True).float()
            
            B, T_seq, C, H, W = target.shape
            dt = torch.full((B, T_seq), args.dt_ref, device=x.device)

            is_accum = (train_step % args.grad_accum_steps != 0)
            sync_ctx = model.no_sync() if is_accum else contextlib.nullcontext()

            with sync_ctx:
                z_pred, _ = model(x, dt)
                
                z_flat = z_pred.reshape(B * T_seq, -1, z_pred.shape[-2], z_pred.shape[-1])
                target_flat = target.reshape(B * T_seq, C, H, W)
                
                t = torch.randint(0, T_diffusion, (B * T_seq,), device=x.device).long()
                noise = torch.randn_like(target_flat)
                
                sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                x_noisy = sqrt_alpha_t * target_flat + sqrt_one_minus_alpha_t * noise
                
                decoder_module = model.module.decoder if isinstance(model, DDP) else model.decoder
                pred_noise = decoder_module(z_flat, x_noisy, t)
                
                loss = F.mse_loss(pred_noise, noise)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    opt.zero_grad(set_to_none=True)
                    continue

                (loss / args.grad_accum_steps).backward()

            if is_accum:
                continue

            grad_norm, max_grad = 0.0, 0.0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            grad_norm, max_grad = get_grad_stats(model)
            
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
            
            global_step += 1

            if rank == 0:
                is_log = (train_step % args.log_every == 0)
                if is_log:
                    loss_val = loss.item()
                    lr_val = opt.param_groups[0]["lr"]
                    msg = f"Ep {ep+1} - {train_step}/{len(train_dataloader)} | Loss: {loss_val:.4e} | GN: {grad_norm:.2f}"
                    if isinstance(train_iter, tqdm): train_iter.set_description(msg)
                    logging.info(msg)
                    
                    if args.use_wandb and (train_step % args.wandb_every == 0):
                        wandb.log({
                            "train/loss": loss_val,
                            "train/lr": lr_val,
                            "train/grad_norm": grad_norm,
                            "train/step": global_step
                        })

                ckpt_steps = int(len(train_dataloader) * args.ckpt_step)
                if train_step % ckpt_steps == 0:
                    save_ckpt(model, opt, ep+1, train_step, loss.item(), args, scheduler)

    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12355")
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)

