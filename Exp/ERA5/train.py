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
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")
console = Console()

MODEL_ARG_KEYS = [
    "input_size",
    "input_ch",
    "out_ch",
    "hidden_factor",
    "emb_ch",
    "convlru_num_blocks",
    "down_mode",
    "dist_mode",
    "ffn_ratio",
    "ConvType",
    "Arch",
    "dt_ref",
    "inj_k",
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

class Args:
    def __init__(self) -> None:
        self.input_size = (721, 1440)
        self.input_ch = 30
        self.out_ch = 30
        self.emb_ch = 64
        self.hidden_factor = (7, 12)
        self.convlru_num_blocks = 6
        self.ffn_ratio = 1.25
        
        self.Arch = "unet"
        self.ConvType = "dcn"
        self.down_mode = "shuffle"
        self.dist_mode = "diffusion"
        
        self.dt_ref = 1.0
        self.inj_k = 2.0
        
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.use_scheduler = False
        self.lr = 1e-5
        self.weight_decay = 0.05
        self.epochs = 128
        self.grad_clip = 1.0
        self.grad_accum_steps = 1
        self.log_path = "./uniphy_base/logs"
        self.ckpt_dir = "./uniphy_base/ckpt"
        self.ckpt = ""
        self.ckpt_step = 0.25
        self.log_every = 1
        self.wandb_every = 1
        self.image_log_every = 500
        
        self.loss = ["lat", "gdl", "spec"]
        self.gdl_every = 1
        self.spec_every = 1
        self.sample_k = 8
        self.T = 6
        
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 16
        self.eval_data_n_frames = 4
        self.eval_sample_num = 1
        self.use_compile = False
        self.use_amp = False
        self.amp_dtype = "bf16"
        self.use_tf32 = False

        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "UniPhy"
        self.wandb_run_name = self.ckpt
        self.wandb_mode = "online"
        self.check_args()
        
    def check_args(self) -> None:
        if bool(self.use_compile):
            self.use_compile = False
        if bool(self.use_amp):
            self.use_amp = False

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            logging.FileHandler(os.path.join(args.log_path, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        ]
    )

def keep_latest_ckpts(ckpt_dir: str) -> None:
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(ckpt_files) <= 10:
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    for file_path in ckpt_files[10:]:
        try:
            os.remove(file_path)
        except Exception:
            pass

def extract_model_args(args_obj: Any) -> Dict[str, Any]:
    return {k: getattr(args_obj, k) for k in MODEL_ARG_KEYS if hasattr(args_obj, k)}

def save_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, loss: float, args: Args, scheduler: Optional[Any] = None) -> None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": opt.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "loss": float(loss),
        "args_all": dict(vars(args)),
        "model_args": extract_model_args(args),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    ckpt_path = os.path.join(args.ckpt_dir, f"e{epoch}_s{step}_l{loss:.6f}.pth")
    torch.save(state, ckpt_path)
    keep_latest_ckpts(args.ckpt_dir)
    gc.collect()

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu") -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    return epoch, step

_LAT_WEIGHT_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

def get_latitude_weights(H: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (H, device, dtype)
    if key in _LAT_WEIGHT_CACHE:
        return _LAT_WEIGHT_CACHE[key]
    lat_edges = torch.linspace(-90, 90, steps=H + 1, device=device, dtype=dtype)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    w = torch.cos(lat_centers * torch.pi / 180.0).clamp_min(0)
    w = w / w.mean()
    _LAT_WEIGHT_CACHE[key] = w
    return w

def gaussian_nll_loss_weighted(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, _, _, H, _ = preds.shape
    C_gt = targets.size(1)
    mu = preds[:, :C_gt]
    sigma = preds[:, C_gt:]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    var = sigma.pow(2)
    nll = 0.5 * (torch.log(var) + (targets - mu).pow(2) / var)
    return (nll * w).mean()

def laplace_nll_loss_weighted(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, _, _, H, _ = preds.shape
    C_gt = targets.size(1)
    mu = preds[:, :C_gt]
    b = preds[:, C_gt:]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    nll = torch.log(2 * b) + torch.abs(targets - mu) / b
    return (nll * w).mean()

def latitude_weighted_l1(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, C_pred, _, H, _ = preds.shape
    C_gt = targets.shape[1]
    if C_pred > C_gt:
        preds = preds[:, :C_gt]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    return ((preds - targets).abs() * w).mean()

def gradient_difference_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    diff_y_pred = preds[:, :, :, 1:, :] - preds[:, :, :, :-1, :]
    diff_y_gt = targets[:, :, :, 1:, :] - targets[:, :, :, :-1, :]
    loss_y = torch.mean(torch.abs(diff_y_pred - diff_y_gt))
    diff_x_pred = preds[:, :, :, :, 1:] - preds[:, :, :, :, :-1]
    diff_x_gt = targets[:, :, :, :, 1:] - targets[:, :, :, :, :-1]
    loss_x = torch.mean(torch.abs(diff_x_pred - diff_x_gt))
    return loss_x + loss_y

def spectral_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    fft_pred = torch.fft.rfft2(preds.float(), norm="ortho")
    fft_target = torch.fft.rfft2(targets.float(), norm="ortho")
    return (fft_pred.abs() - fft_target.abs()).abs().mean()

_LRU_GATE_MEAN: Dict[Any, float] = {}

def register_lru_gate_hooks(ddp_model: torch.nn.Module) -> None:
    model_to_hook = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    for name, module in model_to_hook.named_modules():
        if isinstance(module, torch.nn.Sigmoid):
            tag = name.split(".")[-1]
            def _hook(mod: torch.nn.Module, inp: Tuple[Any, ...], out: torch.Tensor, tag_local: Any = tag) -> None:
                with torch.no_grad():
                    _LRU_GATE_MEAN[tag_local] = float(out.mean().detach())
            module.register_forward_hook(_hook)

def format_gate_means() -> str:
    if not _LRU_GATE_MEAN: return "g=NA"
    keys = sorted(_LRU_GATE_MEAN.keys())
    return " ".join([f"{k}:{_LRU_GATE_MEAN[k]:.3f}" for k in keys])

def get_grad_stats(model: torch.nn.Module) -> Tuple[float, float, int]:
    total_norm_sq = 0.0
    max_abs = 0.0
    cnt = 0
    for p in model.parameters():
        if p.grad is None: continue
        cnt += 1
        g = p.grad.data
        total_norm_sq += g.norm(2).item() ** 2
        max_abs = max(max_abs, g.abs().max().item())
    return float(total_norm_sq**0.5), float(max_abs), int(cnt)

def setup_wandb(rank: int, args: Args) -> None:
    if rank != 0 or not bool(args.use_wandb): return
    wandb_settings = wandb.Settings(init_timeout=600, _disable_stats=True, _disable_meta=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        mode=args.wandb_mode,
        settings=wandb_settings,
        force=True
    )

def log_vis_images(preds, target, step, rank):
    if rank != 0: return
    dist.barrier()
    p_img = preds[0, 0, -1].detach().cpu().numpy()
    t_img = target[0, 0, -1].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(t_img, cmap='RdBu_r'); axes[0].set_title("Target"); fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(p_img, cmap='RdBu_r'); axes[1].set_title("Pred"); fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(np.abs(p_img - t_img), cmap='viridis'); axes[2].set_title("Diff"); fig.colorbar(im2, ax=axes[2])
    wandb.log({"val/visualization": wandb.Image(fig)}, step=step)
    plt.close(fig)
    dist.barrier()

def unwrap_preds(preds_out: Any) -> torch.Tensor:
    if isinstance(preds_out, (tuple, list)): return preds_out[0]
    return preds_out

def run_ddp(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, args: Args) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0: setup_logging(args)
    setup_wandb(rank, args)

    if bool(args.use_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(args).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    register_lru_gate_hooks(model)

    train_dataset = ERA5_Dataset(
        input_dir=args.data_root, year_range=args.year_range, is_train=True,
        sample_len=args.train_data_n_frames, eval_sample=args.eval_sample_num
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=float(args.lr), steps_per_epoch=len(train_loader)//int(args.grad_accum_steps), epochs=int(args.epochs)) if bool(args.use_scheduler) else None

    start_ep, global_step = 0, 0
    if args.ckpt: start_ep, global_step = load_ckpt(model, opt, args.ckpt, scheduler, f"cuda:{local_rank}")

    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    use_amp = bool(args.use_amp)
    dist_mode = str(args.dist_mode).lower()

    T_diffusion, betas = 1000, torch.linspace(1e-4, 0.02, 1000).cuda(local_rank)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    for ep in range(int(start_ep), int(args.epochs)):
        train_sampler.set_epoch(ep)
        with Progress(
            TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=None), MofNCompleteColumn(),
            TaskProgressColumn(), TimeRemainingColumn(), console=console, disable=(rank != 0)
        ) as progress:
            task = progress.add_task(f"Epoch {ep+1}", total=len(train_loader))
            for t_step, data in enumerate(train_loader, 1):
                model.train()
                B, L_tot = data.shape[0], data.shape[1]
                idx = torch.randperm(L_tot, device=data.device)[:int(args.sample_k)+1].sort()[0]
                data_s = data.index_select(1, idx.cpu())
                listT = (idx[1:] - idx[:-1]).float().unsqueeze(0).repeat(B, 1).cuda(local_rank) * float(args.T)
                
                x = data_s[:, :-1].cuda(local_rank, non_blocking=True).float().permute(0, 2, 1, 3, 4).contiguous()
                target = data_s[:, 1:].cuda(local_rank, non_blocking=True).float().permute(0, 2, 1, 3, 4).contiguous()

                is_accum = (t_step % int(args.grad_accum_steps) != 0)
                with (model.no_sync() if is_accum else contextlib.nullcontext()):
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if dist_mode == "diffusion":
                            B_d, C_d, L_d, H_d, W_d = target.shape
                            t_idx = torch.randint(0, T_diffusion, (B_d * L_d,), device=x.device).long()
                            noise = torch.randn_like(target)
                            t_flat = target.permute(0, 2, 1, 3, 4).reshape(B_d * L_d, C_d, H_d, W_d)
                            n_flat = noise.permute(0, 2, 1, 3, 4).reshape(B_d * L_d, C_d, H_d, W_d)
                            x_n = (sqrt_alphas_cumprod[t_idx].view(-1,1,1,1)*t_flat + sqrt_one_minus_alphas_cumprod[t_idx].view(-1,1,1,1)*n_flat).reshape(B_d,L_d,C_d,H_d,W_d).permute(0,2,1,3,4)
                            preds = unwrap_preds(model(x, mode="p", listT=listT, x_noisy=x_n, t=t_idx))
                            loss_m = F.mse_loss(preds, noise); p_det = preds
                        else:
                            preds = unwrap_preds(model(x, mode="p", listT=listT))
                            if dist_mode == "gaussian": loss_m = gaussian_nll_loss_weighted(preds, target)
                            elif dist_mode == "laplace": loss_m = laplace_nll_loss_weighted(preds, target)
                            else: loss_m = latitude_weighted_l1(preds, target)
                            p_det = preds[:, :target.shape[1]]

                        loss = loss_m
                        g_l = torch.tensor(0.0, device=x.device); s_l = torch.tensor(0.0, device=x.device)
                        if dist_mode != "diffusion":
                            if "gdl" in args.loss: g_l = gradient_difference_loss(p_det, target); loss += 0.5 * g_l
                            if "spec" in args.loss: s_l = spectral_loss(p_det, target); loss += 0.1 * s_l
                        
                    (loss / int(args.grad_accum_steps)).backward()

                if not is_accum:
                    if float(args.grad_clip) > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    opt.step(); opt.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    global_step += 1

                    with torch.no_grad():
                        l_t = loss.detach(); dist.all_reduce(l_t, op=dist.ReduceOp.SUM)
                        avg_l = (l_t / world_size).item()
                        if dist_mode != "diffusion":
                            l1_t = F.l1_loss(p_det, target).detach(); dist.all_reduce(l1_t, op=dist.ReduceOp.SUM)
                            avg_l1 = (l1_t / world_size).item()
                        else: avg_l1 = avg_l

                    if rank == 0:
                        progress.advance(task)
                        if global_step % int(args.log_every) == 0:
                            logging.info(f"Step {global_step} | L: {avg_l:.4f} | L1: {avg_l1:.4f} | {format_gate_means()}")
                        if bool(args.use_wandb) and global_step % int(args.wandb_every) == 0:
                            wandb.log({"train/loss": avg_l, "train/l1": avg_l1, "train/lr": opt.param_groups[0]["lr"]}, step=global_step)
                        if bool(args.use_wandb) and global_step % int(args.image_log_every) == 0:
                            log_vis_images(p_det, target, global_step, rank)

                if t_step % max(1, int(len(train_loader)*float(args.ckpt_step))) == 0:
                    if rank == 0: save_ckpt(model, opt, ep+1, t_step, avg_l if 'avg_l' in locals() else loss.item(), args, scheduler)
                if t_step % 100 == 0: gc.collect()

        dist.barrier()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if rank == 0 and bool(args.use_wandb): wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    run_ddp(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"]), os.environ.get("MASTER_ADDR", "127.0.0.1"), os.environ.get("MASTER_PORT", "12355"), args)

