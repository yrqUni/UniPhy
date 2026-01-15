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
from ModelUniPhy import UniPhy

warnings.filterwarnings("ignore")

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
        self.ffn_ratio = 1.5
        
        self.Arch = "unet"
        self.ConvType = "dcn"
        self.down_mode = "shuffle"
        self.dist_mode = "diffusion"
        
        self.dt_ref = 1.0
        self.inj_k = 2.0
        
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.lr = 1e-5
        self.weight_decay = 0.05
        self.epochs = 128
        self.grad_clip = 1.0
        self.grad_accum_steps = 1
        
        self.loss = ["lat", "gdl", "spec"]
        self.gdl_every = 1
        self.spec_every = 1
        self.sample_k = 8
        self.T = 6
        
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 16
        self.eval_data_n_frames = 4
        self.use_compile = False
        self.use_amp = False
        self.amp_dtype = "bf16"
        
        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "UniPhy"
        self.wandb_mode = "online"
        self.log_path = "./uniphy_base/logs"
        self.ckpt_dir = "./uniphy_base/ckpt"
        self.check_args()
        
    def check_args(self) -> None:
        if bool(self.use_compile):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("[Warning] Torch Compile is currently disabled/unstable. Forcing use_compile=False.")
            self.use_compile = False
        if bool(self.use_amp):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("[Warning] AMP is currently disabled/unstable. Forcing use_amp=False.")
            self.use_amp = False
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
    log_filename = os.path.join(args.log_path, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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

def apply_model_args(args_obj: Any, model_args_dict: Optional[Dict[str, Any]], verbose: bool = True) -> None:
    if not model_args_dict:
        return
    is_master = not dist.is_initialized() or dist.get_rank() == 0
    for k, v in model_args_dict.items():
        if hasattr(args_obj, k):
            old = getattr(args_obj, k)
            if verbose and old != v and is_master:
                msg = f"[Args] restore '{k}': {old} -> {v}"
                print(msg)
                if dist.is_initialized() and dist.get_rank() == 0:
                    logging.info(msg)
            setattr(args_obj, k, v)

def load_model_args_from_ckpt(ckpt_path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    if not os.path.isfile(ckpt_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Args] ckpt not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model_args = ckpt.get("model_args", None)
    if model_args is None:
        args_all = ckpt.get("args_all", None)
        if isinstance(args_all, dict):
            model_args = {k: args_all[k] for k in MODEL_ARG_KEYS if k in args_all}
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return model_args

def get_prefix(keys: List[str]) -> str:
    if not keys:
        return ""
    key = keys[0]
    if key.startswith("module._orig_mod."):
        return "module._orig_mod."
    if key.startswith("module."):
        return "module."
    return ""

def adapt_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    ckpt_prefix = get_prefix(ckpt_keys)
    model_prefix = get_prefix(model_keys)
    new_state_dict: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        if ckpt_prefix:
            new_k = new_k[len(ckpt_prefix) :]
        if model_prefix:
            new_k = model_prefix + new_k
        new_state_dict[new_k] = v
    return new_state_dict

def save_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int, loss: float, args: Args, scheduler: Optional[Any] = None) -> None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state: Dict[str, Any] = {
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
    ckpt_path = os.path.join(args.ckpt_dir, f"e{epoch}_s{step}_l{state['loss']:.6f}.pth")
    torch.save(state, ckpt_path)
    keep_latest_ckpts(args.ckpt_dir)
    del state
    gc.collect()
    torch.cuda.empty_cache()

def load_ckpt(model: torch.nn.Module, opt: torch.optim.Optimizer, ckpt_path: str, scheduler: Optional[Any] = None, map_location: str = "cpu", args: Optional[Args] = None, restore_model_args: bool = False) -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("No checkpoint Found")
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if restore_model_args and args is not None:
        model_args = checkpoint.get("model_args", None)
        if model_args is None:
            args_all = checkpoint.get("args_all", None)
            if isinstance(args_all, dict):
                model_args = {k: args_all[k] for k in MODEL_ARG_KEYS if k in args_all}
        apply_model_args(args, model_args, verbose=True)
    state_dict = adapt_state_dict_keys(checkpoint["model"], model)
    model.load_state_dict(state_dict, strict=False)
    opt.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    del state_dict
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized() and dist.get_rank() == 0:
        logging.info(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
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
            tag = "gate"
            if "down_blocks" in name:
                parts = name.split(".")
                try:
                    idx = parts[parts.index("down_blocks") + 1]
                    tag = f"d{idx}"
                except Exception:
                    pass
            elif "up_blocks" in name:
                parts = name.split(".")
                try:
                    idx = parts[parts.index("up_blocks") + 1]
                    tag = f"u{idx}"
                except Exception:
                    pass
            if "global" in name:
                tag += "_g"
            elif "lat" in name:
                tag += "_l"
            elif "pw_conv_in" in name:
                tag += "_pw"
            def _hook(mod: torch.nn.Module, inp: Tuple[Any, ...], out: torch.Tensor, tag_local: Any = tag) -> None:
                with torch.no_grad():
                    _LRU_GATE_MEAN[tag_local] = float(out.mean().detach())
            module.register_forward_hook(_hook)

def format_gate_means() -> str:
    if not _LRU_GATE_MEAN:
        return "g=NA"
    keys = sorted(_LRU_GATE_MEAN.keys(), key=lambda k: str(k))
    return " ".join([f"g[{k}]={_LRU_GATE_MEAN[k]:.4f}" for k in keys])

def get_grad_stats(model: torch.nn.Module) -> Tuple[float, float, int]:
    total_norm_sq = 0.0
    max_abs = 0.0
    cnt = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        cnt += 1
        g = p.grad.data
        n = g.norm(2).item()
        total_norm_sq += n * n
        max_abs = max(max_abs, g.abs().max().item())
    return float(total_norm_sq**0.5 if cnt > 0 else 0.0), float(max_abs), int(cnt)

def setup_wandb(rank: int, args: Args) -> None:
    if rank != 0 or not bool(getattr(args, "use_wandb", False)):
        return
    wandb_kwargs: Dict[str, Any] = {"project": args.wandb_project, "config": vars(args)}
    if args.wandb_entity is not None:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name is not None and str(args.wandb_run_name) != "":
        wandb_kwargs["name"] = args.wandb_run_name
    if args.wandb_group is not None:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_mode is not None:
        wandb_kwargs["mode"] = args.wandb_mode
    wandb.init(**wandb_kwargs)

def should_compute(loss_name: str, global_step: int, args: Args) -> bool:
    if loss_name == "gdl":
        return (global_step % int(args.gdl_every)) == 0
    if loss_name == "spec":
        return (global_step % int(args.spec_every)) == 0
    return True

def unwrap_preds(preds_out: Any) -> torch.Tensor:
    if isinstance(preds_out, (tuple, list)):
        return preds_out[0]
    return preds_out

def run_ddp(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, args: Args) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0:
        setup_logging(args)

    if bool(args.use_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_model_args = load_model_args_from_ckpt(args.ckpt, map_location=f"cuda:{local_rank}")
        if ckpt_model_args:
            if rank == 0:
                print("[Args] applying model args from ckpt before building model.")
                logging.info("[Args] applying model args from ckpt before building model.")
            apply_model_args(args, ckpt_model_args, verbose=True)

    model = UniPhy(args).cuda(local_rank)
    if bool(args.use_compile):
        model = torch.compile(model, mode="default")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    register_lru_gate_hooks(model)
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
        num_workers=1,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    )
    len_train_dataloader = len(train_dataloader)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": float(args.weight_decay)},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=float(args.lr))

    scheduler = None
    if bool(args.use_scheduler):
        scheduler = lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(args.lr),
            steps_per_epoch=len_train_dataloader // int(args.grad_accum_steps),
            epochs=int(args.epochs),
        )

    start_epoch = 0
    if args.ckpt and os.path.isfile(args.ckpt):
        start_epoch, _ = load_ckpt(
            model,
            opt,
            args.ckpt,
            scheduler,
            map_location=f"cuda:{local_rank}",
            args=args,
            restore_model_args=False,
        )

    if not bool(args.use_scheduler):
        for g in opt.param_groups:
            g["lr"] = float(args.lr)

    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    use_amp = bool(args.use_amp)
    grad_accum_steps = int(args.grad_accum_steps)
    use_no_sync = bool(args.enable_no_sync)
    dist_mode = str(getattr(args, "dist_mode", "gaussian")).lower()

    global_step = int(start_epoch) * len_train_dataloader
    opt.zero_grad(set_to_none=True)

    T_diffusion = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, T_diffusion).cuda(local_rank)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    for ep in range(int(start_epoch), int(args.epochs)):
        train_sampler.set_epoch(ep)
        train_iter = tqdm(train_dataloader, desc=f"Epoch {ep + 1}/{args.epochs}") if rank == 0 else train_dataloader
        accum_count = 0

        for train_step, data in enumerate(train_iter, start=1):
            model.train()
            B_full, L_full, _, _, _ = data.shape
            sample_k = int(args.sample_k)

            if L_full > sample_k + 1:
                indices = torch.randperm(L_full, device=data.device)[: sample_k + 1]
                indices, _ = torch.sort(indices)
                data_slice = data.index_select(1, indices.cpu())
                dt_indices = indices[1:] - indices[:-1]
                listT_vals = dt_indices.float() * float(args.T)
                listT = listT_vals.unsqueeze(0).repeat(B_full, 1).to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True)
            else:
                data_slice = data
                listT_vals = [float(args.T)] * (data.shape[1] - 1)
                listT = torch.tensor(listT_vals, device=torch.device(f"cuda:{local_rank}"), dtype=torch.float32).view(1, -1).repeat(B_full, 1)

            x = data_slice[:, :-1].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)
            target = data_slice[:, 1:].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)

            x = x.permute(0, 2, 1, 3, 4).contiguous()
            target = target.permute(0, 2, 1, 3, 4).contiguous()

            accum_count += 1
            is_accum_boundary = (accum_count % grad_accum_steps) == 0
            is_last_batch = train_step == len_train_dataloader
            will_step = is_accum_boundary or is_last_batch

            if isinstance(model, DDP) and use_no_sync and not will_step:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = contextlib.nullcontext()

            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    if dist_mode == "diffusion":
                        B, C, L, H, W = target.shape
                        t = torch.randint(0, T_diffusion, (B * L,), device=x.device).long()
                        noise = torch.randn_like(target)
                        
                        target_flat = target.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
                        noise_flat = noise.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
                        
                        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                        x_noisy_flat = sqrt_alpha_t * target_flat + sqrt_one_minus_alpha_t * noise_flat
                        x_noisy = x_noisy_flat.reshape(B, L, C, H, W).permute(0, 2, 1, 3, 4)
                        
                        preds_out = model(x, mode="p", listT=listT, x_noisy=x_noisy, t=t)
                        preds = unwrap_preds(preds_out)
                        loss_main = F.mse_loss(preds, noise)
                        p_det = preds
                    else:
                        preds_out = model(x, mode="p", listT=listT)
                        preds = unwrap_preds(preds_out)
                        target_norm = target
                        if dist_mode == "gaussian":
                            loss_main = gaussian_nll_loss_weighted(preds, target_norm)
                            p_det = preds[:, : target.shape[1]]
                        elif dist_mode == "laplace":
                            loss_main = laplace_nll_loss_weighted(preds, target_norm)
                            p_det = preds[:, : target.shape[1]]
                        else:
                            loss_main = latitude_weighted_l1(preds, target_norm)
                            p_det = preds[:, : target.shape[1]]

                    loss = loss_main
                    gdl_loss = torch.tensor(0.0, device=x.device)
                    spec_loss = torch.tensor(0.0, device=x.device)

                    if dist_mode != "diffusion":
                        if "gdl" in args.loss and should_compute("gdl", global_step + 1, args):
                            gdl_loss = gradient_difference_loss(p_det, target)
                            loss = loss + 0.5 * gdl_loss
                        if "spec" in args.loss and should_compute("spec", global_step + 1, args):
                            spec_loss = spectral_loss(p_det, target)
                            loss = loss + 0.1 * spec_loss

                    if isinstance(model, DDP):
                        dummy_loss = torch.tensor(0.0, device=loss.device)
                        for p in model.parameters():
                            if p.requires_grad:
                                dummy_loss = dummy_loss + p.view(-1)[0].abs() * 0.0
                        loss = loss + dummy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"[Warning] NaN/Inf loss detected at step {global_step}. Skipping step.")
                    opt.zero_grad(set_to_none=True)
                    continue

                (loss / float(grad_accum_steps)).backward()

            if not will_step:
                continue

            grad_norm_pre = 0.0
            grad_norm_post = 0.0
            if float(args.grad_clip) and float(args.grad_clip) > 0:
                norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                grad_norm_pre = norm_tensor.item()
                gn_val, _, _ = get_grad_stats(model)
                grad_norm_post = gn_val
            else:
                gn_val, _, _ = get_grad_stats(model)
                grad_norm_pre = gn_val
                grad_norm_post = gn_val
            
            opt.step()
            opt.zero_grad(set_to_none=True)

            if bool(args.use_scheduler) and scheduler is not None:
                scheduler.step()

            global_step += 1
            is_log_step = (global_step % int(args.log_every)) == 0 or train_step == len_train_dataloader
            is_wandb_step = bool(getattr(args, "use_wandb", False)) and ((global_step % int(args.wandb_every)) == 0 or train_step == len_train_dataloader)

            avg_loss = None
            avg_l1 = None

            if is_log_step or is_wandb_step:
                with torch.no_grad():
                    loss_tensor = loss.detach()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = (loss_tensor / world_size).item()
                    
                    if dist_mode != "diffusion":
                        metric_l1 = F.l1_loss(p_det, target)
                        l1_tensor = metric_l1.detach()
                        dist.all_reduce(l1_tensor, op=dist.ReduceOp.SUM)
                        avg_l1 = (l1_tensor / world_size).item()
                    else:
                        avg_l1 = avg_loss

            if rank == 0:
                if is_log_step:
                    current_lr = scheduler.get_last_lr()[0] if bool(args.use_scheduler) and scheduler is not None else opt.param_groups[0]["lr"]
                    gate_str = format_gate_means()
                    msg = f"Ep {ep + 1} - step {train_step}/{len_train_dataloader} - L: {avg_loss:.4f} - L1: {avg_l1:.4f} - LR: {current_lr:.2e} - {gate_str}"
                    if isinstance(train_iter, tqdm):
                        train_iter.set_description(msg)
                    logging.info(msg)

                if is_wandb_step:
                    current_lr = scheduler.get_last_lr()[0] if bool(args.use_scheduler) and scheduler is not None else opt.param_groups[0]["lr"]
                    log_dict = {
                        "train/epoch": ep + 1,
                        "train/step": int(global_step),
                        "train/loss": float(avg_loss),
                        "train/loss_l1": float(avg_l1),
                        "train/lr": float(current_lr),
                        "train/grad_norm_pre": float(grad_norm_pre),
                        "train/grad_norm_post": float(grad_norm_post),
                    }
                    if dist_mode != "diffusion":
                        log_dict["train/loss_gdl"] = float(gdl_loss.detach().item())
                        log_dict["train/loss_spec"] = float(spec_loss.detach().item())
                    
                    for k, v in _LRU_GATE_MEAN.items():
                        log_dict[f"train/gate_{k}"] = float(v)
                    wandb.log(log_dict, step=int(global_step))

            ckpt_every = max(1, int(len_train_dataloader * float(args.ckpt_step)))
            if (train_step % ckpt_every == 0) or (train_step == len_train_dataloader):
                if rank == 0:
                    loss_for_ckpt = float(avg_loss) if avg_loss is not None else float(loss.detach().item())
                    save_ckpt(model, opt, ep + 1, train_step, loss_for_ckpt, args, scheduler if (bool(args.use_scheduler) and scheduler is not None) else None)

            if (train_step % 50) == 0:
                gc.collect()

        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank == 0 and bool(getattr(args, "use_wandb", False)):
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12355")
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)

