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

sys.path.append("/nfs/ConvLRU/Model/UniPhy")
sys.path.append("/nfs/ConvLRU/Exp/ERA5")

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
    "lru_rank",
    "down_mode",
    "dist_mode",
    "ffn_ratio",
    "ConvType",
    "Arch",
    "koopman_use_noise",
    "koopman_noise_scale",
    "dt_ref",
    "inj_k",
    "dynamics_mode",
    "interpolation_mode",
    "spectral_modes_h",
    "spectral_modes_w",
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
        self.hidden_factor = (7, 12)
        self.emb_ch = 64
        self.convlru_num_blocks = 6
        self.lru_rank = 64
        self.down_mode = "shuffle"
        self.dist_mode = "gaussian"
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 27
        self.eval_data_n_frames = 4
        self.eval_sample_num = 1
        self.ckpt = ""
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.epochs = 128
        self.log_path = "./uniphy_base/logs"
        self.ckpt_dir = "./uniphy_base/ckpt"
        self.ckpt_step = 0.25
        self.do_eval = False
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-5
        self.weight_decay = 0.05
        self.use_scheduler = False
        self.loss = ["lat", "gdl", "spec"]
        self.gdl_every = 1
        self.spec_every = 1
        self.T = 6
        self.use_amp = False
        self.amp_dtype = "bf16"
        self.grad_clip = 1.0
        self.sample_k = 9
        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "ConvLRU"
        self.wandb_run_name = ""
        self.wandb_group = ""
        self.wandb_mode = "online"
        self.ffn_ratio = 1.5
        self.ConvType = "dcn"
        self.Arch = "unet"
        self.grad_accum_steps = 1
        self.enable_no_sync = True
        self.log_every = 1
        self.wandb_every = 1
        self.koopman_use_noise = True
        self.koopman_noise_scale = 1.0
        self.dt_ref = 1.0
        self.inj_k = 2.0
        self.max_velocity = 5.0
        self.dynamics_mode = "advection"
        self.interpolation_mode = "bilinear"
        self.spectral_modes_h = 12
        self.spectral_modes_w = 12
        self.check_args()

    def check_args(self) -> None:
        if bool(self.use_compile):
            print("[Warning] Torch Compile is currently disabled/unstable. Forcing use_compile=False.")
            self.use_compile = False
        
        if bool(self.use_amp):
            print("[Warning] AMP is currently disabled/unstable. Forcing use_amp=False.")
            self.use_amp = False

        if int(self.grad_accum_steps) < 1:
            raise ValueError("grad_accum_steps must be >= 1")


def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: str, local_rank: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=1800),
    )
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(args: Args) -> None:
    if not dist.is_initialized() or dist.get_rank() != 0:
        return
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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


def apply_model_args(args_obj: Any, model_args_dict: Optional[Dict[str, Any]], verbose: bool = True) -> None:
    if not model_args_dict:
        return
    for k, v in model_args_dict.items():
        if hasattr(args_obj, k):
            old = getattr(args_obj, k)
            if verbose and old != v:
                msg = f"[Args] restore '{k}': {old} -> {v}"
                print(msg)
                if dist.is_initialized() and dist.get_rank() == 0:
                    logging.info(msg)
            setattr(args_obj, k, v)


def load_model_args_from_ckpt(ckpt_path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    if not os.path.isfile(ckpt_path):
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


def load_ckpt(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    ckpt_path: str,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    args: Optional[Args] = None,
    restore_model_args: bool = False,
) -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
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
    C_gt = targets.size(2)
    mu = preds[:, :, :C_gt]
    sigma = preds[:, :, C_gt:]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    var = sigma.pow(2)
    nll = 0.5 * (torch.log(var) + (targets - mu).pow(2) / var)
    return (nll * w).mean()


def laplace_nll_loss_weighted(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, _, _, H, _ = preds.shape
    C_gt = targets.size(2)
    mu = preds[:, :, :C_gt]
    b = preds[:, :, C_gt:]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    nll = torch.log(2 * b) + torch.abs(targets - mu) / b
    return (nll * w).mean()


def latitude_weighted_l1(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, _, C_pred, H, _ = preds.shape
    C_gt = targets.shape[2]
    if C_pred > C_gt:
        preds = preds[:, :, :C_gt]
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
        if isinstance(module, torch.nn.Sigmoid) and "gate" in name:
            tag = "unknown"
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


def make_listT_from_arg_T(B: int, L: int, device: torch.device, dtype: torch.dtype, T: Optional[float]) -> Optional[torch.Tensor]:
    if T is None or T < 0:
        return None
    return torch.full((B, L), float(T), device=device, dtype=dtype)


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


def get_revin(ddp_model: torch.nn.Module) -> Any:
    return ddp_model.module.revin if isinstance(ddp_model, DDP) else ddp_model.revin


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
            print("[Args] applying model args from ckpt before building model.")
            if rank == 0:
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

    revin_mod = get_revin(model)

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
                    revin_stats = revin_mod.stats(x)

                    preds_out = model(x, mode="p", listT=listT, revin_stats=revin_stats)
                    preds = unwrap_preds(preds_out)

                    C_gt = target.shape[2]
                    target_real = revin_mod(target, "denorm", stats=revin_stats)

                    if dist_mode == "gaussian":
                        loss_main = gaussian_nll_loss_weighted(preds, target_real)
                        p_det = preds[:, :, :C_gt]
                    elif dist_mode == "laplace":
                        mu = preds[:, :, :C_gt]
                        b = preds[:, :, C_gt:]
                        mu_denorm = revin_mod(mu, "denorm", stats=revin_stats) if mu.shape[2] == revin_mod.num_features else mu
                        b_denorm = b * revin_stats.stdev if b.shape[2] == revin_mod.num_features else b
                        preds_denorm = torch.cat([mu_denorm, b_denorm], dim=2)
                        loss_main = laplace_nll_loss_weighted(preds_denorm, target_real)
                        p_det = mu_denorm
                    else:
                        loss_main = latitude_weighted_l1(preds, target_real)
                        p_det = preds[:, :, :C_gt] if preds.shape[2] >= C_gt else preds

                    loss = torch.tensor(0.0, device=x.device)
                    if "lat" in args.loss:
                        loss = loss + loss_main

                    gdl_loss = torch.tensor(0.0, device=x.device)
                    if "gdl" in args.loss and should_compute("gdl", global_step + 1, args):
                        gdl_loss = gradient_difference_loss(p_det, target_real)
                        loss = loss + 0.5 * gdl_loss

                    spec_loss = torch.tensor(0.0, device=x.device)
                    if "spec" in args.loss and should_compute("spec", global_step + 1, args):
                        spec_loss = spectral_loss(p_det, target_real)
                        loss = loss + 0.1 * spec_loss

                    if isinstance(model, DDP):
                        dummy_loss = torch.tensor(0.0, device=loss.device)
                        for p in model.parameters():
                            if p.requires_grad:
                                dummy_loss = dummy_loss + p.view(-1)[0].abs() * 0.0
                        loss = loss + dummy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] NaN/Inf loss detected at step {global_step}. Skipping step.")
                    if rank == 0:
                        logging.warning(f"NaN/Inf loss detected at step {global_step}")
                    opt.zero_grad(set_to_none=True)
                    continue

                (loss / float(grad_accum_steps)).backward()

            if not will_step:
                del data, data_slice, x, target, preds_out, preds, listT, loss, loss_main, gdl_loss, spec_loss, p_det, target_real
                continue

            if float(args.grad_clip) and float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            opt.step()
            opt.zero_grad(set_to_none=True)

            if bool(args.use_scheduler) and scheduler is not None:
                scheduler.step()

            global_step += 1

            is_log_step = ((global_step % int(args.log_every)) == 0 or train_step == len_train_dataloader)
            is_wandb_step = bool(getattr(args, "use_wandb", False)) and ((global_step % int(args.wandb_every)) == 0 or train_step == len_train_dataloader)

            avg_loss = None
            avg_l1 = None

            if is_log_step or is_wandb_step:
                with torch.no_grad():
                    metric_l1 = F.l1_loss(p_det, target_real)
                    loss_tensor = loss.detach()
                    l1_tensor = metric_l1.detach()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(l1_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = (loss_tensor / world_size).item()
                    avg_l1 = (l1_tensor / world_size).item()

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
                    grad_norm, _, _ = get_grad_stats(model)
                    log_dict: Dict[str, Any] = {
                        "train/epoch": ep + 1,
                        "train/step": int(global_step),
                        "train/loss": float(avg_loss),
                        "train/loss_l1": float(avg_l1),
                        "train/loss_gdl": float(gdl_loss.detach().item()),
                        "train/loss_spec": float(spec_loss.detach().item()),
                        "train/lr": float(current_lr),
                        "train/grad_norm": float(grad_norm),
                    }
                    for k, v in _LRU_GATE_MEAN.items():
                        g_key = f"train/gate_{k}"
                        log_dict[g_key] = float(v)
                    wandb.log(log_dict, step=int(global_step))

            ckpt_every = max(1, int(len_train_dataloader * float(args.ckpt_step)))
            if (train_step % ckpt_every == 0) or (train_step == len_train_dataloader):
                loss_for_ckpt = float(avg_loss) if avg_loss is not None else float(loss.detach().item())
                save_ckpt(
                    model,
                    opt,
                    ep + 1,
                    train_step,
                    loss_for_ckpt,
                    args,
                    scheduler if (bool(args.use_scheduler) and scheduler is not None) else None,
                )

            del data, data_slice, x, target, preds_out, preds, listT, loss, loss_main, gdl_loss, spec_loss, p_det, target_real
            if (train_step % 50) == 0:
                gc.collect()

        dist.barrier()

        if bool(args.do_eval):
            model.eval()
            eval_dataset = ERA5_Dataset(
                input_dir=args.data_root,
                year_range=args.year_range,
                is_train=False,
                sample_len=args.eval_data_n_frames,
                eval_sample=args.eval_sample_num,
                max_cache_size=8,
                rank=dist.get_rank(),
                gpus=dist.get_world_size(),
            )
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=True)
            eval_dataloader = DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
                num_workers=1,
                pin_memory=True,
                prefetch_factor=1,
                persistent_workers=False,
            )
            eval_iter = tqdm(eval_dataloader, desc=f"Eval Epoch {ep + 1}/{args.epochs}") if rank == 0 else eval_dataloader
            with torch.no_grad():
                for eval_step, data in enumerate(eval_iter, start=1):
                    B_full, L_full, _, _, _ = data.shape
                    half = int(args.eval_data_n_frames) // 2
                    cond_data = data[:, :half].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)

                    listT_cond_vals = [float(args.T)] * cond_data.shape[1]
                    listT_cond = torch.tensor(listT_cond_vals, device=cond_data.device, dtype=cond_data.dtype).view(1, -1).repeat(cond_data.size(0), 1)

                    out_gen_num = int(L_full - cond_data.shape[1])
                    listT_future = make_listT_from_arg_T(B_full, out_gen_num, cond_data.device, cond_data.dtype, float(args.T))
                    target = data[:, cond_data.shape[1] : cond_data.shape[1] + out_gen_num].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        revin_stats = revin_mod.stats(cond_data)
                        preds_out = model(
                            cond_data,
                            mode="i",
                            out_gen_num=out_gen_num,
                            listT=listT_cond,
                            listT_future=listT_future,
                            revin_stats=revin_stats,
                        )
                        preds = unwrap_preds(preds_out)

                        target_real = revin_mod(target, "denorm", stats=revin_stats)

                        if preds.shape[2] >= 2 * target.shape[2]:
                            preds_cmp = preds[:, :, : target.shape[2]]
                        else:
                            preds_cmp = preds[:, :, : target.shape[2]] if preds.shape[2] >= target.shape[2] else preds

                        loss_eval = F.l1_loss(preds_cmp, target_real)

                    tot_tensor = loss_eval.detach()
                    dist.all_reduce(tot_tensor, op=dist.ReduceOp.SUM)
                    avg_total = (tot_tensor / world_size).item()

                    if rank == 0:
                        message = f"Eval step {eval_step} - L1: {avg_total:.6f}"
                        if isinstance(eval_iter, tqdm):
                            eval_iter.set_description(message)
                        logging.info(message)
                        if bool(getattr(args, "use_wandb", False)):
                            wandb.log({"eval/l1_loss": avg_total, "eval/epoch": ep + 1}, step=int(global_step))

                    del data, target, cond_data, preds_out, preds, loss_eval, tot_tensor, listT_cond, listT_future
                    if eval_step % 20 == 0:
                        gc.collect()

            del eval_dataset, eval_sampler, eval_dataloader, eval_iter
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

