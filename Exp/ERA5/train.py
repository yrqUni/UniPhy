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

sys.path.append("/nfs/ConvLRU/Model/ConvLRU")
sys.path.append("/nfs/ConvLRU/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelConvLRU import ConvLRU

warnings.filterwarnings("ignore")

MODEL_ARG_KEYS = [
    "input_size",
    "input_ch",
    "out_ch",
    "hidden_factor",
    "emb_ch",
    "convlru_num_blocks",
    "use_cbam",
    "lru_rank",
    "use_freq_prior",
    "use_sh_prior",
    "sh_Lmax",
    "sh_rank",
    "sh_gain_init",
    "num_expert",
    "activate_expert",
    "static_ch",
    "down_mode",
    "head_mode",
    "use_checkpointing",
    "use_spectral_mixing",
    "use_advection",
    "use_spatial_ssm",
    "use_stochastic",
    "learnable_init_state",
    "use_wavelet_ssm",
    "use_cross_var_attn",
    "use_graph_interaction",
    "ffn_ratio",
    "ConvType",
    "Arch",
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
        self.static_ch = 6
        self.hidden_factor = (7, 12)
        self.emb_ch = 96
        self.convlru_num_blocks = 6
        self.use_cbam = True
        self.num_expert = 16
        self.activate_expert = 4
        self.lru_rank = 32
        self.down_mode = "shuffle"
        self.use_freq_prior = True
        self.use_sh_prior = True
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.head_mode = "gaussian"
        self.diffusion_steps = 1000
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 27
        self.eval_data_n_frames = 4
        self.eval_sample_num = 1
        self.ckpt = "e7_s570_l0.265707.pth"
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.epochs = 128
        self.log_path = "./convlru_base/logs"
        self.ckpt_dir = "./convlru_base/ckpt"
        self.ckpt_step = 0.25
        self.do_eval = False
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-5
        self.weight_decay = 0.05
        self.use_scheduler = False
        self.init_lr_scheduler = False
        self.loss = ["lat", "gdl", "spec"]
        self.T = 6
        self.use_amp = False
        self.amp_dtype = "bf16"
        self.grad_clip = 1.0
        self.sample_k = 9
        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "ConvLRU"
        self.wandb_run_name = "PhyConvLRU_HKLF"
        self.wandb_group = "v4.0.0"
        self.wandb_mode = "online"
        self.use_checkpointing = True
        self.train_mode = "alignment"
        self.use_spectral_mixing = True
        self.use_advection = True
        self.use_spatial_ssm = True
        self.use_stochastic = False
        self.learnable_init_state = True
        self.use_wavelet_ssm = True
        self.use_cross_var_attn = True
        self.use_graph_interaction = False
        self.ffn_ratio = 1.0
        self.ConvType = "dcn"
        self.Arch = "bifpn"
        self.check_args()

    def check_args(self) -> None:
        if bool(self.use_compile):
            print("[Warning] Torch Compile is experimental.")

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
    if len(ckpt_files) <= 64:
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    for file_path in ckpt_files[64:]:
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
    B, L, C2, H, W = preds.shape
    C_gt = targets.size(2)
    mu = preds[:, :, :C_gt]
    sigma = preds[:, :, C_gt:]
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    var = sigma.pow(2)
    nll = 0.5 * (torch.log(var) + (targets - mu).pow(2) / var)
    return (nll * w).mean()

def latitude_weighted_l1(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, _, C_pred, H, _ = preds.shape
    _, _, C_gt, _, _ = targets.shape
    if C_pred == 2 * C_gt:
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

def get_moe_aux_loss(model: torch.nn.Module) -> torch.Tensor:
    total_aux_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    model_to_search = model.module if isinstance(model, DDP) else model
    for module in model_to_search.modules():
        if hasattr(module, "aux_loss") and isinstance(module.aux_loss, torch.Tensor):
            total_aux_loss = total_aux_loss + module.aux_loss
    return total_aux_loss

def get_kl_loss(model: torch.nn.Module) -> torch.Tensor:
    total_kl_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    model_to_search = model.module if isinstance(model, DDP) else model
    if hasattr(model_to_search, "get_total_kl_loss"):
        return model_to_search.get_total_kl_loss()
    return total_kl_loss

_LRU_GATE_MEAN: Dict[Any, float] = {}

def register_lru_gate_hooks(ddp_model: torch.nn.Module) -> None:
    model_to_hook = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    for name, module in model_to_hook.named_modules():
        if name.endswith("lru_layer.gate_conv"):
            if "convlru_blocks." in name:
                try:
                    tag = int(name.split("convlru_blocks.")[1].split(".")[0])
                except Exception:
                    tag = name
            else:
                tag = name
            def _hook(mod: torch.nn.Module, inp: Tuple[Any, ...], out: torch.Tensor, tag_local: Any = tag) -> None:
                with torch.no_grad():
                    _LRU_GATE_MEAN[tag_local] = float(out.mean().detach())
            module.register_forward_hook(_hook)

def format_gate_means() -> str:
    if not _LRU_GATE_MEAN:
        return "g=NA"
    keys = sorted(_LRU_GATE_MEAN.keys(), key=lambda k: (0, k) if isinstance(k, int) else (1, str(k)))
    return " ".join([f"g[b{k}]={_LRU_GATE_MEAN[k]:.4f}" if isinstance(k, int) else f"g[{k}]={_LRU_GATE_MEAN[k]:.4f}" for k in keys])

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

def setup_wandb(rank: int, args: Args, model: torch.nn.Module) -> None:
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

def sample_timestep(args: Args, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if str(getattr(args, "head_mode", "gaussian")).lower() not in ["diffusion", "flow"]:
        return None
    steps = int(getattr(args, "diffusion_steps", 1000))
    t = torch.randint(0, max(1, steps), (batch_size,), device=device)
    return t.to(dtype=dtype)

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

    static_pt_path = "/nfs/ConvLRU/Exp/ERA5/static_feats.pt"
    static_data_cpu = None
    if int(args.static_ch) > 0:
        if os.path.isfile(static_pt_path):
            if rank == 0:
                logging.info(f"Loading static features from {static_pt_path}")
                print(f"Loading static features from {static_pt_path}")
            static_data_cpu = torch.load(static_pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Static features enabled but {static_pt_path} not found!")

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_model_args = load_model_args_from_ckpt(args.ckpt, map_location=f"cuda:{local_rank}")
        if ckpt_model_args:
            print("[Args] applying model args from ckpt before building model.")
            if rank == 0:
                logging.info("[Args] applying model args from ckpt before building model.")
            apply_model_args(args, ckpt_model_args, verbose=True)

    model = ConvLRU(args).cuda(local_rank)
    if bool(args.use_compile):
        model = torch.compile(model, mode="default")
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    register_lru_gate_hooks(model)
    setup_wandb(rank, args, model)

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
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
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
            steps_per_epoch=len_train_dataloader,
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

    if bool(args.init_lr_scheduler) and bool(args.use_scheduler):
        scheduler = lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(args.lr),
            steps_per_epoch=len_train_dataloader,
            epochs=int(args.epochs) - int(start_epoch),
        )

    if not bool(args.use_scheduler):
        for g in opt.param_groups:
            g["lr"] = float(args.lr)

    static_gpu: Optional[torch.Tensor] = None
    if int(args.static_ch) > 0 and static_data_cpu is not None:
        static_gpu = static_data_cpu.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.float32, non_blocking=True)

    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    use_amp = bool(args.use_amp)
    
    micro_steps = int(args.T) 
    
    for ep in range(int(start_epoch), int(args.epochs)):
        train_sampler.set_epoch(ep)
        train_iter = tqdm(train_dataloader, desc=f"Epoch {ep + 1}/{args.epochs}") if rank == 0 else train_dataloader

        for train_step, data in enumerate(train_iter, start=1):
            model.train()
            opt.zero_grad(set_to_none=True)

            B_full, L_full, C, H, W = data.shape
            L_eff = L_full - 1
            
            x = data[:, :L_eff].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)
            target = data[:, 1 : L_eff + 1].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)
            
            listT_vals = [float(args.T)] * x.shape[1]
            listT = torch.tensor(listT_vals, device=x.device, dtype=x.dtype).view(1, -1).repeat(x.size(0), 1)

            static_feats = None
            if static_gpu is not None:
                static_feats = static_gpu.unsqueeze(0).repeat(x.size(0), 1, 1, 1)

            timestep = sample_timestep(args, x.size(0), x.device, x.dtype)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                preds = model(x, mode="p", listT=listT, static_feats=static_feats, timestep=timestep)
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                C_gt = target.shape[2]
                if preds.shape[2] == 2 * C_gt:
                    loss_main = gaussian_nll_loss_weighted(preds, target)
                    p_det = preds[:, :, :C_gt]
                else:
                    loss_main = latitude_weighted_l1(preds, target)
                    p_det = preds

                loss_solver = torch.tensor(0.0, device=x.device)
                loss_consistency = torch.tensor(0.0, device=x.device)
                
                if str(args.train_mode).lower() == "alignment":
                    x_seed = x[:, -1:]
                    target_last = target[:, -1:]
                    
                    dt_micro_val = 1.0
                    listT_seed = torch.full((x.size(0), 1), dt_micro_val, device=x.device, dtype=x.dtype)
                    listT_future = torch.full((x.size(0), micro_steps - 1), dt_micro_val, device=x.device, dtype=x.dtype)
                    
                    preds_recursive_seq = model(
                        x_seed, 
                        mode="i", 
                        out_gen_num=micro_steps, 
                        listT=listT_seed, 
                        listT_future=listT_future,
                        static_feats=static_feats,
                        timestep=timestep
                    )
                    
                    preds_recursive_last = preds_recursive_seq[:, -1:]
                    
                    if preds_recursive_last.shape[2] == 2 * C_gt:
                        loss_solver = F.l1_loss(preds_recursive_last[:, :, :C_gt], target_last)
                        p_rec_det = preds_recursive_last[:, :, :C_gt]
                    else:
                        loss_solver = F.l1_loss(preds_recursive_last, target_last)
                        p_rec_det = preds_recursive_last

                    p_direct_last = p_det[:, -1:].detach()
                    loss_consistency = F.l1_loss(p_rec_det, p_direct_last)

                loss = torch.tensor(0.0, device=x.device)
                if "lat" in args.loss:
                    loss = loss + loss_main
                
                gdl_loss = torch.tensor(0.0, device=x.device)
                if "gdl" in args.loss:
                    gdl_loss = gradient_difference_loss(p_det, target)
                    loss = loss + 0.5 * gdl_loss
                    
                spec_loss = torch.tensor(0.0, device=x.device)
                if "spec" in args.loss:
                    spec_loss = spectral_loss(p_det, target)
                    loss = loss + 0.1 * spec_loss
                
                kl_loss = get_kl_loss(model)
                moe_loss = get_moe_aux_loss(model)
                
                loss = loss + 0.5 * loss_solver + 0.1 * loss_consistency + 0.01 * moe_loss + 1e-6 * kl_loss
                
                if isinstance(model, DDP):
                    dummy_loss = torch.tensor(0.0, device=loss.device)
                    for p in model.parameters():
                        if p.requires_grad:
                            dummy_loss = dummy_loss + p.view(-1)[0].abs() * 0.0
                    loss = loss + dummy_loss

            loss.backward()

            if float(args.grad_clip) and float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            grad_norm, grad_max, _ = get_grad_stats(model) if rank == 0 else (0.0, 0.0, 0)
            opt.step()

            if bool(args.use_scheduler) and scheduler is not None:
                scheduler.step()

            loss_tensor = loss.detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()
            
            solver_tensor = loss_solver.detach()
            dist.all_reduce(solver_tensor, op=dist.ReduceOp.SUM)
            avg_solver = (solver_tensor / world_size).item()

            metric_l1 = F.l1_loss(p_det, target)
            l1_tensor = metric_l1.detach()
            dist.all_reduce(l1_tensor, op=dist.ReduceOp.SUM)
            avg_l1 = (l1_tensor / world_size).item()

            if rank == 0:
                current_lr = scheduler.get_last_lr()[0] if bool(args.use_scheduler) and scheduler is not None else opt.param_groups[0]["lr"]
                gate_str = format_gate_means()
                
                message = (
                    f"Ep {ep + 1} - L: {avg_loss:.4f} - L1: {avg_l1:.4f} - Solv: {avg_solver:.4f} - LR: {current_lr:.2e} "
                    f"- {gate_str}"
                )
                if isinstance(train_iter, tqdm):
                    train_iter.set_description(message)
                logging.info(message)

                if bool(getattr(args, "use_wandb", False)):
                    global_step = ep * len_train_dataloader + train_step
                    log_dict: Dict[str, Any] = {
                        "train/epoch": ep + 1,
                        "train/step": global_step,
                        "train/loss": avg_loss,
                        "train/loss_l1": avg_l1,
                        "train/loss_solver": avg_solver,
                        "train/loss_consistency": loss_consistency.item(),
                        "train/loss_gdl": gdl_loss.item(),
                        "train/loss_spec": spec_loss.item(),
                        "train/lr": float(current_lr),
                        "train/grad_norm": float(grad_norm),
                        "train/moe_aux_loss": moe_loss.item(),
                    }
                    for k, v in _LRU_GATE_MEAN.items():
                        g_key = f"train/gate_b{k}" if isinstance(k, int) else f"train/gate_{k}"
                        log_dict[g_key] = float(v)
                    wandb.log(log_dict, step=int(global_step))

                ckpt_every = max(1, int(len_train_dataloader * float(args.ckpt_step)))
                if (train_step % ckpt_every == 0) or (train_step == len_train_dataloader):
                    save_ckpt(
                        model,
                        opt,
                        ep + 1,
                        train_step,
                        avg_loss,
                        args,
                        scheduler if (bool(args.use_scheduler) and scheduler is not None) else None,
                    )

            del data, x, preds, target, loss, listT, static_feats, timestep, loss_tensor, metric_l1, l1_tensor, solver_tensor
            if train_step % 50 == 0:
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
                    B_full, L_full, C, H, W = data.shape
                    half = int(args.eval_data_n_frames) // 2
                    cond_data = data[:, :half].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)

                    cond_eff = cond_data
                    listT_cond_vals = [float(args.T)] * cond_eff.shape[1]
                    listT_cond = torch.tensor(listT_cond_vals, device=cond_eff.device, dtype=cond_eff.dtype).view(1, -1).repeat(cond_eff.size(0), 1)
                    
                    out_gen_num = int(L_full - cond_eff.shape[1])
                    listT_future = make_listT_from_arg_T(B_full, out_gen_num, cond_eff.device, cond_eff.dtype, float(args.T))
                    target = data[:, cond_eff.shape[1] : cond_eff.shape[1] + out_gen_num].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)

                    static_feats = None
                    if static_gpu is not None:
                        static_feats = static_gpu.unsqueeze(0).repeat(cond_eff.size(0), 1, 1, 1)

                    timestep = sample_timestep(args, cond_eff.size(0), cond_eff.device, cond_eff.dtype)

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        preds = model(
                            cond_eff,
                            mode="i",
                            out_gen_num=out_gen_num,
                            listT=listT_cond,
                            listT_future=listT_future,
                            static_feats=static_feats,
                            timestep=timestep,
                        )

                        if preds.shape[2] == 2 * target.shape[2]:
                            preds_cmp = preds[:, :, :target.shape[2]]
                        else:
                            preds_cmp = preds

                        loss_eval = F.l1_loss(preds_cmp, target)

                    tot_tensor = loss_eval.detach()
                    dist.all_reduce(tot_tensor, op=dist.ReduceOp.SUM)
                    avg_total = (tot_tensor / world_size).item()

                    if rank == 0:
                        message = f"Eval step {eval_step} - L1: {avg_total:.6f}"
                        if isinstance(eval_iter, tqdm):
                            eval_iter.set_description(message)
                        logging.info(message)
                        if bool(getattr(args, "use_wandb", False)):
                            step_id = (ep + 1) * len_train_dataloader
                            wandb.log({"eval/l1_loss": avg_total, "eval/epoch": ep + 1}, step=int(step_id))

                    del data, target, cond_data, cond_eff, preds, loss_eval, tot_tensor, listT_cond, listT_future, static_feats, timestep
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
    if bool(args.use_amp):
        print("[Warning] AMP is disabled by policy. Forcing use_amp=False.")
        logging.warning("AMP is disabled by policy. Forcing use_amp=False.")
        args.use_amp = False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12355")
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)

