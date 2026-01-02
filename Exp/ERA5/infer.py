import argparse
import datetime
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
    "hidden_activation",
    "output_activation",
    "emb_strategy",
    "hidden_factor",
    "emb_ch",
    "emb_hidden_ch",
    "emb_hidden_layers_num",
    "convlru_num_blocks",
    "use_cbam",
    "use_gate",
    "lru_rank",
    "use_freq_prior",
    "freq_rank",
    "freq_gain_init",
    "freq_mode",
    "use_sh_prior",
    "sh_Lmax",
    "sh_rank",
    "sh_gain_init",
    "ffn_hidden_ch",
    "ffn_hidden_layers_num",
    "num_expert",
    "activate_expert",
    "dec_strategy",
    "dec_hidden_ch",
    "dec_hidden_layers_num",
    "static_ch",
    "use_selective",
    "unet",
    "down_mode",
    "head_mode",
    "use_checkpointing",
    "pool_mode",
    "use_spectral_mixing",
    "use_anisotropic_diffusion",
    "use_advection",
    "use_graph_interaction",
    "use_adaptive_ssm",
    "use_neural_operator",
    "learnable_init_state",
    "use_wavelet_ssm",
    "use_cross_var_attn",
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
        self.hidden_activation = "SiLU"
        self.output_activation = "Tanh"
        self.emb_strategy = "pxus"
        self.hidden_factor = (7, 12)
        self.emb_ch = 120
        self.emb_hidden_ch = 150
        self.emb_hidden_layers_num = 2
        self.convlru_num_blocks = 6
        self.use_cbam = True
        self.ffn_hidden_ch = 150
        self.ffn_hidden_layers_num = 2
        self.num_expert = 16
        self.activate_expert = 4
        self.use_gate = True
        self.lru_rank = 32
        self.use_selective = True
        self.unet = True
        self.down_mode = "shuffle"
        self.use_freq_prior = False
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = True
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.head_mode = "gaussian"
        self.diffusion_steps = 1000
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.year_range = [2022, 2022]
        self.train_data_n_frames = 27
        self.eval_data_n_frames = 20
        self.eval_sample_num = 100
        self.ckpt = "e7_s570_l0.265707.pth"
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.epochs = 128
        self.log_path = "./convlru_base/logs"
        self.ckpt_dir = "./convlru_base/ckpt"
        self.ckpt_step = 0.25
        self.do_eval = True
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-5
        self.weight_decay = 0.05
        self.use_scheduler = False
        self.init_lr_scheduler = False
        self.loss = "lat"
        self.T = 6
        self.use_amp = False
        self.amp_dtype = "bf16"
        self.grad_clip = 1.0
        self.sample_k = 9
        self.use_wandb = False
        self.wandb_project = "ERA5"
        self.wandb_entity = "ConvLRU"
        self.wandb_run_name = "Infer"
        self.wandb_group = "Infer"
        self.wandb_mode = "offline"
        self.use_checkpointing = True
        self.use_spectral_mixing = True
        self.use_anisotropic_diffusion = True
        self.use_advection = True
        self.use_graph_interaction = False
        self.use_adaptive_ssm = True
        self.use_neural_operator = False
        self.learnable_init_state = True
        self.use_wavelet_ssm = True
        self.use_cross_var_attn = True
        self.ConvType = "dcn"
        self.Arch = "bifpn"

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
    log_filename = os.path.join(args.log_path, f"inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

def load_model_args_from_ckpt(ckpt_path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model_args = ckpt.get("model_args", None)
    if model_args is None:
        args_all = ckpt.get("args_all", None)
        if isinstance(args_all, dict):
            model_args = {k: args_all[k] for k in MODEL_ARG_KEYS if k in args_all}
    del ckpt
    return model_args

def apply_model_args(args_obj: Any, model_args_dict: Optional[Dict[str, Any]], verbose: bool = True) -> None:
    if not model_args_dict:
        return
    for k, v in model_args_dict.items():
        if hasattr(args_obj, k):
            old = getattr(args_obj, k)
            if verbose and old != v:
                if dist.get_rank() == 0:
                    logging.info(f"[Args] restore '{k}': {old} -> {v}")
            setattr(args_obj, k, v)

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

def load_ckpt(model: torch.nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = adapt_state_dict_keys(checkpoint["model"], model)
    model.load_state_dict(state_dict, strict=False)
    del checkpoint
    del state_dict
    if dist.get_rank() == 0:
        logging.info(f"Loaded checkpoint from {ckpt_path}")

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

def weighted_rmse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, L, C, H, W = preds.shape
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    err = (preds - targets).pow(2)
    weighted_err = err * w
    mse = weighted_err.mean()
    return torch.sqrt(mse)

def weighted_acc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, L, C, H, W = preds.shape
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    
    preds_mean = torch.mean(preds, dim=(3, 4), keepdim=True)
    targets_mean = torch.mean(targets, dim=(3, 4), keepdim=True)
    
    preds_anom = preds - preds_mean
    targets_anom = targets - targets_mean
    
    cov = torch.sum(w * preds_anom * targets_anom, dim=(3, 4))
    var_pred = torch.sum(w * preds_anom * preds_anom, dim=(3, 4))
    var_target = torch.sum(w * targets_anom * targets_anom, dim=(3, 4))
    
    acc = cov / torch.sqrt(var_pred * var_target + 1e-6)
    return acc.mean()

def run_inference(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: str, args: Args) -> None:
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0:
        setup_logging(args)

    static_pt_path = "/nfs/ConvLRU/Exp/ERA5/static_feats.pt"
    static_data_cpu = None
    if int(args.static_ch) > 0:
        if os.path.isfile(static_pt_path):
            if rank == 0:
                logging.info(f"Loading static features from {static_pt_path}")
            static_data_cpu = torch.load(static_pt_path, map_location="cpu")
        else:
            if rank == 0:
                logging.warning(f"Static features enabled but {static_pt_path} not found!")

    ckpt_model_args = load_model_args_from_ckpt(args.ckpt, map_location=f"cuda:{local_rank}")
    if ckpt_model_args:
        apply_model_args(args, ckpt_model_args, verbose=True)

    model = ConvLRU(args).cuda(local_rank)
    load_ckpt(model, args.ckpt, map_location=f"cuda:{local_rank}")
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
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
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=2,
        pin_memory=True,
    )

    static_gpu: Optional[torch.Tensor] = None
    if int(args.static_ch) > 0 and static_data_cpu is not None:
        static_gpu = static_data_cpu.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.float32)

    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    use_amp = bool(args.use_amp)

    eval_iter = tqdm(eval_dataloader, desc="Inference") if rank == 0 else eval_dataloader
    
    total_rmse = 0.0
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for i, data in enumerate(eval_iter):
            B_full, L_full, C, H, W = data.shape
            
            cond_len = 2 
            pred_len = L_full - cond_len
            
            cond_data = data[:, :cond_len].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)
            target_data = data[:, cond_len:].to(device=torch.device(f"cuda:{local_rank}"), non_blocking=True).to(torch.float32)
            
            dt_step = float(args.T)
            
            listT_cond = torch.full((B_full, cond_len), dt_step, device=cond_data.device, dtype=cond_data.dtype)
            listT_future = torch.full((B_full, pred_len - 1), dt_step, device=cond_data.device, dtype=cond_data.dtype)
            
            static_feats = None
            if static_gpu is not None:
                static_feats = static_gpu.unsqueeze(0).repeat(B_full, 1, 1, 1)

            timestep = None
            if str(args.head_mode).lower() in ["diffusion", "flow"]:
                timestep = torch.zeros(B_full, device=cond_data.device, dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                preds = model(
                    cond_data,
                    mode="i",
                    out_gen_num=pred_len,
                    listT=listT_cond,
                    listT_future=listT_future,
                    static_feats=static_feats,
                    timestep=timestep
                )

                C_gt = target_data.shape[2]
                if preds.shape[2] == 2 * C_gt:
                    preds_det = preds[:, :, :C_gt]
                else:
                    preds_det = preds

                rmse_val = weighted_rmse(preds_det, target_data)
                acc_val = weighted_acc(preds_det, target_data)

            total_rmse += rmse_val.item()
            total_acc += acc_val.item()
            count += 1
            
            if rank == 0:
                eval_iter.set_description(f"RMSE: {rmse_val.item():.4f} | ACC: {acc_val.item():.4f}")

    # Synchronize metrics
    metrics_tensor = torch.tensor([total_rmse, total_acc, count], device=torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    final_rmse = metrics_tensor[0] / metrics_tensor[2]
    final_acc = metrics_tensor[1] / metrics_tensor[2]

    if rank == 0:
        logging.info(f"Final Inference Results:")
        logging.info(f"  Weighted RMSE: {final_rmse.item():.5f}")
        logging.info(f"  Weighted ACC : {final_acc.item():.5f}")
        print(f"\nFinal Inference Results:\n  Weighted RMSE: {final_rmse.item():.5f}\n  Weighted ACC : {final_acc.item():.5f}")

    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12356")
    
    run_inference(rank, world_size, local_rank, master_addr, master_port, args)

