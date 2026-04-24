import argparse
import copy
import logging
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Model.UniPhy.ModelUniPhy import UniPhyModel

SURFACE_VARS = ["TCWV", "U10", "V10", "T2", "MSLP", "SP"]
PRESSURE_BASE_VARS = ["VV", "U", "V", "RH", "T", "Z"]
PRESSURE_LEVELS = ["925", "850", "500", "100"]
CHANNEL_NAMES = SURFACE_VARS + [
    f"{var}{level}" for var in PRESSURE_BASE_VARS for level in PRESSURE_LEVELS
]
MODEL_ARG_NAMES = {
    "in_channels",
    "out_channels",
    "embed_dim",
    "expand",
    "depth",
    "patch_size",
    "img_height",
    "img_width",
    "dt_ref",
    "init_noise_scale",
}
DEFAULT_MODEL_CFG = {
    "in_channels": 30,
    "out_channels": 30,
    "embed_dim": 128,
    "expand": 4,
    "depth": 8,
    "patch_size": [7, 15],
    "img_height": 721,
    "img_width": 1440,
    "dt_ref": 6.0,
    "init_noise_scale": 0.0001,
    "ensemble_size": 4,
}
DEFAULT_TRAIN_YEAR_RANGE = [2000, 2008]
DEFAULT_TEST_YEAR_RANGE = [2009, 2009]


class SpeedColumn(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return Text("0.00 it/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} it/s", style="progress.data.speed")


def load_yaml_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_year_range(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    cleaned = text.replace(":", ",").strip()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        value = int(parts[0])
        return [value, value]
    if len(parts) != 2:
        raise ValueError(f"Invalid year range: {text}")
    return [int(parts[0]), int(parts[1])]


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if not text:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def resolve_data_input_dir(
    cfg: Dict[str, object],
    override: Optional[str] = None,
) -> str:
    if override:
        return override
    return str(cfg["data"]["input_dir"])


def resolve_train_year_range(
    cfg: Dict[str, object],
    override: Optional[str] = None,
) -> List[int]:
    parsed = parse_year_range(override)
    if parsed is not None:
        return parsed
    cfg_years = cfg["data"].get("year_range")
    if cfg_years:
        return [int(cfg_years[0]), int(cfg_years[1])]
    return list(DEFAULT_TRAIN_YEAR_RANGE)


def resolve_eval_year_range(override: Optional[str] = None) -> List[int]:
    parsed = parse_year_range(override)
    if parsed is not None:
        return parsed
    return list(DEFAULT_TEST_YEAR_RANGE)


def get_device(device: Optional[str] = None):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(seed_base=42):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    set_seed(seed_base + rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def build_rank_console(rank):
    devnull = open(os.devnull, "w")
    if rank == 0:
        console = Console() if sys.stdout.isatty() else Console(file=devnull)
    else:
        console = Console(file=devnull)
    return console, devnull


def build_lat_weights(height, device):
    lat = torch.linspace(-90, 90, height, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, height, 1)


def compute_crps(pred_ensemble, target):
    ensemble_size = pred_ensemble.shape[0]
    target_exp = target.unsqueeze(0)
    mae = (pred_ensemble - target_exp).abs().mean()
    if ensemble_size > 1:
        idx_i, idx_j = torch.triu_indices(
            ensemble_size,
            ensemble_size,
            offset=1,
            device=target.device,
        )
        pairwise_diff = (pred_ensemble[idx_i] - pred_ensemble[idx_j]).abs().mean()
        num_pairs = idx_i.shape[0]
        return mae - pairwise_diff * num_pairs / (ensemble_size * ensemble_size)
    return mae


def setup_file_logger(name, log_path, rank, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if rank == 0:
        os.makedirs(log_path, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_path, filename))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def build_runtime_arg_parser(
    config_path: str,
    *,
    include_pretrained_ckpt: bool = False,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--data-input-dir", default=None)
    parser.add_argument("--train-year-range", default=None)
    parser.add_argument("--sample-offsets-hours", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--ckpt-path", default=None)
    if include_pretrained_ckpt:
        parser.add_argument("--pretrained-ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def build_progress(console=None):
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        SpeedColumn(),
        console=console,
    )


def build_uniphy_model(model_cfg, device=None):
    filtered_cfg = {key: model_cfg[key] for key in MODEL_ARG_NAMES if key in model_cfg}
    filtered_cfg["patch_size"] = tuple(filtered_cfg["patch_size"])
    model = UniPhyModel(**filtered_cfg)
    if device is not None:
        model = model.to(device)
    return model


def wrap_ddp(model, local_rank):
    return DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers=False,
    )


def build_distributed_loader(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )
    return sampler, loader


def build_adamw_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )


def should_stop_early(cfg, global_step):
    max_steps = cfg.get("runtime", {}).get("max_steps")
    return max_steps is not None and global_step >= int(max_steps)


def flush_remaining_grads(model, optimizer, grad_clip, batch_idx, grad_accum_steps):
    if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def build_runtime_cfg(
    config_path: str,
    *,
    data_input_dir: Optional[str] = None,
    train_year_range: Optional[str] = None,
    sample_offsets_hours: Optional[str] = None,
    epochs: Optional[int] = None,
    log_path: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    pretrained_ckpt: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    cfg = copy.deepcopy(load_yaml_config(config_path))
    cfg.setdefault("logging", {})
    cfg.setdefault("data", {})
    cfg.setdefault("runtime", {})
    cfg["data"]["input_dir"] = resolve_data_input_dir(cfg, data_input_dir)
    cfg["data"]["year_range"] = resolve_train_year_range(cfg, train_year_range)
    offsets = parse_float_list(sample_offsets_hours)
    if offsets is not None:
        cfg["data"]["sample_offsets_hours"] = offsets
    if epochs is not None:
        cfg["train"]["epochs"] = int(epochs)
    if log_path is not None:
        cfg["logging"]["log_path"] = log_path
    if ckpt_dir is not None:
        cfg["logging"]["ckpt_dir"] = ckpt_dir
    if ckpt_path is not None:
        cfg["logging"]["ckpt"] = ckpt_path
    if pretrained_ckpt is not None:
        cfg.setdefault("alignment", {})
        cfg["alignment"]["pretrained_ckpt"] = pretrained_ckpt
    cfg["runtime"]["max_steps"] = None if max_steps is None else int(max_steps)
    return cfg
