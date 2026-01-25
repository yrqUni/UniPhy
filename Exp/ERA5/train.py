import argparse
import datetime
import os
import random
import sys
import warnings
import yaml

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
})

console = Console(theme=custom_theme)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_ddp(rank, world_size, local_rank, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup_ddp():
    dist.destroy_process_group()


def energy_penalty(pred_ensemble, target, threshold=1.5):
    pred_energy = pred_ensemble.pow(2).mean(dim=(-2, -1))
    target_energy = target.unsqueeze(1).pow(2).mean(dim=(-2, -1))
    excess = F.relu(pred_energy - target_energy * threshold)
    return excess.mean()


def spectral_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    H, W = pred_fft.shape[-2:]
    freq_y = torch.arange(H, device=pred.device).float().view(-1, 1)
    freq_x = torch.arange(W, device=pred.device).float().view(1, -1)
    freq_weight = torch.sqrt(freq_y ** 2 + freq_x ** 2) + 1.0
    freq_weight = freq_weight / freq_weight.max()

    diff = torch.abs(pred_fft - target_fft) * freq_weight
    return diff.mean()


def crps_ensemble_loss(pred_ensemble, target):
    B, M, C, H, W = pred_ensemble.shape
    target_expanded = target.unsqueeze(1).expand_as(pred_ensemble)

    mae_term = torch.abs(pred_ensemble - target_expanded).mean(dim=1)

    pred_sorted, _ = torch.sort(pred_ensemble, dim=1)
    diff_matrix = torch.abs(
        pred_sorted.unsqueeze(2) - pred_sorted.unsqueeze(1)
    )
    spread_term = diff_matrix.mean(dim=(1, 2)) / (2.0 * M)

    crps = mae_term.mean() - spread_term.mean()
    return crps, spread_term.mean()


def save_ckpt(model, optimizer, epoch, step, loss, cfg, scheduler=None):
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep{epoch}_step{step}_{timestamp}.pt")
    torch.save(state, ckpt_path)

    if dist.get_rank() == 0:
        console.print(f"[success]Checkpoint saved: {ckpt_path}[/success]")


def load_ckpt(model, optimizer, ckpt_path, scheduler=None):
    if not os.path.exists(ckpt_path):
        console.print(f"[warning]Checkpoint not found: {ckpt_path}[/warning]")
        return 0, 0

    state = torch.load(ckpt_path, map_location="cpu")
    model.module.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    if dist.get_rank() == 0:
        console.print(f"[success]Loaded checkpoint: {ckpt_path}[/success]")

    return state.get("epoch", 0), state.get("step", 0)


def run_ddp(rank, world_size, local_rank, master_addr, master_port, cfg):
    setup_ddp(rank, world_size, local_rank, master_addr, master_port)
    set_seed(42 + rank)

    device = torch.device(f"cuda:{local_rank}")

    if cfg["train"]["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        expand=cfg["model"]["expand"],
        num_experts=cfg["model"]["num_experts"],
        depth=cfg["model"]["depth"],
        patch_size=cfg["model"]["patch_size"],
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_ds = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=cfg["data"]["dt_ref"],
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=cfg["train"]["batch_size"],
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    grad_accum_steps = cfg["train"]["grad_accum_steps"]
    epochs = cfg["train"]["epochs"]

    scheduler = None
    if cfg["train"]["use_scheduler"]:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["train"]["lr"]),
            steps_per_epoch=len(train_loader) // grad_accum_steps,
            epochs=epochs,
        )

    start_epoch = 0
    global_step = 0

    if cfg["logging"]["ckpt"]:
        start_epoch, global_step = load_ckpt(
            model, optimizer, cfg["logging"]["ckpt"], scheduler
        )

    if cfg["logging"]["use_wandb"] and rank == 0:
        run_name = cfg["logging"]["wandb_run_name"] or f"uniphy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=run_name,
            config=cfg,
        )

    save_interval = max(1, int(len(train_loader) * cfg["logging"]["ckpt_step"]))
    log_every = cfg["logging"]["log_every"]
    wandb_every = cfg["logging"]["wandb_every"]
    ensemble_size = cfg["train"]["ensemble_size"]

    energy_weight = cfg["train"].get("energy_penalty_weight", 0.1)
    spectral_weight = cfg["train"].get("spectral_loss_weight", 0.05)

    for ep in range(start_epoch, epochs):
        train_sampler.set_epoch(ep)
        model.train()

        with Progress(
            TextColumn("[bold blue]Epoch {task.fields[ep]}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=(rank != 0),
        ) as progress:

            task = progress.add_task(
                "Training",
                total=len(train_loader),
                ep=f"{ep + 1}/{epochs}",
            )

            for train_step, (data, dt) in enumerate(train_loader):
                data = data.to(device, non_blocking=True)
                dt = dt.to(device, non_blocking=True)

                input_seq = data[:, :-1]
                target = data[:, -1]
                dt_input = dt[:, :-1]

                pred_ensemble_list = []
                for _ in range(ensemble_size):
                    pred = model(input_seq, dt_input)
                    pred_ensemble_list.append(pred[:, -1])

                pred_ensemble = torch.stack(pred_ensemble_list, dim=1)

                B, M, C, H, W = pred_ensemble.shape
                pred_flat = pred_ensemble.reshape(B, M, -1)
                target_flat = target.reshape(B, -1).unsqueeze(-2).expand(-1, C, -1)
                target_flat = target.reshape(B, -1)

                pred_ensemble_reshaped = pred_ensemble.view(B, M, C * H * W)
                target_reshaped = target.view(B, C * H * W)

                crps, spread = crps_ensemble_loss(
                    pred_ensemble.view(B, M, C, H, W),
                    target.view(B, C, H, W),
                )

                pred_mean = pred_ensemble.mean(dim=1)
                mae_val = F.l1_loss(pred_mean, target)
                rmse_val = torch.sqrt(F.mse_loss(pred_mean, target))
                raw_spread = pred_ensemble.std(dim=1).mean()

                loss_collapse = torch.clamp(0.4 * rmse_val - raw_spread, min=0)

                loss_energy = energy_penalty(pred_ensemble, target, threshold=1.5)

                loss_spectral = spectral_loss(pred_mean, target)

                loss = (
                    crps
                    + 2.0 * loss_collapse
                    + energy_weight * loss_energy
                    + spectral_weight * loss_spectral
                )

                (loss / grad_accum_steps).backward()

                if (train_step + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg["train"]["grad_clip"],
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1

                if rank == 0 and (train_step + 1) % log_every == 0:
                    lr_current = optimizer.param_groups[0]["lr"]
                    console.print(
                        f"[info]Step {train_step + 1} | "
                        f"CRPS: {crps.item():.4f} | "
                        f"MAE: {mae_val.item():.4f} | "
                        f"RMSE: {rmse_val.item():.4f} | "
                        f"Spread: {raw_spread.item():.4f} | "
                        f"Energy: {loss_energy.item():.4f} | "
                        f"Spectral: {loss_spectral.item():.4f} | "
                        f"LR: {lr_current:.2e}[/info]"
                    )

                if (
                    rank == 0
                    and cfg["logging"]["use_wandb"]
                    and (train_step + 1) % wandb_every == 0
                ):
                    wandb.log({
                        "train/crps": crps.item(),
                        "train/mae": mae_val.item(),
                        "train/rmse": rmse_val.item(),
                        "train/spread": raw_spread.item(),
                        "train/loss_collapse": loss_collapse.item(),
                        "train/loss_energy": loss_energy.item(),
                        "train/loss_spectral": loss_spectral.item(),
                        "train/total_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": ep,
                        "train/global_step": global_step,
                    })

                if rank == 0 and (train_step + 1) % save_interval == 0:
                    save_ckpt(
                        model, optimizer, ep + 1, train_step + 1, crps.item(), cfg, scheduler
                    )

                progress.advance(task)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if rank == 0:
            save_ckpt(
                model, optimizer, ep + 1, len(train_loader), crps.item(), cfg, scheduler
            )

    if cfg["logging"]["use_wandb"] and rank == 0:
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_ddp(
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        os.environ.get("MASTER_PORT", "12355"),
        cfg,
    )
    