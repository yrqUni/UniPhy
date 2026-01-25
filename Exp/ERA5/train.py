import argparse
import datetime
import gc
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
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
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
    torch.cuda.manual_seed_all(seed)


def save_ckpt(model, optimizer, epoch, step, path, scheduler=None):
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_ckpt(model, optimizer, path, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def compute_spectral_loss(pred, target):
    if pred.is_complex():
        pred_real = pred.real
    else:
        pred_real = pred
    pred_fft = torch.fft.rfft2(pred_real, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(pred_fft.abs(), target_fft.abs())


def compute_energy_penalty(pred, target):
    if pred.is_complex():
        pred_energy = (pred.abs() ** 2).mean(dim=(-2, -1))
    else:
        pred_energy = (pred ** 2).mean(dim=(-2, -1))
    target_energy = (target ** 2).mean(dim=(-2, -1))
    return F.mse_loss(pred_energy, target_energy)


def train_step(model, batch, optimizer, cfg, grad_accum_steps, step_in_accum):
    data, dt = batch
    data = data.cuda(non_blocking=True)
    dt = dt.cuda(non_blocking=True)

    x_input = data[:, :-1]
    x_target = data[:, 1:]

    dt_input = dt[:, :-1] if dt.ndim > 1 else dt[:-1]

    x_pred = model(x_input, dt_input)

    if x_pred.is_complex():
        loss_mse = F.mse_loss(x_pred.real, x_target) + F.mse_loss(x_pred.imag, torch.zeros_like(x_target))
    else:
        loss_mse = F.mse_loss(x_pred, x_target)

    loss_spectral = compute_spectral_loss(x_pred, x_target) * cfg["train"]["spectral_loss_weight"]
    loss_energy = compute_energy_penalty(x_pred, x_target) * cfg["train"]["energy_penalty_weight"]

    loss = loss_mse + loss_spectral + loss_energy
    loss = loss / grad_accum_steps

    loss.backward()

    if (step_in_accum + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss.item() * grad_accum_steps,
        "mse": loss_mse.item(),
        "spectral": loss_spectral.item(),
        "energy": loss_energy.item(),
    }


def run_ddp(rank, world_size, local_rank, master_addr, master_port, cfg):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    set_seed(42 + rank)

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
        max_growth_rate=cfg["model"]["max_growth_rate"],
    ).cuda()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=cfg["data"]["dt_ref"],
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
        os.makedirs(cfg["logging"]["log_path"], exist_ok=True)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        if rank == 0:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            task = progress.add_task(f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))
            progress.start()

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_spectral = 0.0
        epoch_energy = 0.0
        num_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(
                model, batch, optimizer, cfg, grad_accum_steps, batch_idx
            )

            epoch_loss += metrics["loss"]
            epoch_mse += metrics["mse"]
            epoch_spectral += metrics["spectral"]
            epoch_energy += metrics["energy"]
            num_batches += 1

            if scheduler is not None and (batch_idx + 1) % grad_accum_steps == 0:
                scheduler.step()

            global_step += 1

            if rank == 0:
                progress.update(task, advance=1)

                if batch_idx % log_every == 0:
                    console.print(
                        f"[info]Step {global_step}[/info] | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"MSE: {metrics['mse']:.4f} | "
                        f"Spectral: {metrics['spectral']:.4f} | "
                        f"Energy: {metrics['energy']:.4f}"
                    )

                if cfg["logging"]["use_wandb"] and batch_idx % cfg["logging"]["wandb_every"] == 0:
                    wandb.log({
                        "train/loss": metrics["loss"],
                        "train/mse": metrics["mse"],
                        "train/spectral": metrics["spectral"],
                        "train/energy": metrics["energy"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                    })

                if global_step % save_interval == 0:
                    ckpt_path = os.path.join(
                        cfg["logging"]["ckpt_dir"],
                        f"ckpt_epoch{epoch}_step{global_step}.pt"
                    )
                    save_ckpt(model, optimizer, epoch, global_step, ckpt_path, scheduler)
                    console.print(f"[success]Checkpoint saved: {ckpt_path}[/success]")

        if rank == 0:
            progress.stop()

            avg_loss = epoch_loss / num_batches
            avg_mse = epoch_mse / num_batches
            avg_spectral = epoch_spectral / num_batches
            avg_energy = epoch_energy / num_batches

            table = Table(title=f"Epoch {epoch + 1} Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Avg Loss", f"{avg_loss:.4f}")
            table.add_row("Avg MSE", f"{avg_mse:.4f}")
            table.add_row("Avg Spectral", f"{avg_spectral:.4f}")
            table.add_row("Avg Energy", f"{avg_energy:.4f}")
            console.print(table)

            if cfg["logging"]["use_wandb"]:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/mse": avg_mse,
                    "epoch/spectral": avg_spectral,
                    "epoch/energy": avg_energy,
                    "epoch": epoch + 1,
                })

            ckpt_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"ckpt_epoch{epoch + 1}.pt"
            )
            save_ckpt(model, optimizer, epoch + 1, global_step, ckpt_path, scheduler)
            console.print(f"[success]Epoch checkpoint saved: {ckpt_path}[/success]")

        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0 and cfg["logging"]["use_wandb"]:
        wandb.finish()

    dist.destroy_process_group()


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
    