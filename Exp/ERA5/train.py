import os
import sys
import random
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler
import wandb
import yaml
from rich.console import Console
from rich.theme import Theme
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

import warnings
warnings.filterwarnings("ignore")

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
})
console = Console(theme=custom_theme)

def setup_logging(log_path, rank):
    os.makedirs(log_path, exist_ok=True)
    
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    
    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_path, "train.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable

def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

def print_model_summary(model, cfg, rank):
    if rank != 0:
        return
    total, trainable, frozen = get_model_info(model)
    console.print(f"[bold cyan]Model Summary[/bold cyan]")
    console.print(f"  Total Parameters: {format_params(total)}")
    console.print(f"  Trainable: {format_params(trainable)}")
    console.print(f"  Frozen: {format_params(frozen)}")

def get_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, H, 1)

def compute_crps(pred_ensemble, target):
    M = pred_ensemble.shape[0]
    mae = (pred_ensemble - target.unsqueeze(0)).abs().mean(dim=0)
    
    diff = torch.tensor(0.0, device=pred_ensemble.device, dtype=pred_ensemble.dtype)
    for i in range(M):
        for j in range(i + 1, M):
            diff = diff + (pred_ensemble[i] - pred_ensemble[j]).abs().mean()
    
    if M > 1:
        diff = diff / (M * (M - 1) / 2)
    
    crps = mae.mean() - 0.5 * diff
    return crps

def train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()
    
    B, T, C, H, W = data.shape
    
    ensemble_size = cfg["train"]["ensemble_size"]
    ensemble_weight = cfg["train"]["ensemble_weight"]
    
    x_input = data[:, :-1].detach().clone()
    x_target = data[:, 1:].detach().clone()
    dt_input = dt_data[:, :-1].detach().clone()
    
    out = model(x_input, dt_input)
    
    if out.is_complex():
        out_real = out.real.contiguous()
    else:
        out_real = out.contiguous()
    
    lat_weights = get_lat_weights(H, W, device)
    
    l1_loss = (out_real - x_target).abs().mean()
    mse_loss = ((out_real - x_target) ** 2 * lat_weights).mean()
    
    if ensemble_size > 1 and ensemble_weight > 0:
        ensemble_preds = [out_real.detach()]
        
        infer_model = model.module if hasattr(model, "module") else model
        
        with torch.no_grad():
            for _ in range(ensemble_size - 1):
                out_ens = infer_model(x_input, dt_input)
                if out_ens.is_complex():
                    ensemble_preds.append(out_ens.real.contiguous())
                else:
                    ensemble_preds.append(out_ens.contiguous())
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        crps_loss = compute_crps(ensemble_stack, x_target)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        
        loss = l1_loss + ensemble_weight * crps_loss.detach()
    else:
        crps_loss = torch.tensor(0.0, device=device)
        ensemble_std = torch.tensor(0.0, device=device)
        loss = l1_loss
    
    loss_scaled = loss / grad_accum_steps
    loss_scaled.backward()
    
    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"]
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    with torch.no_grad():
        rmse = torch.sqrt(mse_loss)
        mae = l1_loss
    
    metrics = {
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item() if isinstance(crps_loss, torch.Tensor) else crps_loss,
        "mse": mse_loss.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item() if isinstance(ensemble_std, torch.Tensor) else ensemble_std,
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, path):
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "cfg": cfg,
        "model_args": {
            "in_channels": cfg["model"]["in_channels"],
            "out_channels": cfg["model"]["out_channels"],
            "embed_dim": cfg["model"]["embed_dim"],
            "expand": cfg["model"]["expand"],
            "num_experts": cfg["model"]["num_experts"],
            "depth": cfg["model"]["depth"],
            "patch_size": cfg["model"]["patch_size"],
            "img_height": cfg["model"]["img_height"],
            "img_width": cfg["model"]["img_width"],
            "dt_ref": cfg["model"]["dt_ref"],
            "sde_mode": cfg["model"]["sde_mode"],
            "init_noise_scale": cfg["model"]["init_noise_scale"],
            "max_growth_rate": cfg["model"]["max_growth_rate"],
        },
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    
    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)

def train(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    set_seed(42 + rank)
    
    logger = setup_logging(cfg["logging"]["log_path"], rank)
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info("UniPhy Training")
        logger.info(f"World Size: {world_size}")
        logger.info("=" * 70)
    
    if cfg["train"]["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if rank == 0 and cfg["logging"]["use_wandb"]:
        run_name = cfg["logging"]["wandb_run_name"] or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=run_name,
            config=cfg,
        )
    
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
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if rank == 0:
        print_model_summary(model.module, cfg, rank)
        total_params, trainable_params, _ = get_model_info(model.module)
        logger.info(f"Model Parameters: {format_params(total_params)} (Trainable: {format_params(trainable_params)})")
    
    train_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=cfg["data"]["dt_ref"],
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )
    
    if rank == 0:
        logger.info(f"Dataset Size: {len(train_dataset)} samples")
        logger.info(f"Batches per Epoch: {len(train_loader)}")
    
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
        start_epoch, global_step = load_checkpoint(
            cfg["logging"]["ckpt"], model, optimizer, scheduler
        )
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {cfg['logging']['ckpt']}")
    
    log_every = cfg["logging"]["log_every"]
    save_interval = max(1, int(len(train_loader) * cfg["logging"]["ckpt_step"]))
    
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
        
        epoch_metrics = {
            "loss": 0.0,
            "l1_loss": 0.0,
            "crps_loss": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "grad_norm": 0.0,
            "ensemble_std": 0.0,
        }
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
            
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            num_batches += 1
            global_step += 1
            
            if scheduler is not None and (batch_idx + 1) % grad_accum_steps == 0:
                scheduler.step()
            
            if rank == 0:
                progress.update(task, advance=1)
                
                if (batch_idx + 1) % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[{epoch + 1}/{epochs}][{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"L1: {metrics['l1_loss']:.4f} | "
                        f"CRPS: {metrics['crps_loss']:.4f} | "
                        f"RMSE: {metrics['rmse']:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                
                if cfg["logging"]["use_wandb"]:
                    wandb.log({
                        "train/loss": metrics["loss"],
                        "train/l1_loss": metrics["l1_loss"],
                        "train/crps_loss": metrics["crps_loss"],
                        "train/mse": metrics["mse"],
                        "train/rmse": metrics["rmse"],
                        "train/mae": metrics["mae"],
                        "train/ensemble_std": metrics["ensemble_std"],
                        "train/grad_norm": metrics["grad_norm"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }, step=global_step)
                
                if save_interval > 0 and (batch_idx + 1) % save_interval == 0:
                    ckpt_path = os.path.join(
                        cfg["logging"]["ckpt_dir"],
                        f"ckpt_e{epoch + 1}_s{global_step}.pt"
                    )
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
        
        if rank == 0:
            progress.stop()
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(num_batches, 1)
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs} Summary: "
                f"Loss: {epoch_metrics['loss']:.4f} | "
                f"RMSE: {epoch_metrics['rmse']:.4f} | "
                f"MAE: {epoch_metrics['mae']:.4f}"
            )
            
            ckpt_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"ckpt_epoch{epoch + 1}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, ckpt_path)
            logger.info(f"Saved epoch checkpoint: {ckpt_path}")
        
        dist.barrier()
    
    if rank == 0:
        final_ckpt_path = os.path.join(cfg["logging"]["ckpt_dir"], "ckpt_final.pt")
        save_checkpoint(model, optimizer, scheduler, epochs, global_step, cfg, final_ckpt_path)
        logger.info(f"Training completed. Final checkpoint: {final_ckpt_path}")
        
        if cfg["logging"]["use_wandb"]:
            wandb.finish()
    
    train_dataset.cleanup()
    dist.destroy_process_group()

def main():
    with open("train.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
    
