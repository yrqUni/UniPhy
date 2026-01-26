import os
import json
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
from rich.table import Table
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


def format_params(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params, total_params - trainable_params


def print_model_summary(model, cfg, rank):
    if rank != 0:
        return
    total_params, trainable_params, non_trainable_params = get_model_info(model)
    table = Table(title="Model Configuration", header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Parameters", format_params(total_params))
    table.add_row("Trainable Parameters", format_params(trainable_params))
    table.add_row("Embed Dim", str(cfg["model"]["embed_dim"]))
    table.add_row("Depth", str(cfg["model"]["depth"]))
    table.add_row("Num Experts", str(cfg["model"]["num_experts"]))
    table.add_row("Patch Size", str(cfg["model"]["patch_size"]))
    console.print(table)


def setup_logging(log_path, rank):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    if rank == 0:
        os.makedirs(log_path, exist_ok=True)
        
        log_file = os.path.join(
            log_path,
            f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    return logger


def save_ckpt(model, optimizer, scheduler, epoch, global_step, path):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_ckpt(model, optimizer, path, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu")
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"]


def save_metrics(log_path, epoch, metrics):
    metrics_file = os.path.join(log_path, "metrics.jsonl")
    record = {
        "epoch": epoch,
        "timestamp": datetime.datetime.now().isoformat(),
        **metrics
    }
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_l1_loss(pred, target):
    if pred.is_complex():
        pred = pred.real
    return F.l1_loss(pred, target)


def compute_crps_loss(ensemble_preds, target):
    B, E, T, C, H, W = ensemble_preds.shape
    
    target_expanded = target.unsqueeze(1)
    mae_term = (ensemble_preds - target_expanded).abs().mean()
    
    if E > 1:
        idx1 = torch.randperm(E, device=ensemble_preds.device)[:E // 2]
        idx2 = torch.randperm(E, device=ensemble_preds.device)[:E // 2]
        spread_term = (ensemble_preds[:, idx1] - ensemble_preds[:, idx2]).abs().mean()
    else:
        spread_term = torch.tensor(0.0, device=ensemble_preds.device)
    
    crps = mae_term - 0.5 * spread_term
    return crps


def compute_ensemble_loss(ensemble_preds, target):
    ensemble_mean = ensemble_preds.mean(dim=1)
    mean_loss = F.l1_loss(ensemble_mean, target)
    
    ensemble_std = ensemble_preds.std(dim=1)
    actual_error = (ensemble_mean - target).abs()
    spread_loss = F.mse_loss(ensemble_std, actual_error.detach())
    
    return mean_loss, spread_loss


def compute_metrics(pred, target):
    if pred.is_complex():
        pred = pred.real
    
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    mae = F.l1_loss(pred, target)
    
    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
    }


def train_step(model, batch, optimizer, cfg, grad_accum_steps, step_idx):
    data, dt = batch
    data = data.cuda(non_blocking=True)
    dt = dt.cuda(non_blocking=True)
    
    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt[:, :-1] if dt.ndim > 1 else dt[:-1]
    
    ensemble_size = cfg["train"]["ensemble_size"]
    
    ensemble_preds = []
    for _ in range(ensemble_size):
        pred = model(x_input, dt_input)
        if pred.is_complex():
            pred = pred.real
        ensemble_preds.append(pred)
    
    ensemble_preds = torch.stack(ensemble_preds, dim=1)
    ensemble_mean = ensemble_preds.mean(dim=1)
    
    l1_loss = compute_l1_loss(ensemble_mean, x_target)
    crps_loss = compute_crps_loss(ensemble_preds, x_target)
    
    ensemble_weight = cfg["train"].get("ensemble_weight", 0.5)
    loss = l1_loss + ensemble_weight * crps_loss
    
    loss_scaled = loss / grad_accum_steps
    loss_scaled.backward()
    
    grad_norm = 0.0
    if (step_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["train"]["grad_clip"]
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    metrics = compute_metrics(ensemble_mean, x_target)
    metrics.update({
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_preds.std(dim=1).mean().item(),
    })
    
    return metrics


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
        logger.info("Training Started")
        logger.info(f"World Size: {world_size}")
        logger.info("=" * 70)
    
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
    ).cuda()
    
    model = DDP(model, device_ids=[local_rank])
    
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
        start_epoch, global_step = load_ckpt(
            model, optimizer, cfg["logging"]["ckpt"], scheduler
        )
        if rank == 0:
            logger.info(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    
    if cfg["logging"]["use_wandb"] and rank == 0:
        run_name = cfg["logging"]["wandb_run_name"] or f"uniphy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total_params, _, _ = get_model_info(model.module)
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=run_name,
            config={**cfg, "total_params": total_params},
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
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                        }, step=global_step)
                
                if (batch_idx + 1) % save_interval == 0:
                    ckpt_path = os.path.join(
                        cfg["logging"]["ckpt_dir"],
                        f"ckpt_e{epoch + 1}_s{global_step}.pt"
                    )
                    save_ckpt(model, optimizer, scheduler, epoch, global_step, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")
        
        if rank == 0:
            progress.stop()
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(num_batches, 1)
            
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch + 1}/{epochs} Completed")
            logger.info(f"  Avg Loss: {epoch_metrics['loss']:.4f}")
            logger.info(f"  Avg L1 Loss: {epoch_metrics['l1_loss']:.4f}")
            logger.info(f"  Avg CRPS Loss: {epoch_metrics['crps_loss']:.4f}")
            logger.info(f"  Avg RMSE: {epoch_metrics['rmse']:.4f}")
            logger.info(f"  Avg MAE: {epoch_metrics['mae']:.4f}")
            logger.info(f"  Avg Ensemble Std: {epoch_metrics['ensemble_std']:.4f}")
            logger.info("=" * 70)
            
            save_metrics(cfg["logging"]["log_path"], epoch + 1, epoch_metrics)
            
            if cfg["logging"]["use_wandb"]:
                wandb.log({
                    "epoch/loss": epoch_metrics["loss"],
                    "epoch/l1_loss": epoch_metrics["l1_loss"],
                    "epoch/crps_loss": epoch_metrics["crps_loss"],
                    "epoch/rmse": epoch_metrics["rmse"],
                    "epoch/mae": epoch_metrics["mae"],
                    "epoch": epoch + 1,
                }, step=global_step)
            
            ckpt_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"ckpt_epoch{epoch + 1}.pt"
            )
            save_ckpt(model, optimizer, scheduler, epoch + 1, global_step, ckpt_path)
            logger.info(f"Epoch checkpoint saved: {ckpt_path}")
    
    if rank == 0:
        logger.info("Training Completed!")
        if cfg["logging"]["use_wandb"]:
            wandb.finish()
    
    dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    train(cfg)


if __name__ == "__main__":
    main()

