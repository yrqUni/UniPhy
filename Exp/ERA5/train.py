import os
import sys
import random
import datetime
import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler
import wandb
import yaml
import time

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.console import Group
from rich.style import Style
from rich.text import Text

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

import warnings
warnings.filterwarnings("ignore")

console = Console()

def setup_logging(log_path, rank):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    if rank == 0:
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_path, "train.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def create_dashboard(progress, metrics, epoch, max_epoch, last_msg):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    table.add_column("Epoch Avg", justify="right", style="yellow")

    keys = ["loss", "rmse", "l1_loss", "crps_loss", "grad_norm", "lr"]
    display_names = ["Total Loss", "RMSE", "L1 Loss", "CRPS", "Grad Norm", "Learning Rate"]

    for key, name in zip(keys, display_names):
        curr_val = metrics.get(key, 0.0)
        avg_val = metrics.get(f"avg_{key}", 0.0)
        
        fmt = "{:.6f}" if key == "lr" else "{:.4f}"
        table.add_row(name, fmt.format(curr_val), fmt.format(avg_val))

    status_panel = Panel(
        Text(last_msg, style="bold white"),
        title="[bold blue]System Status",
        border_style="blue",
        padding=(1, 2)
    )

    metrics_panel = Panel(
        table,
        title=f"[bold cyan]Training Metrics (Epoch {epoch}/{max_epoch})",
        border_style="cyan"
    )

    return Group(
        Panel(progress, title="[bold green]Progress", border_style="green"),
        metrics_panel,
        status_panel
    )

def train(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    set_seed(42 + rank)
    logger = setup_logging(cfg["logging"]["log_path"], rank)
    
    if rank == 0:
        logger.info(f"Training started on {world_size} GPUs")

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
    
    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
    
    last_msg = "Initializing..."
    
    if rank == 0:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        task_epoch = progress.add_task("[bold]Training...", total=len(train_loader) * epochs)
    
    display_metrics = {}

    if rank == 0:
        live = Live(
            create_dashboard(progress, display_metrics, start_epoch + 1, epochs, last_msg),
            refresh_per_second=10,
            console=console
        )
        live.start()

    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            
            epoch_accum = {
                "loss": 0.0, "l1_loss": 0.0, "crps_loss": 0.0,
                "rmse": 0.0, "grad_norm": 0.0
            }
            num_batches = 0
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
                
                for key in epoch_accum:
                    if key in metrics:
                        epoch_accum[key] += metrics[key]
                num_batches += 1
                global_step += 1
                
                if scheduler is not None and (batch_idx + 1) % grad_accum_steps == 0:
                    scheduler.step()
                
                if rank == 0:
                    progress.update(task_epoch, advance=1, description=f"Epoch {epoch + 1}/{epochs}")
                    
                    if (batch_idx + 1) % log_every == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        display_metrics = {
                            "loss": metrics["loss"],
                            "l1_loss": metrics["l1_loss"],
                            "crps_loss": metrics["crps_loss"],
                            "rmse": metrics["rmse"],
                            "grad_norm": metrics["grad_norm"],
                            "lr": current_lr,
                            "avg_loss": epoch_accum["loss"] / num_batches,
                            "avg_l1_loss": epoch_accum["l1_loss"] / num_batches,
                            "avg_crps_loss": epoch_accum["crps_loss"] / num_batches,
                            "avg_rmse": epoch_accum["rmse"] / num_batches,
                            "avg_grad_norm": epoch_accum["grad_norm"] / num_batches,
                        }
                        
                        live.update(create_dashboard(progress, display_metrics, epoch + 1, epochs, last_msg))

                        logger.info(
                            f"E{epoch+1} B{batch_idx+1} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"L1: {metrics['l1_loss']:.4f} | "
                            f"RMSE: {metrics['rmse']:.4f}"
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
                        last_msg = f"Checkpointed: {os.path.basename(ckpt_path)} at Step {global_step}"
                        logger.info(f"Saved checkpoint: {ckpt_path}")
            
            if rank == 0:
                epoch_path = os.path.join(cfg["logging"]["ckpt_dir"], f"ckpt_epoch{epoch + 1}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, epoch_path)
                last_msg = f"Epoch {epoch + 1} Completed. Saved: {os.path.basename(epoch_path)}"
                logger.info(f"Epoch {epoch + 1} finished. Saved checkpoint.")

            dist.barrier()
        
        if rank == 0:
            final_path = os.path.join(cfg["logging"]["ckpt_dir"], "ckpt_final.pt")
            save_checkpoint(model, optimizer, scheduler, epochs, global_step, cfg, final_path)
            last_msg = "Training Completed Successfully."
            live.update(create_dashboard(progress, display_metrics, epochs, epochs, last_msg))
            logger.info("Training completed.")
            if cfg["logging"]["use_wandb"]:
                wandb.finish()
            live.stop()

    except Exception as e:
        if rank == 0:
            live.stop()
            console.print_exception()
            logger.error(f"Exception: {str(e)}", exc_info=True)
        raise e
    
    train_dataset.cleanup()
    dist.destroy_process_group()

def main():
    with open("train.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()

