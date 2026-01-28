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
import wandb
import yaml
from rich.console import Console
from rich.text import Text
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

import warnings
warnings.filterwarnings("ignore")


class SpeedColumn(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return Text("0.00 it/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} it/s", style="progress.data.speed")


def setup_logging(log_path, rank):
    logger = logging.getLogger("align")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_path, "align_metrics.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
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
    target_exp = target.unsqueeze(0)
    mae = (pred_ensemble - target_exp).abs().mean()

    if M > 1:
        diff = (
            pred_ensemble.unsqueeze(1) - pred_ensemble.unsqueeze(0)
        ).abs().mean()
        loss = mae - 0.5 * diff
    else:
        loss = mae
    return loss


def align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, _ = batch
    data = data.to(device, non_blocking=True).float()

    B, T, C, H, W = data.shape
    ensemble_size = cfg["model"]["ensemble_size"]
    align_factors = cfg["alignment"]["sub_steps"]
    target_dt = cfg["alignment"]["target_dt"]
    max_k = cfg["alignment"].get("max_rollout_steps", 1)

    k = random.randint(1, max_k)

    x_input = data[:, 0:1]  
    x_target = data[:, k]   

    sub_steps = random.choice(align_factors)
    dt_step = target_dt / sub_steps
    
    total_steps = k * sub_steps
    dt_list = [
        torch.tensor(dt_step, device=device, dtype=torch.float32)
        for _ in range(total_steps)
    ]

    if hasattr(model, "module"):
        infer_model = model.module
    else:
        infer_model = model

    pred_seq = infer_model.forward_rollout(x_input, target_dt, dt_list)
    pred_final = pred_seq[:, -1]

    if pred_final.is_complex():
        pred_real = pred_final.real.contiguous()
    else:
        pred_real = pred_final.contiguous()

    lat_weights = get_lat_weights(H, W, device)

    l1_loss = (pred_real - x_target).abs().mean()
    mse_loss = ((pred_real - x_target) ** 2 * lat_weights).mean()

    if ensemble_size > 1:
        ensemble_preds = [pred_real]
        for _ in range(ensemble_size - 1):
            out_ens_seq = infer_model.forward_rollout(
                x_input, target_dt, dt_list
            )
            out_ens = out_ens_seq[:, -1]
            if out_ens.is_complex():
                ensemble_preds.append(out_ens.real.contiguous())
            else:
                ensemble_preds.append(out_ens.contiguous())

        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        crps_loss = compute_crps(ensemble_stack, x_target)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        loss = l1_loss + crps_loss
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

    metrics = {
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item(),
        "rmse": rmse.item(),
        "k_steps": k,
        "sub_steps": sub_steps,
        "total_dt": k * target_dt,
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item(),
    }
    return metrics


def save_checkpoint(model, optimizer, epoch, global_step, cfg, path):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "cfg": cfg,
    }
    torch.save(state, path)


def align(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    set_seed(42 + rank)

    if rank == 0:
        log_dir = cfg["logging"]["log_path"]
        os.makedirs(log_dir, exist_ok=True)
        console = Console()
        print(f"Alignment started. Logs: {log_dir}")
    else:
        console = Console(file=open(os.devnull, "w"))

    logger = setup_logging(cfg["logging"]["log_path"], rank)

    ckpt_path = cfg["alignment"].get("pretrained_ckpt", "")
    ckpt_state = None
    if ckpt_path and os.path.exists(ckpt_path):
        if rank == 0:
            logger.info(f"Loading configuration from {ckpt_path}")
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        if "cfg" in ckpt_state and "model" in ckpt_state["cfg"]:
            cfg["model"].update(ckpt_state["cfg"]["model"])
    elif ckpt_path:
        if rank == 0:
            logger.warning(f"Checkpoint {ckpt_path} not found!")

    if rank == 0 and cfg["logging"]["use_wandb"]:
        run_name = cfg["logging"]["wandb_run_name"] or \
            f"align_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        depth=cfg["model"]["depth"],
        patch_size=cfg["model"]["patch_size"],
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
        ensemble_size=cfg["model"]["ensemble_size"],
        max_growth_rate=cfg["model"]["max_growth_rate"],
    ).cuda()

    if ckpt_state is not None:
        if rank == 0:
            logger.info("Loading model weights from checkpoint")
        state_dict = ckpt_state["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    max_rollout_steps = cfg["alignment"].get("max_rollout_steps", 1)

    train_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=max_rollout_steps + 1,
        look_ahead=0,
        is_train=True,
        dt_ref=cfg["alignment"]["target_dt"],
        sampling_mode=cfg["data"]["sampling_mode"],
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
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    grad_accum_steps = cfg["train"]["grad_accum_steps"]
    epochs = cfg["train"]["epochs"]
    global_step = 0
    log_every = cfg["logging"]["log_every"]

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            SpeedColumn(),
            console=console
        )
        progress.start()
        task_id = progress.add_task(
            "Alignment", total=len(train_loader) * epochs
        )

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            metrics = align_step(
                model, batch, optimizer, cfg, grad_accum_steps, batch_idx
            )
            global_step += 1

            if rank == 0:
                progress.update(task_id, advance=1)
                if (batch_idx + 1) % log_every == 0:
                    log_msg = (
                        f"[E{epoch+1:02d}] "
                        f"K: {metrics['k_steps']} (dt={metrics['total_dt']:.0f}h) | "
                        f"RMSE: {metrics['rmse']:.4f} | "
                        f"Loss: {metrics['loss']:.4f}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

                if cfg["logging"]["use_wandb"]:
                    wandb.log({
                        "align/loss": metrics["loss"],
                        "align/rmse": metrics["rmse"],
                        "align/k_steps": metrics["k_steps"],
                        "align/total_dt": metrics["total_dt"],
                        "align/epoch": epoch,
                    }, step=global_step)

        if rank == 0:
            ckpt_path = os.path.join(
                cfg["logging"]["ckpt_dir"], f"align_epoch{epoch+1}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, global_step, cfg, ckpt_path
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

        dist.barrier()

    if rank == 0:
        progress.stop()
        if cfg["logging"]["use_wandb"]:
            wandb.finish()

    train_dataset.cleanup()
    dist.destroy_process_group()


def main():
    with open("align.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    align(cfg)


if __name__ == "__main__":
    main()

