import argparse
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
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
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__ in {None, ""}:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import build_runtime_cfg
from Model.UniPhy.ModelUniPhy import UniPhyModel
from Model.UniPhy.UniPhyOps import complex_dtype_for

warnings.filterwarnings("ignore")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "train.yaml")


class SpeedColumn(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return Text("0.00 it/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} it/s", style="progress.data.speed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--data-input-dir", default=None)
    parser.add_argument("--train-year-range", default=None)
    parser.add_argument("--sample-offsets-hours", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def setup_logging(log_path, rank):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_path, "train_metrics.log"))
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


def build_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, H, 1)


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


def train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx, lat_weights):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()

    ensemble_size = cfg["model"]["ensemble_size"]
    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt_data[:, 1:]

    infer_model = model.module if hasattr(model, "module") else model

    ensemble_preds = []
    for _ in range(ensemble_size):
        noise = infer_model.sample_noise(x_input)
        out = model(x_input, dt_input, z=noise)
        ensemble_preds.append(out.real if out.is_complex() else out)

    ensemble_stack = torch.stack(ensemble_preds, dim=0)
    out_mean = ensemble_stack.mean(dim=0)

    l1_loss = ((out_mean - x_target).abs() * lat_weights).mean()
    mse_loss = ((out_mean - x_target) ** 2 * lat_weights).mean()

    if ensemble_size > 1:
        crps_loss = compute_crps(ensemble_stack, x_target)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        loss = l1_loss + crps_loss
    else:
        crps_loss = torch.tensor(0.0, device=device)
        ensemble_std = torch.tensor(0.0, device=device)
        loss = l1_loss

    basis_reg_weight = cfg["train"].get("basis_reg_weight", 0.01)
    basis_reg = torch.tensor(0.0, device=device)
    for block in infer_model.blocks:
        basis_dtype = complex_dtype_for(next(block.parameters()).dtype)
        W, W_inv = block.prop.basis.get_matrix(basis_dtype)
        eye = torch.eye(W.shape[0], device=device, dtype=W.dtype)
        basis_reg = basis_reg + (W @ W_inv - eye).abs().pow(2).mean()
    basis_reg = basis_reg / max(1, len(infer_model.blocks))
    loss = loss + basis_reg_weight * basis_reg

    (loss / grad_accum_steps).backward()

    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["train"]["grad_clip"],
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item(),
        "mse": mse_loss.item(),
        "rmse": torch.sqrt(mse_loss).item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item(),
    }
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, path):
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "cfg": cfg,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    clean_state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(clean_state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    saved_epoch = int(ckpt.get("epoch", -1))
    start_epoch = max(0, saved_epoch + 1)
    return start_epoch, ckpt.get("global_step", 0)


def flush_remaining_grads(model, optimizer, cfg, batch_idx, grad_accum_steps):
    if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["train"]["grad_clip"],
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def train(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    set_seed(42 + rank)

    devnull = open(os.devnull, "w")
    if rank == 0:
        os.makedirs(cfg["logging"]["log_path"], exist_ok=True)
        console = Console() if sys.stdout.isatty() else Console(file=devnull)
    else:
        console = Console(file=devnull)

    logger = setup_logging(cfg["logging"]["log_path"], rank)
    if rank == 0:
        logger.info(f"Training started on {world_size} GPUs")

    if cfg["train"]["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        expand=cfg["model"]["expand"],
        depth=cfg["model"]["depth"],
        patch_size=tuple(cfg["model"]["patch_size"]),
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
    ).cuda()

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers=False,
    )

    train_dataset = ERA5Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=cfg["model"]["dt_ref"],
        sample_offsets_hours=cfg["data"].get("sample_offsets_hours"),
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
        weight_decay=float(cfg["train"]["weight_decay"]),
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
    ckpt_path = cfg["logging"].get("ckpt")
    if ckpt_path:
        start_epoch, global_step = load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
        )
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {ckpt_path}")

    log_every = cfg["logging"]["log_every"]
    save_interval = max(1, int(len(train_loader) * cfg["logging"]["ckpt_step"]))
    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    lat_weights = build_lat_weights(
        cfg["model"]["img_height"],
        cfg["model"]["img_width"],
        torch.device(f"cuda:{local_rank}"),
    )

    progress = None
    task_id = None
    if rank == 0:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            SpeedColumn(),
            console=console,
        )
        progress.start()
        task_id = progress.add_task(
            "Training",
            total=len(train_loader) * epochs,
            completed=global_step,
        )

    stop_early = False
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_idx = -1
        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(
                model,
                batch,
                optimizer,
                cfg,
                grad_accum_steps,
                batch_idx,
                lat_weights,
            )
            global_step += 1

            if scheduler is not None and (batch_idx + 1) % grad_accum_steps == 0:
                scheduler.step()

            if rank == 0:
                progress.update(task_id, advance=1)
                if (batch_idx + 1) % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"[E{epoch + 1:03d} B{batch_idx + 1:04d}] "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"L1: {metrics['l1_loss']:.4f} | "
                        f"CRPS: {metrics['crps_loss']:.4f} | "
                        f"RMSE: {metrics['rmse']:.4f} | "
                        f"Grad: {metrics['grad_norm']:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)
                if save_interval > 0 and (batch_idx + 1) % save_interval == 0:
                    save_path = os.path.join(
                        cfg["logging"]["ckpt_dir"],
                        f"ckpt_e{epoch + 1}_s{global_step}.pt",
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        cfg,
                        save_path,
                    )
                    logger.info(f"Saved checkpoint: {save_path}")

            if (
                cfg.get("runtime", {}).get("max_steps") is not None
                and global_step >= int(cfg["runtime"]["max_steps"])
            ):
                stop_early = True
                break

        flush_remaining_grads(
            model,
            optimizer,
            cfg,
            batch_idx,
            grad_accum_steps,
        )

        if rank == 0:
            epoch_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"ckpt_epoch{epoch + 1}.pt",
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                epoch_path,
            )
            logger.info(f"Epoch {epoch + 1} finished. Saved checkpoint.")

        dist.barrier()
        if stop_early:
            break

    if rank == 0:
        final_path = os.path.join(cfg["logging"]["ckpt_dir"], "ckpt_final.pt")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epochs,
            global_step,
            cfg,
            final_path,
        )
        logger.info(f"Training Completed. Final checkpoint: {final_path}")
        progress.console.print(
            f"[bold green]Training Completed. Final checkpoint: {final_path}"
        )
        progress.stop()

    train_dataset.cleanup()
    dist.destroy_process_group()


def main():
    args = parse_args()
    resume_path = args.resume or args.ckpt_path
    cfg = build_runtime_cfg(
        args.config,
        data_input_dir=args.data_input_dir,
        train_year_range=args.train_year_range,
        sample_offsets_hours=args.sample_offsets_hours,
        epochs=args.epochs,
        log_path=args.log_path,
        ckpt_dir=args.ckpt_dir,
        ckpt_path=resume_path,
        max_steps=args.max_steps,
    )
    train(cfg)


if __name__ == "__main__":
    main()
