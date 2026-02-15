import os
import sys
import random
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import yaml
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

from ERA5 import ERA5Dataset
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
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_path, "align.log"))
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


def align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()

    cond_steps = cfg["alignment"]["condition_steps"]
    max_tgt_steps = cfg["alignment"]["max_target_steps"]
    target_dt = cfg["alignment"]["target_dt"]
    sub_steps_list = cfg["alignment"]["sub_steps"]

    x_ctx = data[:, :cond_steps]
    x_targets = data[:, cond_steps:]
    dt_ctx = dt_data[:, :cond_steps]

    actual_tgt_steps = x_targets.shape[1]
    max_t = min(max_tgt_steps, actual_tgt_steps)

    t = random.randint(1, max_t)
    sub_step = random.choice(sub_steps_list)

    dt_per_iter = target_dt / sub_step
    n_iters = t * sub_step

    B = x_ctx.shape[0]
    dt_list = [
        torch.full((B,), dt_per_iter, device=device, dtype=torch.float32)
        for _ in range(n_iters)
    ]

    infer_model = model.module if hasattr(model, "module") else model

    pred_seq = infer_model.forward_rollout(x_ctx, dt_ctx, dt_list)

    if sub_step > 1:
        pred_aligned = pred_seq[:, sub_step - 1::sub_step]
    else:
        pred_aligned = pred_seq

    pred_aligned = pred_aligned[:, :t]
    x_tgt_aligned = x_targets[:, :t]

    mse = ((pred_aligned - x_tgt_aligned) ** 2).mean()
    l1 = (pred_aligned - x_tgt_aligned).abs().mean()

    loss = l1

    loss_scaled = loss / grad_accum_steps
    loss_scaled.backward()

    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        if dist.is_initialized():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"],
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        rmse = torch.sqrt(mse).item()

    metrics = {
        "loss": loss.item(),
        "l1": l1.item(),
        "rmse": rmse,
        "cond_steps": cond_steps,
        "target_t": t,
        "sub_step": sub_step,
        "n_iters": n_iters,
        "total_dt": t * target_dt,
        "grad_norm": grad_norm,
    }
    return metrics


def save_checkpoint(model, optimizer, epoch, global_step, cfg, path):
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "cfg": cfg,
    }
    torch.save(state, path)


def flush_remaining_grads(model, optimizer, cfg, batch_idx, grad_accum_steps):
    if (batch_idx + 1) % grad_accum_steps != 0:
        if dist.is_initialized():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"],
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def align(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    set_seed(42 + rank)

    logger = setup_logging(cfg["logging"]["log_path"], rank)

    cond_steps = cfg["alignment"]["condition_steps"]
    max_tgt_steps = cfg["alignment"]["max_target_steps"]
    sample_k = cond_steps + max_tgt_steps

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UniPhy Alignment Training")
        logger.info(f"  Condition Steps: {cond_steps}")
        logger.info(f"  Max Target Steps: {max_tgt_steps}")
        logger.info(f"  Sample K: {sample_k}")
        logger.info(f"  Sub Steps: {cfg['alignment']['sub_steps']}")
        logger.info("=" * 60)

        if cfg["logging"]["use_wandb"]:
            wandb.init(
                project=cfg["logging"]["wandb_project"],
                entity=cfg["logging"]["wandb_entity"],
                name=cfg["logging"]["wandb_run_name"],
                config=cfg,
            )

    ckpt_path = cfg["alignment"].get("pretrained_ckpt", "")
    ckpt_state = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_state = torch.load(
            ckpt_path, map_location="cpu", weights_only=False,
        )
        if "cfg" in ckpt_state and "model" in ckpt_state["cfg"]:
            cfg["model"].update(ckpt_state["cfg"]["model"])
        if rank == 0:
            logger.info(f"Loaded config from checkpoint: {ckpt_path}")

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
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
        ensemble_size=cfg["model"]["ensemble_size"],
        max_growth_rate=cfg["model"]["max_growth_rate"],
    ).cuda()

    if ckpt_state is not None:
        if rank == 0:
            logger.info("Loading model weights from checkpoint")
        state_dict = ckpt_state["model"]
        clean_state = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state, strict=False)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_dataset = ERA5Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=sample_k,
        look_ahead=0,
        is_train=True,
        dt_ref=cfg["model"]["dt_ref"],
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
    log_every = cfg["logging"]["log_every"]
    global_step = 0

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        progress = None
        task_id = None
        if rank == 0:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                SpeedColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            task_id = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}",
                total=len(train_loader),
            )

        for batch_idx, batch in enumerate(train_loader):
            metrics = align_step(
                model, batch, optimizer, cfg, grad_accum_steps, batch_idx,
            )
            global_step += 1

            if rank == 0:
                progress.update(task_id, advance=1)

                if (batch_idx + 1) % log_every == 0:
                    log_msg = (
                        f"[E{epoch + 1:02d}] "
                        f"cond={metrics['cond_steps']} "
                        f"t={metrics['target_t']} "
                        f"sub={metrics['sub_step']} "
                        f"n={metrics['n_iters']} "
                        f"dt={metrics['total_dt']:.0f}h | "
                        f"Loss: {metrics['loss']:.4f} "
                        f"RMSE: {metrics['rmse']:.4f}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

                if cfg["logging"]["use_wandb"]:
                    wandb.log(
                        {
                            "align/loss": metrics["loss"],
                            "align/l1": metrics["l1"],
                            "align/rmse": metrics["rmse"],
                            "align/cond_steps": metrics["cond_steps"],
                            "align/target_t": metrics["target_t"],
                            "align/sub_step": metrics["sub_step"],
                            "align/n_iters": metrics["n_iters"],
                            "align/total_dt": metrics["total_dt"],
                            "align/grad_norm": metrics["grad_norm"],
                            "align/epoch": epoch,
                        },
                        step=global_step,
                    )

        flush_remaining_grads(
            model, optimizer, cfg, batch_idx, grad_accum_steps,
        )

        if rank == 0:
            progress.stop()
            ckpt_save_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"align_epoch{epoch + 1}.pt",
            )
            save_checkpoint(
                model, optimizer, epoch, global_step, cfg, ckpt_save_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_save_path}")

        dist.barrier()

    if rank == 0:
        final_ckpt_path = os.path.join(
            cfg["logging"]["ckpt_dir"], "align_final.pt",
        )
        save_checkpoint(
            model, optimizer, epochs - 1, global_step, cfg, final_ckpt_path,
        )
        logger.info(f"Saved final checkpoint: {final_ckpt_path}")

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
    