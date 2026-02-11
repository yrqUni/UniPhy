import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
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

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

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
        handler = logging.FileHandler(os.path.join(log_path, "align.log"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_infer_model(model):
    return model.module if hasattr(model, "module") else model


def align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, _ = batch
    data = data.to(device, non_blocking=True).float()

    cond_steps = int(cfg["alignment"]["condition_steps"])
    max_tgt_steps = int(cfg["alignment"]["max_target_steps"])
    sub_steps_list = list(cfg["alignment"]["sub_steps"])

    target_dt = float(cfg["model"]["dt_ref"])

    x_ctx = data[:, :cond_steps]
    x_targets = data[:, cond_steps:]

    actual_tgt_steps = int(x_targets.shape[1])
    max_t = min(max_tgt_steps, actual_tgt_steps)
    t = random.randint(1, max_t)
    sub_step = int(random.choice(sub_steps_list))

    dt_per_iter = target_dt / float(sub_step)
    n_iters = int(t * sub_step)

    dt_list = [
        torch.tensor(dt_per_iter, device=device, dtype=torch.float32)
        for _ in range(n_iters)
    ]

    infer_model = get_infer_model(model)
    ensemble_size = int(cfg["model"]["ensemble_size"])
    bsz = int(x_ctx.shape[0])

    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (bsz,), device=device)
    else:
        member_idx = None

    pred_seq = infer_model.forward_rollout(
        x_ctx, target_dt, dt_list, member_idx=member_idx
    )

    if sub_step > 1:
        pred_aligned = pred_seq[:, sub_step - 1 :: sub_step, ...]
    else:
        pred_aligned = pred_seq

    pred_aligned = pred_aligned[:, :t]
    x_tgt_aligned = x_targets[:, :t]

    mse = ((pred_aligned - x_tgt_aligned) ** 2).mean()
    l1 = (pred_aligned - x_tgt_aligned).abs().mean()
    loss = l1

    (loss / float(grad_accum_steps)).backward()

    grad_norm = 0.0
    if (batch_idx + 1) % int(grad_accum_steps) == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["train"]["grad_clip"])
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    rmse = torch.sqrt(mse).item()

    metrics = {
        "loss": loss.item(),
        "l1": l1.item(),
        "rmse": rmse,
        "cond_steps": cond_steps,
        "target_t": t,
        "sub_step": sub_step,
        "n_iters": n_iters,
        "total_dt": float(t) * target_dt,
        "grad_norm": grad_norm,
        "member_idx": int(member_idx[0].item()) if member_idx is not None else -1,
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, global_step, cfg, path):
    state_dict = get_infer_model(model).state_dict()
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
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

    logger = setup_logging(cfg["logging"]["log_path"], rank)

    cond_steps = int(cfg["alignment"]["condition_steps"])
    max_tgt_steps = int(cfg["alignment"]["max_target_steps"])
    sample_k = cond_steps + max_tgt_steps

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UniPhy Alignment Training")
        logger.info(f"  Condition Steps: {cond_steps}")
        logger.info(f"  Max Target Steps: {max_tgt_steps}")
        logger.info(f"  Sample K: {sample_k}")
        logger.info(f"  Sub Steps: {cfg['alignment']['sub_steps']}")
        logger.info(f"  dt_ref: {float(cfg['model']['dt_ref'])}")
        logger.info(f"  ensemble_size: {int(cfg['model']['ensemble_size'])}")
        logger.info("=" * 60)

        if bool(cfg["logging"]["use_wandb"]):
            wandb.init(
                project=str(cfg["logging"]["wandb_project"]),
                entity=str(cfg["logging"]["wandb_entity"]),
                name=str(cfg["logging"]["wandb_run_name"]),
                config=cfg,
            )

    ckpt_path = str(cfg["alignment"].get("pretrained_ckpt", ""))
    ckpt_state = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt_state, dict) and "cfg" in ckpt_state and "model" in ckpt_state["cfg"]:
            cfg["model"].update(ckpt_state["cfg"]["model"])
        if rank == 0:
            logger.info(f"Loaded config from checkpoint: {ckpt_path}")

    model = UniPhyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        expand=int(cfg["model"]["expand"]),
        depth=int(cfg["model"]["depth"]),
        patch_size=tuple(cfg["model"]["patch_size"]),
        img_height=int(cfg["model"]["img_height"]),
        img_width=int(cfg["model"]["img_width"]),
        dt_ref=float(cfg["model"]["dt_ref"]),
        sde_mode=str(cfg["model"]["sde_mode"]),
        init_noise_scale=float(cfg["model"]["init_noise_scale"]),
        ensemble_size=int(cfg["model"]["ensemble_size"]),
        max_growth_rate=float(cfg["model"]["max_growth_rate"]),
    ).cuda()

    if ckpt_state is not None and "model" in ckpt_state:
        if rank == 0:
            logger.info("Loading model weights from checkpoint")
        state_dict = ckpt_state["model"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_dataset = ERA5_Dataset(
        input_dir=str(cfg["data"]["input_dir"]),
        year_range=list(cfg["data"]["year_range"]),
        window_size=int(cfg["data"]["window_size"]),
        sample_k=int(sample_k),
        look_ahead=0,
        is_train=True,
        dt_ref=float(cfg["model"]["dt_ref"]),
        sampling_mode=str(cfg["data"]["sampling_mode"]),
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
        batch_size=int(cfg["train"]["batch_size"]),
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

    grad_accum_steps = int(cfg["train"]["grad_accum_steps"])
    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["logging"]["log_every"])
    global_step = 0

    if rank == 0:
        os.makedirs(str(cfg["logging"]["ckpt_dir"]), exist_ok=True)

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
                model, batch, optimizer, cfg, grad_accum_steps, batch_idx
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
                        f"dt={metrics['total_dt']:.0f}h "
                        f"m={metrics['member_idx']} | "
                        f"Loss: {metrics['loss']:.4f} "
                        f"RMSE: {metrics['rmse']:.4f}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

                if bool(cfg["logging"]["use_wandb"]):
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
                            "align/member_idx": metrics["member_idx"],
                            "align/epoch": epoch,
                        },
                        step=global_step,
                    )

        if rank == 0:
            progress.stop()
            ckpt_save_path = os.path.join(
                str(cfg["logging"]["ckpt_dir"]),
                f"align_epoch{epoch + 1}.pt",
            )
            save_checkpoint(model, optimizer, epoch, global_step, cfg, ckpt_save_path)
            logger.info(f"Saved checkpoint: {ckpt_save_path}")

        dist.barrier()

    if rank == 0:
        final_ckpt_path = os.path.join(
            str(cfg["logging"]["ckpt_dir"]),
            "align_final.pt",
        )
        save_checkpoint(model, optimizer, epochs - 1, global_step, cfg, final_ckpt_path)
        logger.info(f"Saved final checkpoint: {final_ckpt_path}")

        if bool(cfg["logging"]["use_wandb"]):
            wandb.finish()

    train_dataset.cleanup()
    dist.destroy_process_group()


def main():
    with open("align.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    align(cfg)


if __name__ == "__main__":
    main()

