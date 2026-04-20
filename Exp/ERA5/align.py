import argparse
import logging
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
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

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "align.yaml")


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
    parser.add_argument("--pretrained-ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


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


def _get_curriculum_sub_steps(epoch, total_epochs, all_sub_steps):
    sorted_steps = sorted(all_sub_steps)
    n = len(sorted_steps)
    progress = epoch / max(total_epochs - 1, 1)
    unlock_count = max(1, math.ceil(progress * n))
    if epoch < total_epochs // 3:
        unlock_count = max(1, n // 2)
    return sorted_steps[:unlock_count]


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


def build_lat_weights(h, device):
    lat = torch.linspace(-90, 90, h, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, h, 1)


def load_matching_pretrained_weights(model, ckpt_state):
    if ckpt_state is None:
        return [], []
    state_dict = ckpt_state.get("model", ckpt_state)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    target_state = model.state_dict()
    matched_state = {
        name: tensor
        for name, tensor in clean_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    missing = sorted(
        name for name in target_state.keys() if name not in matched_state
    )
    skipped = sorted(
        name for name, tensor in clean_state.items() if name not in matched_state
    )
    target_state.update(matched_state)
    model.load_state_dict(target_state)
    return missing, skipped


def load_alignment_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(clean_state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    saved_epoch = int(ckpt.get("epoch", -1))
    start_epoch = max(0, saved_epoch + 1)
    return start_epoch, int(ckpt.get("global_step", 0))


def align_step(
    model,
    batch,
    optimizer,
    cfg,
    grad_accum_steps,
    batch_idx,
    epoch,
    total_epochs,
):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()

    cond_steps = cfg["alignment"]["condition_steps"]
    max_tgt_steps = cfg["alignment"]["max_target_steps"]
    target_dt = cfg["alignment"]["target_dt"]
    all_sub_steps = cfg["alignment"]["sub_steps"]
    max_rollout_steps = cfg["alignment"]["max_rollout_steps"]
    chunk_size = cfg["alignment"]["chunk_size"]
    ensemble_size = cfg["model"].get("ensemble_size", 1)

    x_ctx = data[:, :cond_steps]
    x_targets = data[:, cond_steps:]
    dt_ctx = dt_data[:, :cond_steps]

    max_t = min(max_tgt_steps, x_targets.shape[1])
    available_sub_steps = _get_curriculum_sub_steps(
        epoch, total_epochs, all_sub_steps
    )
    sub_step = random.choice(available_sub_steps)
    max_t_for_step = max(1, min(max_t, max_rollout_steps // sub_step))
    target_t = random.randint(1, max_t_for_step)

    dt_per_iter = target_dt / sub_step
    n_iters = target_t * sub_step
    batch_size = x_ctx.shape[0]
    dt_list = [
        torch.full((batch_size,), dt_per_iter, device=device, dtype=torch.float32)
        for _ in range(n_iters)
    ]

    infer_model = model.module if hasattr(model, "module") else model
    output_offset = sub_step - 1
    x_tgt_aligned = x_targets[:, :target_t]
    lat_weights_cpu = build_lat_weights(
        x_tgt_aligned.shape[-2], torch.device("cpu")
    )
    target_cpu = x_tgt_aligned.detach().cpu()

    cached_preds = []
    cached_noises = []
    pred_sum_cpu = None
    mae_sum = torch.tensor(0.0)
    pairwise_sum = torch.tensor(0.0)

    for _ in range(ensemble_size):
        z_context = infer_model.sample_noise(x_ctx)
        z_rollout = infer_model.sample_rollout_noise(
            batch_size,
            n_iters,
            device,
            dtype=x_ctx.dtype,
        )
        with torch.no_grad():
            pred_seq = infer_model.forward_rollout(
                x_ctx,
                dt_ctx,
                dt_list,
                z_context=z_context,
                z_rollout=z_rollout,
                chunk_size=chunk_size,
                output_stride=sub_step,
                output_offset=output_offset,
            )
            pred_seq = pred_seq.real if pred_seq.is_complex() else pred_seq
            pred_seq = pred_seq[:, :target_t]
        pred_cpu = pred_seq.detach().cpu()
        pred_sum_cpu = (
            pred_cpu.clone() if pred_sum_cpu is None else pred_sum_cpu + pred_cpu
        )
        mae_sum = mae_sum + (pred_cpu - target_cpu).abs().mean()
        for prev_pred in cached_preds:
            pairwise_sum = pairwise_sum + (pred_cpu - prev_pred).abs().mean()
        cached_preds.append(pred_cpu)
        cached_noises.append((z_context, z_rollout))

    pred_mean_cpu = pred_sum_cpu / ensemble_size
    mse = ((pred_mean_cpu - target_cpu) ** 2 * lat_weights_cpu).mean()
    l1 = ((pred_mean_cpu - target_cpu).abs() * lat_weights_cpu).mean()
    crps = mae_sum / ensemble_size - pairwise_sum / (ensemble_size * ensemble_size)
    loss = crps

    basis_residual = torch.tensor(0.0, device=device)
    for block in infer_model.blocks:
        basis_dtype = complex_dtype_for(next(block.parameters()).dtype)
        W, W_inv = block.prop.basis.get_matrix(basis_dtype)
        eye = torch.eye(W.shape[0], device=device, dtype=W.dtype)
        basis_residual = basis_residual + (W @ W_inv - eye).abs().mean()
    basis_residual = basis_residual / max(len(infer_model.blocks), 1)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for member_idx, (z_context, z_rollout) in enumerate(cached_noises):
        pred_seq = infer_model.forward_rollout(
            x_ctx,
            dt_ctx,
            dt_list,
            z_context=z_context,
            z_rollout=z_rollout,
            chunk_size=chunk_size,
            output_stride=sub_step,
            output_offset=output_offset,
        )
        pred_seq = pred_seq.real if pred_seq.is_complex() else pred_seq
        pred_seq = pred_seq[:, :target_t]
        other_sum = (pred_sum_cpu - cached_preds[member_idx]).to(device)
        pred_mean = (pred_seq + other_sum) / ensemble_size
        del pred_mean
        crps_loss = (pred_seq - x_tgt_aligned).abs().mean() / ensemble_size
        if ensemble_size > 1:
            pairwise_i = torch.tensor(0.0, device=device)
            for other_idx, other_pred in enumerate(cached_preds):
                if other_idx == member_idx:
                    continue
                pairwise_i = pairwise_i + (
                    pred_seq - other_pred.to(device)
                ).abs().mean()
            crps_loss = crps_loss - pairwise_i / (ensemble_size * ensemble_size)
        (crps_loss / grad_accum_steps).backward()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["train"]["grad_clip"],
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss.item(),
        "l1": l1.item(),
        "crps": crps.item(),
        "rmse": torch.sqrt(mse).item(),
        "cond_steps": cond_steps,
        "target_t": target_t,
        "sub_step": sub_step,
        "n_iters": n_iters,
        "total_dt": target_t * target_dt,
        "dt_minutes": dt_per_iter * 60.0,
        "grad_norm": grad_norm,
        "basis_residual": basis_residual.item(),
    }


def save_checkpoint(model, optimizer, epoch, global_step, cfg, path):
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
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
    if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["train"]["grad_clip"],
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
    all_sub_steps = cfg["alignment"]["sub_steps"]
    total_epochs = cfg["train"]["epochs"]

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UniPhy Alignment Training")
        logger.info(f"  Condition Steps:   {cond_steps}")
        logger.info(f"  Max Target Steps:  {max_tgt_steps}")
        logger.info(f"  Sample K:          {sample_k}")
        logger.info(f"  Sub Steps:         {all_sub_steps}")
        logger.info(f"  Max Rollout Steps: {cfg['alignment']['max_rollout_steps']}")
        logger.info(f"  Chunk Size:        {cfg['alignment']['chunk_size']}")
        logger.info("=" * 60)

    pretrained_path = cfg["alignment"].get("pretrained_ckpt", "")
    pretrained_state = None
    if pretrained_path and os.path.exists(pretrained_path):
        pretrained_state = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        if rank == 0:
            logger.info(
                "Loaded Stage I checkpoint for initialization: "
                f"{pretrained_path}"
            )

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

    if pretrained_state is not None:
        missing, skipped = load_matching_pretrained_weights(model, pretrained_state)
        if rank == 0:
            logger.info(
                "Initialized from Stage I checkpoint with "
                "shape-compatible weights: "
                f"loaded={len(model.state_dict()) - len(missing)} "
                f"missing={len(missing)} skipped={len(skipped)}"
            )

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
        sample_k=sample_k,
        look_ahead=0,
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
    log_every = cfg["logging"]["log_every"]
    global_step = 0
    start_epoch = 0

    resume_path = cfg["logging"].get("ckpt", "")
    if resume_path:
        start_epoch, global_step = load_alignment_checkpoint(
            resume_path,
            model,
            optimizer,
        )
        if rank == 0:
            logger.info(f"Resumed alignment checkpoint: {resume_path}")

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    stop_early = False
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        unlocked = _get_curriculum_sub_steps(epoch, total_epochs, all_sub_steps)
        if rank == 0:
            min_dt_min = cfg["alignment"]["target_dt"] / max(unlocked) * 60
            logger.info(
                f"Epoch {epoch + 1}: unlocked sub_steps={unlocked}, "
                f"finest dt={min_dt_min:.1f}min"
            )

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
                f"Epoch {epoch + 1}/{epochs}", total=len(train_loader)
            )

        batch_idx = -1
        for batch_idx, batch in enumerate(train_loader):
            metrics = align_step(
                model,
                batch,
                optimizer,
                cfg,
                grad_accum_steps,
                batch_idx,
                epoch,
                total_epochs,
            )
            global_step += 1

            if rank == 0:
                progress.update(task_id, advance=1)
                if (batch_idx + 1) % log_every == 0:
                    log_msg = (
                        f"[E{epoch + 1:02d}] cond={metrics['cond_steps']} "
                        f"t={metrics['target_t']} sub={metrics['sub_step']} "
                        f"dt={metrics['dt_minutes']:.1f}min n={metrics['n_iters']} "
                        f"T={metrics['total_dt']:.0f}h | Loss: {metrics['loss']:.4f} "
                        f"L1: {metrics['l1']:.4f} CRPS: {metrics['crps']:.4f} "
                        f"RMSE: {metrics['rmse']:.4f}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

            if (
                cfg.get("runtime", {}).get("max_steps") is not None
                and global_step >= int(cfg["runtime"]["max_steps"])
            ):
                stop_early = True
                break

        flush_remaining_grads(model, optimizer, cfg, batch_idx, grad_accum_steps)

        if rank == 0:
            progress.stop()
            ckpt_save_path = os.path.join(
                cfg["logging"]["ckpt_dir"],
                f"align_epoch{epoch + 1}.pt",
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                cfg,
                ckpt_save_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_save_path}")

        dist.barrier()
        if stop_early:
            break

    if rank == 0:
        final_ckpt_path = os.path.join(
            cfg["logging"]["ckpt_dir"], "align_final.pt"
        )
        save_checkpoint(
            model,
            optimizer,
            max(start_epoch, epochs - 1),
            global_step,
            cfg,
            final_ckpt_path,
        )
        logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    train_dataset.cleanup()
    dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = build_runtime_cfg(
        args.config,
        data_input_dir=args.data_input_dir,
        train_year_range=args.train_year_range,
        sample_offsets_hours=args.sample_offsets_hours,
        epochs=args.epochs,
        log_path=args.log_path,
        ckpt_dir=args.ckpt_dir,
        ckpt_path=args.ckpt_path,
        pretrained_ckpt=args.pretrained_ckpt,
        max_steps=args.max_steps,
    )
    align(cfg)


if __name__ == "__main__":
    main()
