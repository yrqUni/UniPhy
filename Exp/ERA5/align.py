import argparse
import datetime
import os
import sys
import random
from pathlib import Path

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def reduce_mean(tensor):
    if not is_dist():
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= float(get_world_size())
    return out


def pad_dt(dt_between, t_len):
    if dt_between.ndim == 1:
        dt_between = dt_between.unsqueeze(0)
    if dt_between.shape[1] == t_len:
        return dt_between
    bsz = dt_between.shape[0]
    if dt_between.shape[1] == 0:
        return torch.zeros((bsz, t_len), device=dt_between.device, dtype=dt_between.dtype)
    last = dt_between[:, -1:].expand(bsz, 1)
    while dt_between.shape[1] < t_len:
        dt_between = torch.cat([dt_between, last], dim=1)
    return dt_between[:, :t_len]


def get_infer_model(model):
    return model.module if hasattr(model, "module") else model


def init_wandb(cfg):
    log_cfg = cfg.get("logging", {})
    if not bool(log_cfg.get("use_wandb", False)):
        return False
    project = str(log_cfg.get("wandb_project", "UniPhy-Align"))
    entity = str(log_cfg.get("wandb_entity", ""))
    run_name = str(log_cfg.get("wandb_run_name", "")).strip()
    if not run_name:
        run_name = datetime.datetime.now().strftime("align-%Y%m%d-%H%M%S")
    wandb.init(project=project, entity=entity or None, name=run_name, config=cfg)
    return True


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)


def save_checkpoint(model, optimizer, epoch, global_step, cfg, path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "cfg": cfg,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, dt_between = batch
    data = data.to(device, non_blocking=True).float()
    dt_between = dt_between.to(device, non_blocking=True).float()

    cond_steps = int(cfg["alignment"]["condition_steps"])
    max_tgt_steps = int(cfg["alignment"]["max_target_steps"])
    sub_steps_list = list(cfg["alignment"]["sub_steps"])

    target_dt = float(cfg["model"].get("target_dt_hours", cfg["model"].get("dt_ref", 6.0)))

    x_ctx = data[:, :cond_steps]
    x_targets = data[:, cond_steps:]

    dt_ctx_between = dt_between[:, : max(cond_steps - 1, 0)]
    dt_ctx = pad_dt(dt_ctx_between, cond_steps)

    actual_tgt_steps = int(x_targets.shape[1])
    max_t = min(int(max_tgt_steps), int(actual_tgt_steps))
    if max_t <= 0:
        metrics = {
            "loss": torch.zeros((), device=device),
            "l1": torch.zeros((), device=device),
            "rmse": torch.zeros((), device=device),
            "t": torch.zeros((), device=device),
            "sub_step": torch.zeros((), device=device),
            "grad_norm": torch.zeros((), device=device),
        }
        for k in metrics:
            metrics[k] = reduce_mean(metrics[k]).item()
        return metrics

    t = random.randint(1, max_t)
    sub_step = int(random.choice(sub_steps_list))

    dt_per_iter = target_dt / float(sub_step)
    n_iters = int(t * sub_step)
    dt_future = torch.full((n_iters,), float(dt_per_iter), device=device, dtype=torch.float32)

    ensemble_size = int(cfg["model"].get("ensemble_size", 1))
    bsz = int(x_ctx.shape[0])
    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (bsz,), device=device)
    else:
        member_idx = None

    infer_model = get_infer_model(model)
    pred_seq = infer_model.forward_rollout(x_ctx, dt_ctx, dt_future, member_idx=member_idx)

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
            model.parameters(),
            float(cfg["train"]["grad_clip"]),
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.detach(),
        "l1": l1.detach(),
        "rmse": torch.sqrt(mse).detach(),
        "t": torch.tensor(float(t), device=device),
        "sub_step": torch.tensor(float(sub_step), device=device),
        "grad_norm": torch.tensor(float(grad_norm), device=device),
    }
    for k in metrics:
        metrics[k] = reduce_mean(metrics[k]).item()
    return metrics


def align(cfg):
    rank, world_size, local_rank = setup_distributed()

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    use_tf32 = bool(cfg.get("train", {}).get("use_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    frame_hours = float(cfg.get("data", {}).get("frame_hours", cfg["model"].get("dt_ref", 6.0)))

    cond_steps = int(cfg["alignment"]["condition_steps"])
    max_target_steps = int(cfg["alignment"]["max_target_steps"])
    sample_k_default = max(2, cond_steps + max_target_steps + 1)

    sample_k = int(cfg.get("data", {}).get("sample_k", sample_k_default))
    look_ahead = int(cfg.get("data", {}).get("look_ahead", 0))

    align_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=int(cfg["data"]["window_size"]),
        sample_k=sample_k,
        look_ahead=look_ahead,
        is_train=True,
        frame_hours=frame_hours,
        sampling_mode=str(cfg["data"].get("sampling_mode", "mixed")),
    )

    use_wandb = init_wandb(cfg) if is_main() else False

    try:
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                align_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

        num_workers = int(cfg.get("train", {}).get("num_workers", 4))

        align_loader = DataLoader(
            align_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        model = UniPhyModel(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            embed_dim=int(cfg["model"]["embed_dim"]),
            expand=int(cfg["model"]["expand"]),
            depth=int(cfg["model"]["depth"]),
            patch_size=cfg["model"]["patch_size"],
            img_height=int(cfg["model"]["img_height"]),
            img_width=int(cfg["model"]["img_width"]),
            tau_ref_hours=float(cfg["model"].get("tau_ref_hours", cfg["model"].get("dt_ref", 6.0))),
            sde_mode=str(cfg["model"].get("sde_mode", "sde")),
            init_noise_scale=float(cfg["model"].get("init_noise_scale", 0.01)),
            max_growth_rate=float(cfg["model"].get("max_growth_rate", 0.3)),
            ensemble_size=int(cfg["model"].get("ensemble_size", 1)),
        ).to(device)

        pretrained_ckpt = str(cfg["alignment"]["pretrained_ckpt"])
        load_checkpoint(model, pretrained_ckpt)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, gradient_as_bucket_view=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"]["lr"]),
            betas=(0.9, 0.95),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )

        epochs = int(cfg["train"]["epochs"])
        grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))

        log_cfg = cfg.get("logging", {})
        log_every = int(log_cfg.get("log_every", 50))
        wandb_every = int(log_cfg.get("wandb_every", 50))
        ckpt_step_frac = float(log_cfg.get("ckpt_step", 0.0))
        ckpt_dir = str(log_cfg.get("ckpt_dir", "./align_ckpt"))

        steps_per_epoch = max(1, len(align_loader))
        ckpt_every_steps = 0
        if ckpt_step_frac > 0:
            ckpt_every_steps = max(1, int(round(steps_per_epoch * ckpt_step_frac)))

        global_step = 0
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            model.train()
            running = {}

            for batch_idx, batch in enumerate(align_loader):
                metrics = align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
                global_step += 1

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + float(v)

                if is_main() and global_step % log_every == 0:
                    denom = float(log_every)
                    msg = f"epoch={epoch} step={global_step}"
                    for k in sorted(running.keys()):
                        msg += f" {k}={running[k] / denom:.6f}"
                    print(msg, flush=True)
                    running = {}

                if use_wandb and global_step % wandb_every == 0:
                    wandb.log({f"align/{k}": float(v) for k, v in metrics.items()}, step=global_step)

                if is_main() and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        cfg,
                        os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt"),
                    )

            if is_main():
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    cfg,
                    os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1}.pt"),
                )

        if is_main():
            save_checkpoint(
                model,
                optimizer,
                epochs - 1,
                global_step,
                cfg,
                os.path.join(ckpt_dir, "ckpt_final.pt"),
            )

        if is_dist():
            dist.barrier()

    finally:
        align_dataset.cleanup()
        if use_wandb:
            wandb.finish()
        if is_dist():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="align.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    align(cfg)


if __name__ == "__main__":
    main()
