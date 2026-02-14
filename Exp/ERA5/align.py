import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
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


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def pad_dt(dt_between, t_len):
    B = dt_between.shape[0]
    if dt_between.shape[1] == t_len:
        return dt_between
    last = dt_between[:, -1:].expand(B, 1)
    return torch.cat([dt_between, last], dim=1)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)


def save_checkpoint(model, optimizer, cfg, path, epoch, step):
    if dist.get_rank() != 0:
        return
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "step": step,
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

    dt_ctx_between = dt_between[:, : max(cond_steps - 1, 1)]
    dt_ctx = pad_dt(dt_ctx_between, cond_steps)

    actual_tgt_steps = int(x_targets.shape[1])
    max_t = min(max_tgt_steps, actual_tgt_steps)
    t = random.randint(1, max_t)

    sub_step = int(random.choice(sub_steps_list))
    dt_per_iter = target_dt / float(sub_step)
    n_iters = int(t * sub_step)
    dt_future = torch.full((n_iters,), float(dt_per_iter), device=device, dtype=torch.float32)

    ensemble_size = int(cfg["model"]["ensemble_size"])
    B = int(x_ctx.shape[0])
    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (B,), device=device)
    else:
        member_idx = None

    infer_model = model.module
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
        "grad_norm": torch.tensor(grad_norm, device=device),
    }
    for k in metrics:
        metrics[k] = reduce_mean(metrics[k]).item()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    set_seed(int(cfg["train"]["seed"]) + rank)

    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("tf32", True))

    dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=int(cfg["data"]["window_size"]),
        sample_k=int(cfg["data"]["sample_k"]),
        look_ahead=int(cfg["data"]["look_ahead"]),
        is_train=True,
        frame_hours=float(cfg["data"].get("frame_hours", 6.0)),
        sampling_mode=str(cfg["data"].get("sampling_mode", "mixed")),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        num_workers=int(cfg["train"]["num_workers"]),
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

    load_checkpoint(model, cfg["alignment"]["pretrained_ckpt"])
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 1))
    epochs = int(cfg["train"]["epochs"])
    ckpt_dir = cfg["train"].get("ckpt_dir", "./ckpt_align")

    step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running = {}
        for batch_idx, batch in enumerate(loader):
            metrics = align_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
            step += 1
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v)
            if step % log_every == 0 and dist.get_rank() == 0:
                denom = float(log_every)
                msg = f"epoch={epoch} step={step}"
                for k in sorted(running.keys()):
                    msg += f" {k}={running[k] / denom:.6f}"
                msg += f" time={time.time() - t0:.2f}s"
                print(msg, flush=True)
                running = {}
                t0 = time.time()
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                cfg,
                os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1}.pt"),
                epoch,
                step,
            )

    save_checkpoint(model, optimizer, cfg, os.path.join(ckpt_dir, "ckpt_final.pt"), epochs - 1, step)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
