import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return not is_dist() or dist.get_rank() == 0


def reduce_mean(tensor):
    if not is_dist():
        return tensor
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def pad_dt(dt_between, t_len):
    if dt_between.shape[1] == t_len:
        return dt_between
    bsz = dt_between.shape[0]
    last = dt_between[:, -1:].expand(bsz, 1)
    return torch.cat([dt_between, last], dim=1)


def compute_crps(ensemble_preds, target):
    y = target.unsqueeze(0).expand_as(ensemble_preds)
    abs_err = torch.abs(ensemble_preds - y).mean(dim=0)
    pairwise = torch.abs(ensemble_preds.unsqueeze(0) - ensemble_preds.unsqueeze(1)).mean(dim=(0, 1))
    return abs_err - 0.5 * pairwise


def train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, dt_between = batch
    data = data.to(device, non_blocking=True).float()
    dt_between = dt_between.to(device, non_blocking=True).float()

    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_in = pad_dt(dt_between, x_input.shape[1])

    ensemble_size = int(cfg["model"].get("ensemble_size", 1))
    bsz = x_input.shape[0]
    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (bsz,), device=device)
    else:
        member_idx = None

    out = model(x_input, dt_in, member_idx=member_idx)
    l1 = torch.abs(out - x_target).mean()

    if ensemble_size > 1:
        preds = []
        for m in range(ensemble_size):
            idx = torch.full((bsz,), m, device=device, dtype=torch.long)
            preds.append(model(x_input, dt_in, member_idx=idx))
        ensemble_preds = torch.stack(preds, dim=0)
        crps = compute_crps(ensemble_preds, x_target).mean()
    else:
        crps = torch.zeros((), device=device)

    crps_weight = float(cfg["train"].get("crps_weight", 0.0))
    loss = l1 + crps_weight * crps

    (loss / float(grad_accum_steps)).backward()

    grad_norm = 0.0
    if (batch_idx + 1) % int(grad_accum_steps) == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(cfg["train"]["grad_clip"]),
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    mse = F.mse_loss(out, x_target)
    metrics = {
        "loss": loss.detach(),
        "l1": l1.detach(),
        "crps": crps.detach(),
        "rmse": torch.sqrt(mse).detach(),
        "grad_norm": torch.tensor(grad_norm, device=device),
    }
    for k in metrics:
        metrics[k] = reduce_mean(metrics[k]).item()
    return metrics


def save_checkpoint(model, optimizer, cfg, path, epoch, step):
    if not is_main():
        return
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "step": step,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    use_tf32 = bool(cfg.get("train", {}).get("use_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    frame_hours = float(cfg.get("data", {}).get("frame_hours", cfg["model"].get("dt_ref", 6.0)))

    dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=int(cfg["data"]["window_size"]),
        sample_k=int(cfg["data"]["sample_k"]),
        look_ahead=int(cfg["data"]["look_ahead"]),
        is_train=True,
        frame_hours=frame_hours,
        sampling_mode=str(cfg["data"].get("sampling_mode", "mixed")),
    )

    try:
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = None

        num_workers = int(cfg.get("train", {}).get("num_workers", 4))

        loader = DataLoader(
            dataset,
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

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"]["lr"]),
            betas=(0.9, 0.95),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )

        epochs = int(cfg["train"]["epochs"])
        grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))

        log_every = int(cfg.get("logging", {}).get("log_every", 50))
        ckpt_dir = cfg.get("logging", {}).get("ckpt_dir", "./ckpt")
        ckpt_every_epochs = int(cfg.get("logging", {}).get("save_every", 1))

        step = 0
        running = {}
        t0 = time.time()

        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            model.train()

            for batch_idx, batch in enumerate(loader):
                metrics = train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
                step += 1

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + float(v)

                if is_main() and step % log_every == 0:
                    denom = float(log_every)
                    msg = f"epoch={epoch} step={step}"
                    for k in sorted(running.keys()):
                        msg += f" {k}={running[k] / denom:.6f}"
                    msg += f" time={time.time() - t0:.2f}s"
                    print(msg, flush=True)
                    running = {}
                    t0 = time.time()

            if (epoch + 1) % max(1, ckpt_every_epochs) == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1}.pt"),
                    epoch,
                    step,
                )

        save_checkpoint(
            model,
            optimizer,
            cfg,
            os.path.join(ckpt_dir, "ckpt_final.pt"),
            epochs - 1,
            step,
        )

        if is_dist():
            dist.barrier()

    finally:
        dataset.cleanup()
        if is_dist():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
