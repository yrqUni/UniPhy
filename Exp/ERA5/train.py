import argparse
import datetime
import os
import random
import sys
import time

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
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


def get_lat_weights(h, w, device):
    lat = torch.linspace(-90.0, 90.0, h, device=device)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    return weights.reshape(1, 1, 1, h, 1).expand(1, 1, 1, h, w)


def compute_crps(ensemble_preds, target):
    y = target.unsqueeze(0).expand_as(ensemble_preds)
    abs_err = torch.abs(ensemble_preds - y).mean(dim=0)
    pairwise = torch.abs(ensemble_preds.unsqueeze(0) - ensemble_preds.unsqueeze(1)).mean(dim=(0, 1))
    return abs_err.mean() - 0.5 * pairwise.mean()


def train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx):
    device = next(model.parameters()).device
    data, dt_step = batch
    data = data.to(device, non_blocking=True).float()
    dt_step = dt_step.to(device, non_blocking=True).float()

    bsz, k_len, _, hgt, wdt = data.shape
    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt_step

    ensemble_size = int(cfg["model"].get("ensemble_size", 1))
    if ensemble_size > 1:
        member_idx = torch.randint(0, ensemble_size, (bsz,), device=device)
    else:
        member_idx = None

    out = model(x_input, dt_input, member_idx=member_idx)
    out_real = out.real.contiguous() if out.is_complex() else out.contiguous()
    lat_weights = get_lat_weights(hgt, wdt, device)

    l1_loss = (out_real - x_target).abs().mean()
    mse_loss = ((out_real - x_target) ** 2 * lat_weights).mean()

    if ensemble_size > 1:
        infer_model = model.module if hasattr(model, "module") else model
        preds = [out_real]
        with torch.no_grad():
            for _ in range(ensemble_size - 1):
                rand_idx = torch.randint(0, ensemble_size, (bsz,), device=device)
                out_ens = infer_model(x_input, dt_input, member_idx=rand_idx)
                preds.append(out_ens.real.contiguous() if out_ens.is_complex() else out_ens.contiguous())
        ensemble_stack = torch.stack(preds, dim=0)
        crps_loss = compute_crps(ensemble_stack, x_target)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        loss = l1_loss + crps_loss
    else:
        crps_loss = torch.zeros((), device=device)
        ensemble_std = torch.zeros((), device=device)
        loss = l1_loss

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
        "l1": l1_loss.detach(),
        "mse": mse_loss.detach(),
        "crps": crps_loss.detach(),
        "ens_std": ensemble_std.detach(),
        "grad_norm": torch.tensor(float(grad_norm), device=device),
    }
    for k in metrics:
        metrics[k] = reduce_mean(metrics[k]).item()
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, cfg, path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    state = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "cfg": cfg,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    if isinstance(ckpt, dict) and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if isinstance(ckpt, dict) and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    return epoch, global_step


def init_wandb(cfg):
    log_cfg = cfg.get("logging", {})
    if not bool(log_cfg.get("use_wandb", False)):
        return False
    project = str(log_cfg.get("wandb_project", "UniPhy"))
    entity = str(log_cfg.get("wandb_entity", ""))
    run_name = str(log_cfg.get("wandb_run_name", "")).strip()
    if not run_name:
        run_name = datetime.datetime.now().strftime("train-%Y%m%d-%H%M%S")
    wandb.init(project=project, entity=entity or None, name=run_name, config=cfg)
    return True


def train(cfg):
    rank, world_size, local_rank = setup_distributed()
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    use_tf32 = bool(cfg.get("train", {}).get("use_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    frame_hours = float(cfg.get("data", {}).get("frame_hours", cfg["model"].get("dt_ref", 6.0)))
    train_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=int(cfg["data"]["window_size"]),
        sample_k=int(cfg["data"]["sample_k"]),
        look_ahead=int(cfg["data"].get("look_ahead", 0)),
        is_train=True,
        frame_hours=frame_hours,
        sampling_mode=str(cfg["data"].get("sampling_mode", "mixed")),
    )

    use_wandb = init_wandb(cfg) if is_main() else False
    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )

    model = UniPhyModel(**cfg["model"]).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched_cfg = cfg.get("scheduler", {})
    scheduler = None
    if bool(sched_cfg.get("use_scheduler", False)):
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))

    start_epoch = 0
    global_step = 0
    ckpt_path = str(cfg["train"].get("resume", "")).strip()
    if ckpt_path:
        start_epoch, global_step = load_checkpoint(ckpt_path, model, optimizer, scheduler)

    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 10))
    save_every = int(cfg["train"].get("save_every", 1))
    out_dir = str(cfg["train"].get("out_dir", "./outputs"))

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            metrics = train_step(model, batch, optimizer, cfg, grad_accum_steps, batch_idx)
            global_step += 1
            if is_main() and global_step % log_every == 0:
                metrics_out = dict(metrics)
                metrics_out["epoch"] = epoch
                metrics_out["step"] = global_step
                metrics_out["time"] = time.time() - t0
                if use_wandb:
                    wandb.log(metrics_out, step=global_step)
                else:
                    print(metrics_out)
        if scheduler is not None:
            scheduler.step()
        if is_main() and (epoch + 1) % save_every == 0:
            ckpt_name = f"ckpt-epoch{epoch+1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, cfg, os.path.join(out_dir, ckpt_name))

    if use_wandb and is_main():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
