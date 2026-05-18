import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import (
    build_lat_weights,
    build_progress,
    build_rank_console,
    build_runtime_cfg,
    compute_basis_residual,
    compute_weighted_crps,
    distributed_barrier,
    flush_remaining_grads,
    get_unwrapped_model,
    init_distributed,
    setup_file_logger,
    should_stop_early,
    wrap_ddp,
)
from Exp.Ablation.protocol import build_run_manifest, write_json
from Exp.Ablation.variants import (
    build_variant,
    build_variant_optimizer,
    describe_variant,
)

CONFIG_PATH = str(ROOT / "Exp" / "ERA5" / "train.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="UniPhy ablation training runner")
    p.add_argument("--variant", required=True, help="Ablation variant name")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--data-input-dir", default=None)
    p.add_argument("--train-year-range", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--log-path", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpt-path", default=None, help="Resume from checkpoint")
    p.add_argument("--pretrained-ckpt", default=None, help="Warm-start Stage I ckpt")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_step(
    model, batch, optimizer, cfg, grad_accum_steps, batch_idx, lat_weights, variant
):
    device = next(model.parameters()).device
    data, dt_data = batch
    data = data.to(device, non_blocking=True).float()
    dt_data = dt_data.to(device, non_blocking=True).float()

    ensemble_size = cfg["model"]["ensemble_size"]
    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt_data[:, 1:]

    is_deterministic = variant in (
        "E1_l1_only",
        "C1_deterministic",
        "G1_swin_transformer",
        "G2_convlstm",
    )
    n_members = 1 if is_deterministic else max(ensemble_size, 2)

    ensemble_preds = []
    for _ in range(n_members):
        z = None if is_deterministic else True
        out = model(x_input, dt_input, z=z)
        out = out.real if torch.is_complex(out) else out
        ensemble_preds.append(out)

    ensemble_stack = torch.stack(ensemble_preds, dim=0)
    out_mean = ensemble_stack.mean(dim=0)

    l1_loss = ((out_mean - x_target).abs() * lat_weights).mean()
    mse_loss = ((out_mean - x_target) ** 2 * lat_weights).mean()

    if is_deterministic:
        loss = l1_loss
        crps_loss = l1_loss.detach()
        ensemble_std = torch.tensor(0.0, device=device)
    else:
        crps_loss = compute_weighted_crps(ensemble_stack, x_target, lat_weights)
        ensemble_std = ensemble_stack.std(dim=0).mean()
        loss = l1_loss + crps_loss

    basis_free_variants = (
        "B1_complex_latent",
        "G1_swin_transformer",
        "G2_convlstm",
    )
    basis_reg_weight = 0.0 if variant in basis_free_variants else float(cfg["train"].get("basis_reg_weight", 0.01))
    if basis_reg_weight > 0.0:
        basis_reg = compute_basis_residual(model)
        loss = loss + basis_reg_weight * basis_reg
    else:
        basis_reg = torch.tensor(0.0, device=device)

    rollout_weight = float(cfg["train"].get("rollout_loss_weight", 0.0))
    rollout_loss = torch.tensor(0.0, device=device)
    rollout_variants = (
        "baseline",
        "A1_no_dt",
        "B1_complex_latent",
        "B2_fixed_decay",
        "C1_deterministic",
        "C2_no_readout_residual",
        "C3_constant_readout",
        "D1_single_scale",
        "D2_fixed_scale_weights",
        "E1_l1_only",
        "F1_etd1_integrator",
    )
    if rollout_weight > 0.0 and variant in rollout_variants and x_target.shape[1] > 1:
        rollout_steps = min(
            int(cfg["train"].get("rollout_loss_steps", x_target.shape[1])),
            x_target.shape[1],
        )
        dt_rollout = [dt_input[:, i] for i in range(rollout_steps)]
        rollout_pred = model.forward_rollout(
            data[:, :1],
            dt_data[:, :1],
            dt_rollout,
            z_context=None,
            z_rollout=None,
            chunk_size=rollout_steps,
        )
        rollout_pred = rollout_pred.real if torch.is_complex(rollout_pred) else rollout_pred
        rollout_target = x_target[:, :rollout_steps]
        rollout_loss = ((rollout_pred - rollout_target).abs() * lat_weights).mean()
        loss = loss + rollout_weight * rollout_loss

    (loss / grad_accum_steps).backward()

    grad_norm = 0.0
    if (batch_idx + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["train"]["grad_clip"])
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss.item(),
        "l1": l1_loss.item(),
        "crps": crps_loss.item(),
        "rmse": torch.sqrt(mse_loss).item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item(),
        "basis_residual": basis_reg.item(),
        "rollout": rollout_loss.item(),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, variant, path):
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    torch.save(
        {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": int(epoch),
            "global_step": int(step),
            "cfg": cfg,
            "variant": str(variant),
            "seed": int(cfg.get("runtime", {}).get("seed", 42)),
            "ablation_spec": describe_variant(variant),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = get_unwrapped_model(model)
    target.load_state_dict(ckpt["model"], strict=True)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return max(0, int(ckpt.get("epoch", -1)) + 1), int(ckpt.get("global_step", 0))


def load_pretrained(path, model, rank, logger):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = get_unwrapped_model(model)
    target.load_state_dict(ckpt["model"], strict=True)
    if rank == 0:
        logger.info(f"Loaded pretrained ckpt={path}")


def train(cfg, variant, num_workers=4, ckpt_path=None, pretrained_ckpt=None, seed=42):
    rank, world_size, local_rank, device = init_distributed(seed_base=int(seed))
    logger = setup_file_logger(
        "ablation_runner", cfg["logging"]["log_path"], rank, "train.log"
    )
    console, devnull = build_rank_console(rank)

    if rank == 0:
        logger.info(f"variant={variant}")
        logger.info(f"variant_spec={describe_variant(variant)}")
        logger.info(f"seed={int(seed)}")
        logger.info(f"Training on {world_size} GPUs")
        write_json(
            Path(cfg["logging"]["ckpt_dir"]) / "manifest.json",
            build_run_manifest(
                variant=variant,
                seed=int(seed),
                cfg=cfg,
                command=sys.argv,
            ),
        )

    model = build_variant(variant, cfg["model"], device=device)
    model = wrap_ddp(model, local_rank)

    if pretrained_ckpt:
        load_pretrained(pretrained_ckpt, model, rank, logger)

    train_dataset = ERA5Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["year_range"],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["data"]["sample_k"],
        look_ahead=cfg["data"]["look_ahead"],
        is_train=True,
        dt_ref=float(cfg["model"]["dt_ref"]),
        sample_offsets_hours=cfg["data"].get("sample_offsets_hours"),
    )
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        num_workers=max(0, num_workers),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    optimizer = build_variant_optimizer(model, cfg, variant)
    grad_accum_steps = int(cfg["train"]["grad_accum_steps"])
    epochs = int(cfg["train"]["epochs"])

    scheduler = None
    if cfg["train"].get("use_scheduler", False):
        base_lr = float(cfg["train"]["lr"])
        max_lr_factor = float(cfg["train"].get("max_lr_factor", 10.0))
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr * max_lr_factor,
            steps_per_epoch=max(1, len(loader) // max(1, grad_accum_steps)),
            epochs=epochs,
        )

    start_epoch, global_step = 0, 0
    if ckpt_path:
        start_epoch, global_step = load_checkpoint(
            ckpt_path, model, optimizer, scheduler
        )
        if rank == 0:
            logger.info(f"Resumed from ckpt={ckpt_path}")

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
    lat_weights = build_lat_weights(int(cfg["model"]["img_height"]), device)
    log_every = int(cfg["logging"]["log_every"])

    progress = None
    if rank == 0:
        progress = build_progress(console=console)
        progress.start()
        task = progress.add_task(
            f"[{variant}]", total=len(loader) * epochs, completed=global_step
        )

    stop_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            metrics = train_step(
                model,
                batch,
                optimizer,
                cfg,
                grad_accum_steps,
                batch_idx,
                lat_weights,
                variant,
            )
            global_step += 1
            if scheduler and (batch_idx + 1) % grad_accum_steps == 0:
                scheduler.step()

            if rank == 0:
                progress.update(task, advance=1)
                if (batch_idx + 1) % log_every == 0:
                    msg = (
                        f"[{variant} E{epoch+1:03d} B{batch_idx+1:04d}] "
                        f"Loss={metrics['loss']:.4f} L1={metrics['l1']:.4f} "
                        f"CRPS={metrics['crps']:.4f} RMSE={metrics['rmse']:.4f} "
                        f"Basis={metrics['basis_residual']:.4f} "
                        f"Rollout={metrics['rollout']:.4f} "
                        f"Grad={metrics['grad_norm']:.4f}"
                    )
                    progress.console.print(msg)
                    logger.info(msg)

            if should_stop_early(cfg, global_step):
                stop_early = True
                break

        flush_remaining_grads(
            model,
            optimizer,
            float(cfg["train"]["grad_clip"]),
            batch_idx,
            grad_accum_steps,
        )
        last_epoch = epoch

        if rank == 0:
            ckpt_path_ep = os.path.join(
                cfg["logging"]["ckpt_dir"], f"ckpt_epoch{epoch+1}.pt"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                variant,
                ckpt_path_ep,
            )
            logger.info(f"Saved ckpt={ckpt_path_ep}")

        distributed_barrier(local_rank)
        if stop_early:
            break

    if rank == 0:
        final_path = os.path.join(cfg["logging"]["ckpt_dir"], "ckpt_final.pt")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            max(start_epoch - 1, last_epoch),
            global_step,
            cfg,
            variant,
            final_path,
        )
        logger.info(f"Saved final ckpt={final_path}")
        progress.stop()

    train_dataset.cleanup()
    dist.destroy_process_group()
    devnull.close()


def main():
    args = parse_args()
    cfg = build_runtime_cfg(
        args.config,
        data_input_dir=args.data_input_dir,
        train_year_range=args.train_year_range,
        epochs=args.epochs,
        log_path=args.log_path,
        ckpt_dir=args.ckpt_dir,
        ckpt_path=args.ckpt_path,
        max_steps=args.max_steps,
    )
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.ckpt_dir:
        cfg["logging"]["ckpt_dir"] = args.ckpt_dir
    if args.log_path is None and args.ckpt_dir:
        cfg["logging"]["log_path"] = args.ckpt_dir
    cfg.setdefault("runtime", {})
    cfg["runtime"]["seed"] = int(args.seed)

    os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["log_path"], exist_ok=True)

    train(
        cfg,
        variant=args.variant,
        num_workers=args.num_workers,
        ckpt_path=args.ckpt_path,
        pretrained_ckpt=args.pretrained_ckpt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
