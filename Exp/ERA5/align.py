import math
import os
import random

import torch

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import (
    build_adamw_optimizer,
    build_distributed_loader,
    build_lat_weights,
    build_progress,
    build_runtime_arg_parser,
    build_runtime_cfg,
    build_uniphy_model,
    compute_basis_residual,
    compute_weighted_crps,
    flush_remaining_grads,
    get_unwrapped_model,
    init_distributed,
    setup_file_logger,
    should_stop_early,
    wrap_ddp,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "align.yaml")


def parse_args():
    return build_runtime_arg_parser(
        CONFIG_PATH,
        include_pretrained_ckpt=True,
    ).parse_args()


def setup_logging(log_path, rank):
    return setup_file_logger("align", log_path, rank, "align.log")


def load_alignment_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = get_unwrapped_model(model)
    target.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    saved_epoch = int(ckpt.get("epoch", -1))
    start_epoch = max(0, saved_epoch + 1)
    return start_epoch, int(ckpt.get("global_step", 0))


def get_curriculum_sub_steps(epoch, total_epochs, all_sub_steps):
    sorted_steps = sorted(all_sub_steps)
    n = len(sorted_steps)
    progress = epoch / max(total_epochs - 1, 1)
    unlock_count = max(1, math.ceil(progress * n))
    if epoch < total_epochs // 3:
        unlock_count = max(1, n // 2)
    return sorted_steps[:unlock_count]


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
    available_sub_steps = get_curriculum_sub_steps(
        epoch,
        total_epochs,
        all_sub_steps,
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

    infer_model = get_unwrapped_model(model)
    output_offset = sub_step - 1
    x_tgt_aligned = x_targets[:, :target_t]
    lat_weights = build_lat_weights(x_tgt_aligned.shape[-2], device)
    lat_weights_cpu = lat_weights.cpu()
    target_cpu = x_tgt_aligned.detach().cpu()

    ensemble_preds = []
    for _ in range(ensemble_size):
        z_context = infer_model.sample_noise(x_ctx)
        z_rollout = infer_model.sample_rollout_noise(
            batch_size,
            n_iters,
            device,
            dtype=x_ctx.dtype,
        )
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
        ensemble_preds.append(pred_seq[:, :target_t])

    ensemble_stack = torch.stack(ensemble_preds, dim=0)
    pred_mean = ensemble_stack.mean(dim=0)

    pred_mean_cpu = pred_mean.detach().cpu()
    l1 = ((pred_mean_cpu - target_cpu).abs() * lat_weights_cpu).mean()
    mse = (((pred_mean_cpu - target_cpu) ** 2) * lat_weights_cpu).mean()

    crps_loss = compute_weighted_crps(ensemble_stack, x_tgt_aligned, lat_weights)
    basis_reg_weight = cfg["train"].get("basis_reg_weight", 0.0)
    basis_residual = compute_basis_residual(model)
    loss = crps_loss + basis_reg_weight * basis_residual
    (loss / grad_accum_steps).backward()

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
        "crps": crps_loss.item(),
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


def align(cfg):
    rank, world_size, local_rank, device = init_distributed()
    logger = setup_logging(cfg["logging"]["log_path"], rank)
    cond_steps = cfg["alignment"]["condition_steps"]
    max_tgt_steps = cfg["alignment"]["max_target_steps"]
    sample_k = cond_steps + max_tgt_steps
    all_sub_steps = cfg["alignment"]["sub_steps"]
    total_epochs = cfg["train"]["epochs"]

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UniPhy Alignment Training")
        logger.info(f"Condition Steps={cond_steps}")
        logger.info(f"Max Target Steps={max_tgt_steps}")
        logger.info(f"Sample K={sample_k}")
        logger.info(f"Sub Steps={all_sub_steps}")
        logger.info(f"Max Rollout Steps={cfg['alignment']['max_rollout_steps']}")
        logger.info(f"Chunk Size={cfg['alignment']['chunk_size']}")
        logger.info(
            f"Basis Reg Weight={cfg['train'].get('basis_reg_weight', 0.0):.4f}"
        )
        logger.info("=" * 60)

    pretrained_path = cfg["alignment"].get("pretrained_ckpt", "")
    pretrained_state = None
    if pretrained_path:
        pretrained_state = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )

    model = build_uniphy_model(cfg["model"], device=device)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state["model"], strict=True)
        if rank == 0:
            logger.info(f"Initialized from checkpoint={pretrained_path}")
    model = wrap_ddp(model, local_rank)

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
    train_sampler, train_loader = build_distributed_loader(
        train_dataset,
        cfg["train"]["batch_size"],
        world_size,
        rank,
    )

    optimizer = build_adamw_optimizer(model, cfg)
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
            logger.info(f"Resumed alignment checkpoint={resume_path}")

    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    stop_early = False
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        unlocked = get_curriculum_sub_steps(epoch, total_epochs, all_sub_steps)
        if rank == 0:
            min_dt_min = cfg["alignment"]["target_dt"] / max(unlocked) * 60
            logger.info(
                f"Epoch {epoch + 1}: unlocked sub_steps={unlocked}, "
                f"finest dt={min_dt_min:.1f}min"
            )

        progress = None
        task_id = None
        if rank == 0:
            progress = build_progress()
            progress.start()
            task_id = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}",
                total=len(train_loader),
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
                        f"RMSE: {metrics['rmse']:.4f} "
                        f"Basis: {metrics['basis_residual']:.4f}"
                    )
                    progress.console.print(log_msg)
                    logger.info(log_msg)

            if should_stop_early(cfg, global_step):
                stop_early = True
                break

        flush_remaining_grads(
            model,
            optimizer,
            cfg["train"]["grad_clip"],
            batch_idx,
            grad_accum_steps,
        )

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
            logger.info(f"Saved checkpoint={ckpt_save_path}")

        torch.distributed.barrier()
        if stop_early:
            break

    if rank == 0:
        final_ckpt_path = os.path.join(
            cfg["logging"]["ckpt_dir"],
            "align_final.pt",
        )
        save_checkpoint(
            model,
            optimizer,
            max(start_epoch, epochs - 1),
            global_step,
            cfg,
            final_ckpt_path,
        )
        logger.info(f"Saved final checkpoint={final_ckpt_path}")

    train_dataset.cleanup()
    torch.distributed.destroy_process_group()


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

