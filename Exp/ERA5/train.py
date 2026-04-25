import os

import torch
from torch.optim import lr_scheduler

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import (
    build_adamw_optimizer,
    build_distributed_loader,
    build_lat_weights,
    build_progress,
    build_rank_console,
    build_runtime_arg_parser,
    build_runtime_cfg,
    build_uniphy_model,
    compute_basis_residual,
    compute_crps,
    flush_remaining_grads,
    init_distributed,
    setup_file_logger,
    should_stop_early,
    wrap_ddp,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "train.yaml")


def parse_args():
    return build_runtime_arg_parser(CONFIG_PATH).parse_args()


def setup_logging(log_path, rank):
    return setup_file_logger("train", log_path, rank, "train_metrics.log")


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
    basis_reg = compute_basis_residual(model)
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

    return {
        "loss": loss.item(),
        "l1_loss": l1_loss.item(),
        "crps_loss": crps_loss.item(),
        "mse": mse_loss.item(),
        "rmse": torch.sqrt(mse_loss).item(),
        "grad_norm": grad_norm,
        "ensemble_std": ensemble_std.item(),
        "basis_residual": basis_reg.item(),
    }


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
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    saved_epoch = int(ckpt.get("epoch", -1))
    start_epoch = max(0, saved_epoch + 1)
    return start_epoch, ckpt.get("global_step", 0)


def train(cfg):
    rank, world_size, local_rank, device = init_distributed()
    console, devnull = build_rank_console(rank)

    logger = setup_logging(cfg["logging"]["log_path"], rank)
    if rank == 0:
        logger.info(f"Training started on {world_size} GPUs")

    if cfg["train"]["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_uniphy_model(cfg["model"], device=device)
    model = wrap_ddp(model, local_rank)

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
    train_sampler, train_loader = build_distributed_loader(
        train_dataset,
        cfg["train"]["batch_size"],
        world_size,
        rank,
    )

    optimizer = build_adamw_optimizer(model, cfg)
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
            logger.info(f"Resumed from checkpoint={ckpt_path}")

    log_every = cfg["logging"]["log_every"]
    save_interval = max(1, int(len(train_loader) * cfg["logging"]["ckpt_step"]))
    if rank == 0:
        os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    lat_weights = build_lat_weights(cfg["model"]["img_height"], device)

    progress = None
    task_id = None
    if rank == 0:
        progress = build_progress(console=console)
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
                        f"Basis: {metrics['basis_residual']:.4f} | "
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
                    logger.info(f"Saved checkpoint={save_path}")

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
            logger.info(f"Saved checkpoint={epoch_path}")

        torch.distributed.barrier()
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
        logger.info(f"Saved final checkpoint={final_path}")
        progress.console.print(f"Saved final checkpoint={final_path}")
        progress.stop()

    train_dataset.cleanup()
    torch.distributed.destroy_process_group()
    devnull.close()


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
        max_steps=args.max_steps,
    )
    train(cfg)


if __name__ == "__main__":
    main()
