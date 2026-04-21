import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import CHANNEL_NAMES, get_device, resolve_eval_year_range
from Model.UniPhy.ModelUniPhy import UniPhyModel

VALID_MODEL_ARGS = {
    "in_channels",
    "out_channels",
    "embed_dim",
    "expand",
    "depth",
    "patch_size",
    "img_height",
    "img_width",
    "dt_ref",
    "init_noise_scale",
}
DEFAULT_MODEL_CFG = {
    "in_channels": 30,
    "out_channels": 30,
    "embed_dim": 128,
    "expand": 4,
    "depth": 8,
    "patch_size": [7, 15],
    "img_height": 721,
    "img_width": 1440,
    "dt_ref": 6.0,
    "init_noise_scale": 0.0001,
    "ensemble_size": 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--data-input-dir", default=None)
    parser.add_argument("--eval-year-range", default=None)
    parser.add_argument("--lead-times", default="6,24,72,120,240")
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_int_list(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint or "model" not in checkpoint["cfg"]:
        raise ValueError("Checkpoint missing cfg.model")
    if "model" not in checkpoint:
        raise ValueError("Checkpoint missing model state_dict")
    model_cfg = dict(checkpoint["cfg"]["model"])
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in VALID_MODEL_ARGS}
    filtered_cfg["patch_size"] = tuple(filtered_cfg["patch_size"])
    model = UniPhyModel(**filtered_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint["cfg"], model_cfg


def build_eval_dataset(data_input_dir, eval_year_range, model_cfg, cond_steps, lead_times):
    dt_ref = float(model_cfg.get("dt_ref", 6.0))
    context_offsets = [dt_ref * step for step in range(cond_steps)]
    target_offsets = [context_offsets[-1] + float(lead) for lead in lead_times]
    sample_offsets_hours = context_offsets + target_offsets
    window_size = max(1, int(math.ceil(max(sample_offsets_hours) / dt_ref)) + 1)
    dataset = ERA5Dataset(
        input_dir=data_input_dir,
        year_range=eval_year_range,
        window_size=window_size,
        sample_k=len(sample_offsets_hours),
        look_ahead=0,
        is_train=False,
        dt_ref=dt_ref,
        sample_offsets_hours=sample_offsets_hours,
    )
    lead_deltas = [float(lead_times[0])] + [
        float(curr - prev) for prev, curr in zip(lead_times[:-1], lead_times[1:])
    ]
    return dataset, lead_deltas


def build_lat_weights(height, device):
    lat = torch.linspace(-90, 90, height, device=device)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(1e-6)
    weights = weights / weights.mean()
    return weights.view(1, 1, height, 1)


def weighted_channel_mean(values, lat_weights):
    return (values * lat_weights).mean(dim=(-2, -1))


def compute_channel_crps(pred_ensemble, target, lat_weights):
    ensemble_size = pred_ensemble.shape[0]
    target_exp = target.unsqueeze(0)
    mae = weighted_channel_mean((pred_ensemble - target_exp).abs(), lat_weights).mean(
        dim=0
    )
    if ensemble_size <= 1:
        return mae
    idx_i, idx_j = torch.triu_indices(
        ensemble_size,
        ensemble_size,
        offset=1,
        device=pred_ensemble.device,
    )
    pairwise = weighted_channel_mean(
        (pred_ensemble[idx_i] - pred_ensemble[idx_j]).abs(),
        lat_weights,
    ).mean(dim=0)
    num_pairs = idx_i.shape[0]
    return mae - pairwise * num_pairs / (ensemble_size * ensemble_size)


def compute_channel_acc(pred, target, climatology, lat_weights):
    pred_anom = pred - climatology
    target_anom = target - climatology
    numerator = (pred_anom * target_anom * lat_weights).sum(dim=(-2, -1))
    pred_energy = (pred_anom.square() * lat_weights).sum(dim=(-2, -1))
    target_energy = (target_anom.square() * lat_weights).sum(dim=(-2, -1))
    denom = torch.sqrt(pred_energy * target_energy).clamp_min(1e-8)
    return numerator / denom


def safe_channel_values(values, *, metric_name, fill_value=-1.0):
    finite_mask = torch.isfinite(values)
    if finite_mask.all():
        return values, 0
    invalid_count = int((~finite_mask).sum().item())
    print(
        f"warning: replaced {invalid_count} non-finite per-channel {metric_name} values with {fill_value}"
    )
    safe_values = values.clone()
    safe_values[~finite_mask] = float(fill_value)
    return safe_values, invalid_count


def format_metric_dict(metric_dict):
    return {str(key): round(float(value), 6) for key, value in metric_dict.items()}


def format_invalid_metric_counts(invalid_counts):
    return {key: int(value) for key, value in invalid_counts.items() if int(value) > 0}


def summarize_channel_metrics(channel_names, per_channel_metrics):
    summary = []
    for idx, name in enumerate(channel_names):
        entry = {"channel": name}
        for metric_name, metric_values in per_channel_metrics.items():
            entry[metric_name] = {
                str(lead): round(float(metric_values[lead][idx]), 6)
                for lead in metric_values
            }
        summary.append(entry)
    return summary


def print_summary(result):
    print("=" * 88)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"eval_year_range: {result['eval_year_range']}")
    print(f"num_samples: {result['num_samples']}")
    print(f"ensemble_size: {result['ensemble_size']}")
    print(f"params: {result['params']}")
    print("=" * 88)
    print(f"{'lead_h':>8} | {'rmse':>12} | {'acc':>12} | {'crps':>12}")
    print("-" * 88)
    for lead in result["lead_times_hours"]:
        print(
            f"{lead:>8} | "
            f"{result['overall']['rmse'][str(lead)]:>12.6f} | "
            f"{result['overall']['acc'][str(lead)]:>12.6f} | "
            f"{result['overall']['crps'][str(lead)]:>12.6f}"
        )
    print("=" * 88)
    print(
        f"{'channel':<10} | {'rmse@24h':>12} | {'rmse@120h':>12} | "
        f"{'acc@24h':>12} | {'acc@120h':>12} | {'crps@24h':>12}"
    )
    print("-" * 88)
    for entry in result["per_channel"]:
        rmse_24 = entry["rmse"].get("24", float("nan"))
        rmse_120 = entry["rmse"].get("120", float("nan"))
        acc_24 = entry["acc"].get("24", float("nan"))
        acc_120 = entry["acc"].get("120", float("nan"))
        crps_24 = entry["crps"].get("24", float("nan"))
        print(
            f"{entry['channel']:<10} | {rmse_24:>12.6f} | {rmse_120:>12.6f} | "
            f"{acc_24:>12.6f} | {acc_120:>12.6f} | {crps_24:>12.6f}"
        )
    print("=" * 88)


def main():
    args = parse_args()
    lead_times = parse_int_list(args.lead_times)
    if not lead_times:
        raise ValueError("lead_times must be non-empty")
    if any(curr <= prev for prev, curr in zip(lead_times[:-1], lead_times[1:])):
        raise ValueError("lead_times must be strictly increasing")
    device = torch.device(get_device(args.device))
    model, checkpoint_cfg, model_cfg = build_model_from_checkpoint(
        args.checkpoint,
        device,
    )
    cond_steps = int(checkpoint_cfg.get("alignment", {}).get("condition_steps", 1))
    eval_year_range = resolve_eval_year_range(args.eval_year_range)
    if args.data_input_dir is None:
        raise ValueError("--data-input-dir is required")
    data_input_dir = args.data_input_dir
    dataset, lead_deltas = build_eval_dataset(
        data_input_dir,
        eval_year_range,
        model_cfg,
        cond_steps,
        lead_times,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    ensemble_size = int(
        args.ensemble_size or model_cfg.get("ensemble_size", DEFAULT_MODEL_CFG["ensemble_size"])
    )
    channel_names = CHANNEL_NAMES[: int(model_cfg.get("out_channels", len(CHANNEL_NAMES)))]
    height = int(model_cfg.get("img_height", DEFAULT_MODEL_CFG["img_height"]))
    width = int(model_cfg.get("img_width", DEFAULT_MODEL_CFG["img_width"]))
    if len(channel_names) != int(model_cfg.get("out_channels", len(channel_names))):
        channel_names = [
            f"ch_{idx}"
            for idx in range(int(model_cfg.get("out_channels", len(channel_names))))
        ]
    climatology_sum = torch.zeros(
        len(lead_times),
        len(channel_names),
        height,
        width,
        dtype=torch.float32,
    )
    sample_count = 0
    for batch_idx, (data, _) in enumerate(loader):
        if args.max_samples is not None and sample_count >= args.max_samples:
            break
        remaining = None if args.max_samples is None else args.max_samples - sample_count
        if remaining is not None and data.shape[0] > remaining:
            data = data[:remaining]
        targets = data[:, cond_steps:].float()
        climatology_sum += targets.sum(dim=0).cpu()
        sample_count += data.shape[0]
        if (batch_idx + 1) % max(1, args.log_every) == 0:
            print(f"climatology batches={batch_idx + 1} samples={sample_count}")
    if sample_count == 0:
        raise RuntimeError("No evaluation samples available")
    climatology = climatology_sum / float(sample_count)
    lat_weights = build_lat_weights(height, device)
    mse_sum = torch.zeros(len(lead_times), len(channel_names), dtype=torch.float64)
    acc_sum = torch.zeros(len(lead_times), len(channel_names), dtype=torch.float64)
    crps_sum = torch.zeros(len(lead_times), len(channel_names), dtype=torch.float64)
    processed = 0
    dt_ref = float(model_cfg.get("dt_ref", 6.0))
    started_at = time.time()
    with torch.no_grad():
        for batch_idx, (data, dt_data) in enumerate(loader):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            remaining = None if args.max_samples is None else args.max_samples - processed
            if remaining is not None and data.shape[0] > remaining:
                data = data[:remaining]
                dt_data = dt_data[:remaining]
            data = data.to(device, non_blocking=True).float()
            dt_data = dt_data.to(device, non_blocking=True).float()
            x_context = data[:, :cond_steps]
            x_targets = data[:, cond_steps:]
            dt_context = dt_data[:, :cond_steps]
            batch_size = x_context.shape[0]
            dt_list = [
                torch.full(
                    (batch_size,),
                    delta,
                    device=device,
                    dtype=torch.float32,
                )
                for delta in lead_deltas
            ]
            ensemble_preds = []
            for _ in range(ensemble_size):
                z_context = model.sample_noise(x_context)
                z_rollout = model.sample_rollout_noise(
                    batch_size,
                    len(dt_list),
                    device,
                    dtype=x_context.dtype,
                )
                pred_seq = model.forward_rollout(
                    x_context,
                    dt_context,
                    dt_list,
                    z_context=z_context,
                    z_rollout=z_rollout,
                    chunk_size=args.chunk_size,
                )
                pred_seq = pred_seq.real if pred_seq.is_complex() else pred_seq
                ensemble_preds.append(pred_seq)
            pred_ensemble = torch.stack(ensemble_preds, dim=0)
            pred_mean = pred_ensemble.mean(dim=0)
            target = x_targets[:, : len(lead_times)]
            squared_error = (pred_mean - target).square()
            mse_sum += weighted_channel_mean(squared_error, lat_weights).double().sum(dim=0).cpu()
            crps = compute_channel_crps(pred_ensemble, target, lat_weights)
            crps, _ = safe_channel_values(crps, metric_name="crps")
            crps_sum += crps.double().cpu() * batch_size
            climatology_target = climatology.to(device).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            acc = compute_channel_acc(pred_mean, target, climatology_target, lat_weights)
            acc, _ = safe_channel_values(acc, metric_name="acc")
            acc_sum += acc.double().cpu() * batch_size
            processed += batch_size
            if (batch_idx + 1) % max(1, args.log_every) == 0:
                elapsed = time.time() - started_at
                print(
                    f"eval batches={batch_idx + 1} samples={processed} elapsed_sec={elapsed:.1f}"
                )
    overall_rmse = {}
    overall_acc = {}
    overall_crps = {}
    per_channel_rmse = {}
    per_channel_acc = {}
    per_channel_crps = {}
    for lead_idx, lead in enumerate(lead_times):
        rmse_channels = torch.sqrt(mse_sum[lead_idx] / float(processed))
        acc_channels = acc_sum[lead_idx] / float(processed)
        crps_channels = crps_sum[lead_idx] / float(processed)
        rmse_channels, _ = safe_channel_values(rmse_channels, metric_name="rmse")
        acc_channels, _ = safe_channel_values(acc_channels, metric_name="acc")
        crps_channels, _ = safe_channel_values(crps_channels, metric_name="crps")
        overall_rmse[str(lead)] = float(rmse_channels.mean().item())
        overall_acc[str(lead)] = float(acc_channels.mean().item())
        overall_crps[str(lead)] = float(crps_channels.mean().item())
        per_channel_rmse[lead] = rmse_channels.tolist()
        per_channel_acc[lead] = acc_channels.tolist()
        per_channel_crps[lead] = crps_channels.tolist()
    result = {
        "checkpoint": args.checkpoint,
        "eval_year_range": eval_year_range,
        "num_samples": processed,
        "ensemble_size": ensemble_size,
        "lead_times_hours": lead_times,
        "params": int(sum(param.numel() for param in model.parameters())),
        "overall": {
            "rmse": format_metric_dict(overall_rmse),
            "acc": format_metric_dict(overall_acc),
            "crps": format_metric_dict(overall_crps),
        },
        "per_channel": summarize_channel_metrics(
            channel_names,
            {
                "rmse": per_channel_rmse,
                "acc": per_channel_acc,
                "crps": per_channel_crps,
            },
        ),
        "invalid_metric_counts": format_invalid_metric_counts({}),
    }
    print_summary(result)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()

                torch.full((batch_size,), float(delta), device=device, dtype=torch.float32)
                for delta in lead_deltas
            ]

            ensemble_preds = []
            for _ in range(ensemble_size):
                z_context = model.sample_noise(x_context)
                z_rollout = model.sample_rollout_noise(
                    batch_size,
                    len(lead_times),
                    device,
                    dtype=x_context.dtype,
                )
                pred_seq = model.forward_rollout(
                    x_context,
                    dt_context if cond_steps > 1 else torch.full((batch_size, 1), dt_ref, device=device, dtype=torch.float32),
                    dt_list,
                    z_context=z_context,
                    z_rollout=z_rollout,
                    chunk_size=max(1, args.chunk_size),
                )
                pred_seq = pred_seq.real if pred_seq.is_complex() else pred_seq
                ensemble_preds.append(pred_seq[:, : len(lead_times)])

            pred_ensemble = torch.stack(ensemble_preds, dim=0)
            pred_mean = pred_ensemble.mean(dim=0)

            for lead_idx, lead in enumerate(lead_times):
                target = x_targets[:, lead_idx]
                pred = pred_mean[:, lead_idx]
                climatology_t = climatology[lead_idx].to(device)
                mse_channel = weighted_channel_mean((pred - target).square(), lat_weights)
                acc_channel = compute_channel_acc(pred, target, climatology_t, lat_weights)
                crps_channel = compute_channel_crps(pred_ensemble[:, :, lead_idx], target, lat_weights)
                mse_sum[lead_idx] += mse_channel.sum(dim=0).cpu().to(torch.float64)
                acc_sum[lead_idx] += acc_channel.sum(dim=0).cpu().to(torch.float64)
                crps_sum[lead_idx] += crps_channel.sum(dim=0).cpu().to(torch.float64)

            processed += batch_size
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (batch_idx + 1) % max(1, args.log_every) == 0:
                elapsed = time.time() - started_at
                print(f"evaluation batches={batch_idx + 1} samples={processed} elapsed_sec={elapsed:.1f}")

    mse_mean = mse_sum / float(processed)
    rmse_channel = torch.sqrt(mse_mean)
    acc_channel = acc_sum / float(processed)
    crps_channel = crps_sum / float(processed)

    rmse_channel, rmse_invalid = safe_channel_values(rmse_channel, metric_name="rmse")
    acc_channel, acc_invalid = safe_channel_values(acc_channel, metric_name="acc")
    crps_channel, crps_invalid = safe_channel_values(crps_channel, metric_name="crps")
    invalid_metric_counts = format_invalid_metric_counts(
        {
            "rmse": rmse_invalid,
            "acc": acc_invalid,
            "crps": crps_invalid,
        }
    )

    overall_rmse = {lead: float(rmse_channel[idx].mean().item()) for idx, lead in enumerate(lead_times)}
    overall_acc = {lead: float(acc_channel[idx].mean().item()) for idx, lead in enumerate(lead_times)}
    overall_crps = {lead: float(crps_channel[idx].mean().item()) for idx, lead in enumerate(lead_times)}

    per_channel_metrics = {
        "rmse": {lead: rmse_channel[idx].tolist() for idx, lead in enumerate(lead_times)},
        "acc": {lead: acc_channel[idx].tolist() for idx, lead in enumerate(lead_times)},
        "crps": {lead: crps_channel[idx].tolist() for idx, lead in enumerate(lead_times)},
    }

    result = {
        "checkpoint": args.checkpoint,
        "eval_year_range": eval_year_range,
        "lead_times_hours": lead_times,
        "condition_steps": cond_steps,
        "ensemble_size": ensemble_size,
        "num_samples": processed,
        "params": int(sum(param.numel() for param in model.parameters())),
        "wall_time_sec": round(time.time() - started_at, 3),
        "overall": {
            "rmse": format_metric_dict(overall_rmse),
            "acc": format_metric_dict(overall_acc),
            "crps": format_metric_dict(overall_crps),
        },
        "per_channel": summarize_channel_metrics(channel_names, per_channel_metrics),
    }
    if invalid_metric_counts:
        result["warnings"] = {
            "non_finite_per_channel_metrics_replaced": invalid_metric_counts,
            "replacement_value": -1.0,
        }

    print_summary(result)

    output_json = args.output_json
    if output_json is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_json = os.path.join(os.path.dirname(args.checkpoint), f"{ckpt_stem}_eval.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_json={output_json}")


if __name__ == "__main__":
    main()
