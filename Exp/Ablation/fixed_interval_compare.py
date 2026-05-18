import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from Exp.Ablation.eval import compute_acc_terms, load_variant_model
from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import (
    DEFAULT_MODEL_CFG,
    build_lat_weights,
    compute_channelwise_crps,
    get_device,
    weighted_channel_mean,
)
from Exp.Ablation.protocol import write_json


def parse_args():
    p = argparse.ArgumentParser(description="Fixed-interval operational baseline comparison")
    p.add_argument("--variant", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-input-dir", required=True)
    p.add_argument("--climatology-dir", required=True)
    p.add_argument("--climatology-year-range", default="2000,2001")
    p.add_argument("--eval-year-range", default="2002,2002")
    p.add_argument("--lead-times", default="6,12,18,24")
    p.add_argument("--step-hours", type=int, default=6)
    p.add_argument("--mode", choices=("direct", "recursive"), default="recursive")
    p.add_argument("--ensemble-size", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_year_range(text):
    parts = [p.strip() for p in text.replace(":", ",").split(",") if p.strip()]
    if len(parts) == 1:
        value = int(parts[0])
        return [value, value]
    return [int(parts[0]), int(parts[1])]


def _build_dataset(data_dir, year_range, model_cfg, cond_steps, step_hours, max_lead):
    dt_ref = float(model_cfg.get("dt_ref", 6.0))
    ctx_offsets = [dt_ref * i for i in range(cond_steps)]
    step_count = int(max_lead // step_hours)
    tgt_offsets = [ctx_offsets[-1] + step_hours * (i + 1) for i in range(step_count)]
    offsets = ctx_offsets + tgt_offsets
    dataset = ERA5Dataset(
        input_dir=data_dir,
        year_range=year_range,
        window_size=max(1, int(-(-max(offsets) // dt_ref)) + 1),
        sample_k=len(offsets),
        look_ahead=0,
        is_train=False,
        dt_ref=dt_ref,
        sample_offsets_hours=offsets,
    )
    return dataset, step_count


def _safe(t, fill=-1.0):
    mask = torch.isfinite(t)
    if mask.all():
        return t
    out = t.clone()
    out[~mask] = fill
    return out


def evaluate(args):
    lead_times = _parse_ints(args.lead_times)
    if any(lead % int(args.step_hours) != 0 for lead in lead_times):
        raise ValueError("fixed_interval_leads_must_be_multiples_of_step_hours")
    device = torch.device(get_device(args.device))
    model, ckpt_cfg, model_cfg, ckpt = load_variant_model(
        args.checkpoint,
        args.variant,
        device,
    )
    cond_steps = int(ckpt_cfg.get("alignment", {}).get("condition_steps", 1))
    eval_years = _parse_year_range(args.eval_year_range)
    max_lead = max(lead_times)
    dataset, step_count = _build_dataset(
        args.data_input_dir,
        eval_years,
        model_cfg,
        cond_steps,
        int(args.step_hours),
        max_lead,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    clim_years = _parse_year_range(args.climatology_year_range)
    clim_dataset, _ = _build_dataset(
        args.climatology_dir,
        clim_years,
        model_cfg,
        cond_steps,
        int(args.step_hours),
        max_lead,
    )
    clim_loader = DataLoader(
        clim_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    channels = int(model_cfg.get("out_channels", 30))
    height = int(model_cfg.get("img_height", DEFAULT_MODEL_CFG["img_height"]))
    width = int(model_cfg.get("img_width", DEFAULT_MODEL_CFG["img_width"]))
    clim_sum = torch.zeros(len(lead_times), channels, height, width)
    clim_n = 0
    lead_indices = [lead // int(args.step_hours) - 1 for lead in lead_times]
    for data, _ in clim_loader:
        if args.max_samples and clim_n >= args.max_samples:
            break
        target_steps = data[:, cond_steps:].float()
        clim_sum += target_steps[:, lead_indices].sum(dim=0).cpu()
        clim_n += data.shape[0]
    climatology = (clim_sum / max(clim_n, 1)).to(device)
    lat_w = build_lat_weights(height, device).squeeze(2)
    mse_sum = torch.zeros(len(lead_times), channels, dtype=torch.float64, device=device)
    overall_mse = torch.zeros(len(lead_times), dtype=torch.float64, device=device)
    crps_sum = torch.zeros(len(lead_times), channels, dtype=torch.float64, device=device)
    acc_num = torch.zeros(len(lead_times), channels, dtype=torch.float64, device=device)
    acc_pe = torch.zeros(len(lead_times), channels, dtype=torch.float64, device=device)
    acc_te = torch.zeros(len(lead_times), channels, dtype=torch.float64, device=device)
    processed = 0
    with torch.no_grad():
        for data, dt_data in loader:
            if args.max_samples and processed >= args.max_samples:
                break
            rem = args.max_samples - processed if args.max_samples else None
            if rem and data.shape[0] > rem:
                data, dt_data = data[:rem], dt_data[:rem]
            data = data.to(device, non_blocking=True).float()
            dt_data = dt_data.to(device, non_blocking=True).float()
            batch_size = data.shape[0]
            x_ctx = data[:, :cond_steps]
            dt_ctx = dt_data[:, :cond_steps]
            target = data[:, cond_steps:][:, lead_indices]
            members = []
            deterministic = args.variant in (
                "G1_swin_transformer",
                "G2_convlstm",
                "C1_deterministic",
                "E1_l1_only",
            )
            n_members = 1 if deterministic else max(1, int(args.ensemble_size))
            for _ in range(n_members):
                if args.mode == "direct":
                    dt_list = [
                        torch.full((batch_size,), float(lead), device=device)
                        for lead in lead_times
                    ]
                    pred = model.forward_rollout(
                        x_ctx,
                        dt_ctx,
                        dt_list,
                        z_context=True,
                        z_rollout=True,
                        chunk_size=args.chunk_size,
                    )
                else:
                    dt_list = [
                        torch.full((batch_size,), float(args.step_hours), device=device)
                        for _ in range(step_count)
                    ]
                    pred_all = model.forward_rollout(
                        x_ctx,
                        dt_ctx,
                        dt_list,
                        z_context=True,
                        z_rollout=True,
                        chunk_size=args.chunk_size,
                    )
                    pred = pred_all[:, lead_indices]
                members.append(pred.real if torch.is_complex(pred) else pred)
            ens = torch.stack(members, dim=0)
            mean_pred = ens.mean(dim=0)
            sq = (mean_pred - target).square()
            mse_sum += weighted_channel_mean(sq, lat_w).double().sum(dim=0)
            overall_mse += (sq * lat_w).mean(dim=(-3, -2, -1)).double().sum(dim=0)
            crps_sum += _safe(compute_channelwise_crps(ens, target, lat_w)).double().sum(dim=0)
            clim_exp = climatology.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            n, pe, te = compute_acc_terms(mean_pred, target, clim_exp, lat_w)
            acc_num += n.double().sum(dim=0)
            acc_pe += pe.double().sum(dim=0)
            acc_te += te.double().sum(dim=0)
            processed += batch_size
    overall = {"rmse": {}, "acc": {}, "crps": {}}
    for i, lead in enumerate(lead_times):
        acc = acc_num[i] / torch.sqrt(acc_pe[i] * acc_te[i]).clamp_min(1e-8)
        overall["rmse"][str(lead)] = round(float(torch.sqrt(overall_mse[i] / processed).item()), 6)
        overall["acc"][str(lead)] = round(float(_safe(acc).mean().item()), 6)
        overall["crps"][str(lead)] = round(float(_safe(crps_sum[i] / processed).mean().item()), 6)
    result = {
        "variant": args.variant,
        "seed": ckpt.get("seed"),
        "mode": args.mode,
        "step_hours": int(args.step_hours),
        "lead_times_hours": lead_times,
        "num_samples": processed,
        "overall": overall,
    }
    write_json(args.output_json, result)
    print(json.dumps(result["overall"], indent=2))


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
