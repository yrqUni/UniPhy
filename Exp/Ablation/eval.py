import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.runtime_config import (
    CHANNEL_NAMES,
    DEFAULT_MODEL_CFG,
    build_lat_weights,
    compute_channelwise_crps,
    get_device,
    get_unwrapped_model,
    resolve_eval_year_range,
    weighted_channel_mean,
)
from Exp.Ablation.protocol import get_variant_spec, write_json
from Exp.Ablation.variants import build_variant


def parse_args():
    p = argparse.ArgumentParser(description="UniPhy ablation evaluation")
    p.add_argument("--variant", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-input-dir", required=True)
    p.add_argument("--climatology-dir", default=None)
    p.add_argument(
        "--climatology-year-range",
        default=None,
        help="Year range for climatology data, e.g. '2000,2016'. "
        "Defaults to eval-year-range when not specified.",
    )
    p.add_argument("--eval-year-range", default=None)
    p.add_argument("--lead-times", default="6,24,72,120,240")
    p.add_argument("--ensemble-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=1)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_year_range(s):
    parts = [p.strip() for p in s.replace(":", ",").split(",") if p.strip()]
    if len(parts) == 1:
        yr = int(parts[0])
        return [yr, yr]
    return [int(parts[0]), int(parts[1])]


def load_variant_model(ckpt_path, variant, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = dict(ckpt.get("cfg", {}).get("model", {})) or dict(DEFAULT_MODEL_CFG)
    model = build_variant(variant, model_cfg, device=device)
    target = get_unwrapped_model(model)
    target.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt.get("cfg", {}), model_cfg, ckpt


def build_eval_dataset(data_dir, year_range, model_cfg, cond_steps, lead_times):
    dt_ref = float(model_cfg.get("dt_ref", 6.0))
    ctx_offsets = [dt_ref * i for i in range(cond_steps)]
    tgt_offsets = [ctx_offsets[-1] + float(lt) for lt in lead_times]
    all_offsets = ctx_offsets + tgt_offsets
    window_size = max(1, int(-(-max(all_offsets) // dt_ref)) + 1)
    dataset = ERA5Dataset(
        input_dir=data_dir,
        year_range=year_range,
        window_size=window_size,
        sample_k=len(all_offsets),
        look_ahead=0,
        is_train=False,
        dt_ref=dt_ref,
        sample_offsets_hours=all_offsets,
    )
    lead_deltas = [float(lead_times[0])] + [
        float(b - a) for a, b in zip(lead_times[:-1], lead_times[1:])
    ]
    return dataset, lead_deltas


def compute_acc_terms(pred, target, clim, lat_w):
    pred_a = pred - clim
    tgt_a = target - clim
    num = (pred_a * tgt_a * lat_w).sum(dim=(-2, -1))
    pred_e = (pred_a.square() * lat_w).sum(dim=(-2, -1))
    tgt_e = (tgt_a.square() * lat_w).sum(dim=(-2, -1))
    return num, pred_e, tgt_e


def safe_values(t, fill=-1.0):
    mask = torch.isfinite(t)
    if mask.all():
        return t
    out = t.clone()
    out[~mask] = fill
    return out


def evaluate(args):
    lead_times = _parse_int_list(args.lead_times)
    device = torch.device(get_device(args.device))

    model, ckpt_cfg, model_cfg, ckpt = load_variant_model(
        args.checkpoint, args.variant, device
    )
    cond_steps = int(ckpt_cfg.get("alignment", {}).get("condition_steps", 1))
    eval_yr = resolve_eval_year_range(args.eval_year_range)

    dataset, lead_deltas = build_eval_dataset(
        args.data_input_dir, eval_yr, model_cfg, cond_steps, lead_times
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    n_ch = int(model_cfg.get("out_channels", len(CHANNEL_NAMES)))
    H = int(model_cfg.get("img_height", DEFAULT_MODEL_CFG["img_height"]))
    W = int(model_cfg.get("img_width", DEFAULT_MODEL_CFG["img_width"]))
    ch_names = (
        CHANNEL_NAMES[:n_ch]
        if n_ch <= len(CHANNEL_NAMES)
        else [f"ch_{i}" for i in range(n_ch)]
    )

    ensemble_size = args.ensemble_size or int(
        model_cfg.get("ensemble_size", DEFAULT_MODEL_CFG["ensemble_size"])
    )
    if args.variant in ("C1_deterministic", "E1_l1_only"):
        ensemble_size = 1

    clim_dir = args.climatology_dir or args.data_input_dir
    clim_src = "provided" if args.climatology_dir else "eval_set (approx)"
    if not args.climatology_dir:
        print("WARNING: --climatology-dir not supplied; using eval-set mean for ACC.")

    clim_yr_str = args.climatology_year_range or args.eval_year_range
    clim_yr = _parse_year_range(clim_yr_str) if clim_yr_str else eval_yr
    clim_dataset, _ = build_eval_dataset(
        clim_dir, clim_yr, model_cfg, cond_steps, lead_times
    )
    clim_loader = DataLoader(
        clim_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    clim_sum = torch.zeros(len(lead_times), n_ch, H, W, dtype=torch.float32)
    clim_n = 0
    for data, _ in clim_loader:
        if args.max_samples and clim_n >= args.max_samples:
            break
        tgt = data[:, cond_steps:].float()
        clim_sum += tgt.sum(dim=0).cpu()
        clim_n += data.shape[0]
    climatology = (clim_sum / max(clim_n, 1)).to(device)

    lat_w = build_lat_weights(H, device).squeeze(2)

    mse_sum = torch.zeros(len(lead_times), n_ch, dtype=torch.float64, device=device)
    overall_mse_sum = torch.zeros(len(lead_times), dtype=torch.float64, device=device)
    crps_sum = torch.zeros(len(lead_times), n_ch, dtype=torch.float64, device=device)
    acc_num = torch.zeros(len(lead_times), n_ch, dtype=torch.float64, device=device)
    acc_pe = torch.zeros(len(lead_times), n_ch, dtype=torch.float64, device=device)
    acc_te = torch.zeros(len(lead_times), n_ch, dtype=torch.float64, device=device)

    processed = 0
    t0 = time.time()

    is_det = args.variant in ("C1_deterministic", "E1_l1_only")

    with torch.no_grad():
        for batch_idx, (data, dt_data) in enumerate(loader):
            if args.max_samples and processed >= args.max_samples:
                break
            rem = (args.max_samples - processed) if args.max_samples else None
            if rem and data.shape[0] > rem:
                data, dt_data = data[:rem], dt_data[:rem]

            data = data.to(device, non_blocking=True).float()
            dt_data = dt_data.to(device, non_blocking=True).float()
            x_ctx = data[:, :cond_steps]
            x_tgt = data[:, cond_steps:]
            dt_ctx = dt_data[:, :cond_steps]
            B = x_ctx.shape[0]

            dt_list = [
                torch.full((B,), d, device=device, dtype=torch.float32)
                for d in lead_deltas
            ]

            members = []
            for _ in range(max(ensemble_size, 1)):
                z_ctx = None if is_det else True
                pred = model.forward_rollout(
                    x_ctx,
                    dt_ctx,
                    dt_list,
                    z_context=z_ctx,
                    chunk_size=args.chunk_size,
                )
                pred = pred.real if torch.is_complex(pred) else pred
                members.append(pred[:, : len(lead_times)])

            ens = torch.stack(members, dim=0)
            mean_pred = ens.mean(dim=0)
            target = x_tgt[:, : len(lead_times)]

            sq_err = (mean_pred - target).square()
            mse_sum += weighted_channel_mean(sq_err, lat_w).double().sum(dim=0)
            overall_mse_sum += (
                (sq_err * lat_w).mean(dim=(-3, -2, -1)).double().sum(dim=0)
            )

            crps = compute_channelwise_crps(ens, target, lat_w)
            crps_sum += safe_values(crps).double().sum(dim=0)

            clim_exp = climatology.unsqueeze(0).expand(B, -1, -1, -1, -1)
            n, pe, te = compute_acc_terms(mean_pred, target, clim_exp, lat_w)
            acc_num += n.double().sum(dim=0)
            acc_pe += pe.double().sum(dim=0)
            acc_te += te.double().sum(dim=0)

            processed += B
            if (batch_idx + 1) % max(1, args.log_every) == 0:
                elapsed = time.time() - t0
                print(
                    f"eval batch={batch_idx+1} samples={processed} elapsed={elapsed:.1f}s"
                )

    def fmt(d):
        return {str(k): round(float(v), 6) for k, v in d.items()}

    overall_rmse, overall_acc, overall_crps = {}, {}, {}
    per_ch_rmse, per_ch_acc, per_ch_crps = {}, {}, {}

    for li, lt in enumerate(lead_times):
        rmse_ch = torch.sqrt(mse_sum[li] / float(processed))
        acc_denom = torch.sqrt(acc_pe[li] * acc_te[li]).clamp_min(1e-8)
        acc_ch = acc_num[li] / acc_denom
        crps_ch = crps_sum[li] / float(processed)

        overall_rmse[lt] = float(
            torch.sqrt(overall_mse_sum[li] / float(processed)).item()
        )
        overall_acc[lt] = float(safe_values(acc_ch).mean().item())
        overall_crps[lt] = float(safe_values(crps_ch).mean().item())

        per_ch_rmse[lt] = safe_values(rmse_ch).cpu().tolist()
        per_ch_acc[lt] = safe_values(acc_ch).cpu().tolist()
        per_ch_crps[lt] = safe_values(crps_ch).cpu().tolist()

    per_channel = [
        {
            "channel": ch_names[i],
            "rmse": {str(lt): round(per_ch_rmse[lt][i], 6) for lt in lead_times},
            "acc": {str(lt): round(per_ch_acc[lt][i], 6) for lt in lead_times},
            "crps": {str(lt): round(per_ch_crps[lt][i], 6) for lt in lead_times},
        }
        for i in range(n_ch)
    ]

    result = {
        "variant": args.variant,
        "variant_spec": get_variant_spec(args.variant).to_dict(),
        "seed": args.seed if args.seed is not None else ckpt.get("seed"),
        "checkpoint": args.checkpoint,
        "eval_year_range": eval_yr,
        "climatology_source": clim_src,
        "num_samples": processed,
        "ensemble_size": ensemble_size,
        "lead_times_hours": lead_times,
        "params": int(sum(p.numel() for p in model.parameters())),
        "overall": {
            "rmse": fmt(overall_rmse),
            "acc": fmt(overall_acc),
            "crps": fmt(overall_crps),
        },
        "per_channel": per_channel,
    }

    print("=" * 72)
    print(f"variant={args.variant}  samples={processed}  ensemble={ensemble_size}")
    print(f"{'lead_h':>8} | {'rmse':>12} | {'acc':>12} | {'crps':>12}")
    print("-" * 72)
    for lt in lead_times:
        print(
            f"{lt:>8} | {result['overall']['rmse'][str(lt)]:>12.6f} | "
            f"{result['overall']['acc'][str(lt)]:>12.6f} | "
            f"{result['overall']['crps'][str(lt)]:>12.6f}"
        )
    print("=" * 72)

    out_path = args.output_json
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(
            os.path.dirname(args.checkpoint), f"{stem}_{args.variant}_eval.json"
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_json(out_path, result)
    print(f"saved → {out_path}")
    return result


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
