import os

if __package__ in {None, ""}:
    import sys

    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        ),
    )

import numpy as np
import torch
from torch.utils.data import DataLoader

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.exp.exp_utils import (
    find_latest_checkpoint,
    get_data_input_dir,
    get_device,
    load_config_and_model,
)


DT_REF = 6.0
TOTAL_HOURS = 12.0
DT_LIST = [1, 2, 3, 6]


def get_synthetic_pair(device, model_cfg):
    channels = int(model_cfg.get("in_channels", 30))
    height = int(model_cfg.get("img_height", 721))
    width = int(model_cfg.get("img_width", 1440))
    x_ctx = torch.randn(1, 1, channels, height, width, device=device)
    x_target = torch.randn(1, channels, height, width, device=device)
    print(
        f"Using synthetic tensors with shape: "
        f"{tuple(x_ctx.shape)} -> {tuple(x_target.shape)}"
    )
    return x_ctx, x_target


def get_test_data(device, model_cfg, force_synthetic=False):
    if not force_synthetic:
        try:
            dataset = ERA5Dataset(
                input_dir=get_data_input_dir(),
                year_range=[2009, 2009],
                window_size=4,
                sample_k=3,
                is_train=False,
                dt_ref=DT_REF,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            data, _ = next(iter(loader))
            x_ctx = data[:, 0:1].to(device).float()
            x_target = data[:, 2].to(device).float()
            expected_shape = (
                int(model_cfg.get("in_channels", x_ctx.shape[2])),
                int(model_cfg.get("img_height", x_ctx.shape[-2])),
                int(model_cfg.get("img_width", x_ctx.shape[-1])),
            )
            actual_shape = (x_ctx.shape[2], x_ctx.shape[-2], x_ctx.shape[-1])
            if actual_shape == expected_shape:
                print("Real ERA5 data loaded successfully.")
                return x_ctx, x_target
            print(
                f"Loaded ERA5 shape {actual_shape} does not match checkpoint shape "
                f"{expected_shape}. Falling back to synthetic tensors."
            )
        except Exception as e:
            print(f"Data loading failed ({e}). Falling back to synthetic tensors.")
    return get_synthetic_pair(device, model_cfg)


@torch.no_grad()
def rollout_last(model, x_ctx, dt_step, total_hours, dt_context):
    device = x_ctx.device
    n_iters = int(round(total_hours / dt_step))
    dt_list = [
        torch.tensor(dt_step, device=device, dtype=torch.float32)
        for _ in range(n_iters)
    ]
    dt_ctx = torch.full(
        (x_ctx.shape[0], x_ctx.shape[1]),
        dt_context,
        device=device,
        dtype=torch.float32,
    )
    pred_seq = model.forward_rollout(x_ctx, dt_ctx, dt_list)
    out = pred_seq[:, -1]
    if out.is_complex():
        out = out.real
    return out


def summarize_consistency(results):
    print("\nConsistency spread across dt choices:")
    print("-" * 65)
    for name, res in results.items():
        vals = [v for v in res.values() if np.isfinite(v)]
        if not vals:
            print(f"{name:<15} | no valid results")
            continue
        spread = max(vals) - min(vals)
        mean_val = float(np.mean(vals))
        print(
            f"{name:<15} | mean RMSE={mean_val:.4f} | "
            f"spread={spread:.4f}"
        )


def main():
    device = get_device()
    print("=" * 65)
    print("Experiment 4: Temporal Consistency Over a Fixed 12h Horizon")
    print("=" * 65)

    force_synthetic = os.environ.get("UNIPHY_FORCE_SYNTHETIC") == "1"

    ckpt_pre = find_latest_checkpoint(
        "./uniphy/ckpt/*.pt",
        "./**/ckpt/*.pt",
    )
    ckpt_align = find_latest_checkpoint(
        "./uniphy/align_ckpt/*.pt",
        "./**/align_ckpt/*.pt",
    )

    checkpoints = {
        "Pre-trained": ckpt_pre,
        "Fine-tuned": ckpt_align,
    }

    if all(path is None for path in checkpoints.values()):
        print("Checkpoints not found.")
        return

    results = {}
    data_ready = False
    x_ctx = None
    x_target = None

    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            results[name] = {dt: float("nan") for dt in DT_LIST}
            continue

        print(f"Evaluating {name} Model...")
        print(f"   Path: {ckpt_path}")

        model, cfg = load_config_and_model(
            ckpt_path,
            device,
            allow_missing=True,
        )
        if model is None:
            results[name] = {dt: float("nan") for dt in DT_LIST}
            continue

        dt_context = float(cfg.get("dt_ref", DT_REF))
        if not data_ready:
            x_ctx, x_target = get_test_data(
                device,
                cfg,
                force_synthetic=force_synthetic,
            )
            data_ready = True

        model_res = {}
        for dt in DT_LIST:
            try:
                x_pred = rollout_last(
                    model,
                    x_ctx,
                    dt_step=float(dt),
                    total_hours=TOTAL_HOURS,
                    dt_context=dt_context,
                )
                mse = torch.mean((x_pred - x_target) ** 2)
                rmse = torch.sqrt(mse).item()
                model_res[dt] = rmse
                n_steps = int(TOTAL_HOURS // dt)
                print(f"   -> dt={dt}h ({n_steps} steps): RMSE = {rmse:.4f}")
            except Exception as e:
                print(f"   -> dt={dt}h: Failed ({e})")
                model_res[dt] = float("nan")

        results[name] = model_res
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    header = f"{'Model Type':<15}"
    for dt in DT_LIST:
        header += f" | {'dt=' + str(dt) + 'h':<10}"
    print(header)
    print("-" * 65)

    for name in ["Pre-trained", "Fine-tuned"]:
        row = f"{name:<15}"
        res = results.get(name, {})
        for dt in DT_LIST:
            val = res.get(dt, float("nan"))
            cell = "N/A" if np.isnan(val) else f"{val:.4f}"
            row += f" | {cell:<10}"
        print(row)

    print("=" * 65)
    summarize_consistency(results)


if __name__ == "__main__":
    main()
