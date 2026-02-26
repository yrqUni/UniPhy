import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5Dataset
from exp_utils import get_device, load_config_and_model


DT_REF = 6.0
TOTAL_HOURS = 12.0
DT_LIST = [1, 2, 3, 6]


def get_test_data(device):
    try:
        dataset = ERA5Dataset(
            input_dir="/nfs/ERA5_data/data_norm",
            year_range=[2009, 2009],
            window_size=4,
            sample_k=3,
            sampling_mode="sequential",
            is_train=False,
            dt_ref=DT_REF,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        data, _ = next(iter(loader))
        x_ctx = data[:, 0:1].to(device).float()
        x_target = data[:, 2].to(device).float()
        print("Real ERA5 data loaded successfully.")
        return x_ctx, x_target
    except Exception as e:
        print(f"Data loading failed ({e}). Using random tensors.")
        x_ctx = torch.randn(1, 1, 30, 721, 1440, device=device)
        x_target = torch.randn(1, 30, 721, 1440, device=device)
        return x_ctx, x_target


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
        dt_context, device=device, dtype=torch.float32,
    )
    pred_seq = model.forward_rollout(x_ctx, dt_ctx, dt_list)
    out = pred_seq[:, -1]
    if out.is_complex():
        out = out.real
    return out


def main():
    device = get_device()
    print("=" * 65)
    print("Experiment 4: Temporal Consistency (forward_rollout)")
    print("=" * 65)

    x_ctx, x_target = get_test_data(device)

    ckpt_pre_list = sorted(
        glob.glob("./uniphy/ckpt/*.pt"), key=os.path.getmtime,
    )
    ckpt_align_list = sorted(
        glob.glob("./uniphy/align_ckpt/*.pt"), key=os.path.getmtime,
    )

    checkpoints = {
        "Pre-trained": ckpt_pre_list[-1] if ckpt_pre_list else None,
        "Fine-tuned": ckpt_align_list[-1] if ckpt_align_list else None,
    }

    results = {}

    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            results[name] = {dt: float("nan") for dt in DT_LIST}
            continue

        print(f"Evaluating {name} Model...")
        print(f"   Path: {ckpt_path}")

        model, cfg = load_config_and_model(
            ckpt_path, device, allow_missing=True,
        )
        if model is None:
            results[name] = {dt: float("nan") for dt in DT_LIST}
            continue

        dt_context = float(cfg.get("dt_ref", DT_REF))

        model_res = {}
        for dt in DT_LIST:
            try:
                x_pred = rollout_last(
                    model, x_ctx, dt_step=float(dt),
                    total_hours=TOTAL_HOURS, dt_context=dt_context,
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


if __name__ == "__main__":
    main()
