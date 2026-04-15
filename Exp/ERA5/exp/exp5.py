import os

if __package__ in {None, ""}:
    import sys

    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        ),
    )

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch.utils.data import DataLoader

from Exp.ERA5.ERA5 import ERA5Dataset
from Exp.ERA5.exp.exp_utils import (
    find_latest_checkpoint,
    get_data_input_dir,
    get_device,
    load_config_and_model,
)


DT_REF = 6.0

rcParams["font.family"] = "serif"
rcParams["font.serif"] = [
    "DejaVu Serif",
    "Bitstream Vera Serif",
    "Computer Modern Roman",
]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 9
rcParams["axes.titlesize"] = 9
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


def get_synthetic_triplet(device, model_cfg):
    channels = int(model_cfg.get("in_channels", 30))
    height = int(model_cfg.get("img_height", 721))
    width = int(model_cfg.get("img_width", 1440))
    data = torch.randn(1, 3, channels, height, width, device=device)
    print(f"Using synthetic visualization tensors with shape: {tuple(data.shape)}")
    return data


def get_vis_data(device, model_cfg, force_synthetic=False):
    if not force_synthetic:
        try:
            dataset = ERA5Dataset(
                input_dir=get_data_input_dir(),
                year_range=[2009, 2009],
                window_size=3,
                sample_k=3,
                is_train=False,
                dt_ref=DT_REF,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            data, _ = next(iter(loader))
            data = data.to(device).float()
            expected_shape = (
                int(model_cfg.get("in_channels", data.shape[2])),
                int(model_cfg.get("img_height", data.shape[-2])),
                int(model_cfg.get("img_width", data.shape[-1])),
            )
            actual_shape = (data.shape[2], data.shape[-2], data.shape[-1])
            if actual_shape == expected_shape:
                print("Real ERA5 data loaded successfully.")
                return data
            print(
                f"Loaded ERA5 shape {actual_shape} does not match checkpoint shape "
                f"{expected_shape}. Falling back to synthetic tensors."
            )
        except Exception as e:
            print(f"Data loading failed ({e}). Falling back to synthetic tensors.")
    return get_synthetic_triplet(device, model_cfg)


@torch.no_grad()
def run_inference_sequence(model, x_ctx, dt_step, total_steps, dt_context):
    device = x_ctx.device
    dt_list = [
        torch.tensor(dt_step, device=device, dtype=torch.float32)
        for _ in range(int(total_steps))
    ]
    dt_ctx = torch.full(
        (x_ctx.shape[0], x_ctx.shape[1]),
        dt_context,
        device=device,
        dtype=torch.float32,
    )
    pred_seq = model.forward_rollout(x_ctx, dt_ctx, dt_list)
    if pred_seq.is_complex():
        pred_seq = pred_seq.real
    return pred_seq


def compute_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def find_best_channel(real_data, pre_seq, fine_seq, num_channels=30):
    step_indices = [5, 11]
    gt_indices = [1, 2]

    best_ch = 0
    best_improvement = -float("inf")

    print("\nChannel-wise RMSE comparison:")
    print("-" * 70)
    print(
        f"{'Ch':>4} | {'Pre-RMSE':>10} | {'Fine-RMSE':>10}"
        f" | {'Improv%':>10} | {'Status'}"
    )
    print("-" * 70)

    for ch in range(num_channels):
        pre_rmse_total = 0.0
        fine_rmse_total = 0.0

        for step_idx, gt_idx in zip(step_indices, gt_indices):
            gt_img = real_data[0, gt_idx, ch].cpu().numpy()
            pre_img = pre_seq[0, step_idx, ch].cpu().numpy()
            fine_img = fine_seq[0, step_idx, ch].cpu().numpy()
            pre_rmse_total += compute_rmse(pre_img, gt_img)
            fine_rmse_total += compute_rmse(fine_img, gt_img)

        pre_rmse_avg = pre_rmse_total / len(step_indices)
        fine_rmse_avg = fine_rmse_total / len(step_indices)

        improvement = 0.0
        if pre_rmse_avg > 1e-6:
            improvement = (pre_rmse_avg - fine_rmse_avg) / pre_rmse_avg * 100.0

        status = "BETTER" if improvement > 0 else "WORSE"
        print(
            f"{ch:>4} | {pre_rmse_avg:>10.4f} | {fine_rmse_avg:>10.4f}"
            f" | {improvement:>+10.2f}% | {status}"
        )

        if improvement > best_improvement:
            best_improvement = improvement
            best_ch = ch

    print("-" * 70)
    print(f"Best channel: {best_ch} (improvement: {best_improvement:+.2f}%)")
    return best_ch


def visualize_comparison(real_data, pre_seq, fine_seq, channel_idx, save_path):
    step_indices = [5, 11]
    time_labels = ["6h", "12h"]
    gt_indices = [1, 2]
    row_labels = ["Ground Truth", "Pre-trained", "Fine-tuned"]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(4.2, 4.0),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )

    gt_imgs = [
        real_data[0, idx, channel_idx].cpu().numpy() for idx in gt_indices
    ]
    pre_imgs = [
        pre_seq[0, idx, channel_idx].cpu().numpy() for idx in step_indices
    ]
    fine_imgs = [
        fine_seq[0, idx, channel_idx].cpu().numpy() for idx in step_indices
    ]

    all_imgs = gt_imgs + pre_imgs + fine_imgs
    vmin = min(img.min() for img in all_imgs)
    vmax = max(img.max() for img in all_imgs)

    cmap = "RdBu_r"
    im = None

    for col_idx in range(2):
        ax = axes[0, col_idx]
        im = ax.imshow(
            gt_imgs[col_idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(rf"$t$ = {time_labels[col_idx]}", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    for row_idx, (imgs, row_offset) in enumerate(
        [(pre_imgs, 1), (fine_imgs, 2)]
    ):
        for col_idx in range(2):
            ax = axes[row_offset, col_idx]
            pred_img = imgs[col_idx]
            gt_img = gt_imgs[col_idx]
            ax.imshow(
                pred_img,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            rmse = compute_rmse(pred_img, gt_img)
            mae = compute_mae(pred_img, gt_img)
            ax.text(
                0.97,
                0.03,
                f"RMSE={rmse:.3f}\nMAE={mae:.3f}",
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    linewidth=0.3,
                ),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7, width=0.3, length=2)
    cbar.outline.set_linewidth(0.3)
    cbar.set_label("Normalized", fontsize=8, labelpad=3)

    fig.suptitle(
        r"Fixed-Horizon Temporal Comparison ($\Delta t$=1h)"
        f" - Channel {channel_idx}",
        fontsize=10,
        y=0.98,
    )

    plt.savefig(
        save_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    print(f"Figure saved to {save_path}")
    plt.close(fig)


def main():
    device = get_device()
    print("=" * 60)
    print("Experiment 5: Visualizing Fixed-Horizon Temporal Consistency (dt=1h)")
    print("=" * 60)

    force_synthetic = os.environ.get("UNIPHY_FORCE_SYNTHETIC") == "1"

    ckpt_pre = find_latest_checkpoint(
        "./uniphy/ckpt/*.pt",
        "./**/ckpt/*.pt",
    )
    ckpt_align = find_latest_checkpoint(
        "./uniphy/align_ckpt/*.pt",
        "./**/align_ckpt/*.pt",
    )

    if not ckpt_pre or not ckpt_align:
        print("Checkpoints not found.")
        return

    print("Inferencing Pre-trained...")
    model_pre, cfg_pre = load_config_and_model(ckpt_pre, device)
    data = get_vis_data(device, cfg_pre, force_synthetic=force_synthetic)
    x_ctx = data[:, 0:1]
    dt_context_pre = float(cfg_pre.get("dt_ref", DT_REF))
    seq_pre = run_inference_sequence(
        model_pre,
        x_ctx,
        dt_step=1.0,
        total_steps=12,
        dt_context=dt_context_pre,
    )
    del model_pre
    torch.cuda.empty_cache()

    print("Inferencing Fine-tuned...")
    model_align, cfg_align = load_config_and_model(ckpt_align, device)
    dt_context_align = float(cfg_align.get("dt_ref", DT_REF))
    seq_align = run_inference_sequence(
        model_align,
        x_ctx,
        dt_step=1.0,
        total_steps=12,
        dt_context=dt_context_align,
    )
    del model_align
    torch.cuda.empty_cache()

    best_ch = find_best_channel(data, seq_pre, seq_align, num_channels=30)

    save_name = f"forecast_comparison_dt1h_ch{best_ch}.pdf"
    print(f"\nPlotting best channel: {best_ch}...")
    visualize_comparison(
        data,
        seq_pre,
        seq_align,
        channel_idx=best_ch,
        save_path=save_name,
    )


if __name__ == "__main__":
    main()
