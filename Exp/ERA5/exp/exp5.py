import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch.utils.data import DataLoader

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel


DT_REF = 6.0

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"] = 9
rcParams["axes.titlesize"] = 9
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


def load_config_and_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return None, None

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "cfg" in checkpoint:
        model_cfg = checkpoint["cfg"]["model"]
    else:
        model_cfg = {
            "in_channels": 30,
            "out_channels": 30,
            "embed_dim": 512,
            "expand": 4,
            "depth": 8,
            "patch_size": (7, 15),
            "img_height": 721,
            "img_width": 1440,
            "dt_ref": DT_REF,
            "sde_mode": "sde",
            "init_noise_scale": 0.0001,
            "max_growth_rate": 0.3,
            "ensemble_size": 4,
        }

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if "encoder.pos_emb" in state_dict:
        model_cfg["embed_dim"] = state_dict["encoder.pos_emb"].shape[1]

    valid_args = {
        "in_channels",
        "out_channels",
        "embed_dim",
        "expand",
        "depth",
        "patch_size",
        "img_height",
        "img_width",
        "dt_ref",
        "sde_mode",
        "init_noise_scale",
        "ensemble_size",
        "max_growth_rate",
        "num_experts",
    }
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in valid_args}
    filtered_cfg["patch_size"] = tuple(filtered_cfg["patch_size"])

    model = UniPhyModel(**filtered_cfg).to(device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, model_cfg


def get_vis_data(device):
    try:
        dataset = ERA5_Dataset(
            input_dir="/nfs/ERA5_data/data_norm",
            year_range=[2009, 2009],
            window_size=3,
            sample_k=3,
            sampling_mode="sequential",
            is_train=False,
            dt_ref=DT_REF,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        data, _ = next(iter(loader))
        return data.to(device)
    except Exception as e:
        print(f"Data loading failed ({e}). Using random tensors.")
        return torch.randn(1, 3, 30, 721, 1440, device=device)


@torch.no_grad()
def run_inference_sequence_ensemble_mean(model, x_ctx, dt_step, total_steps, dt_context, seed=1234):
    device = x_ctx.device
    dt_list = [
        torch.tensor(float(dt_step), device=device, dtype=torch.float32)
        for _ in range(int(total_steps))
    ]

    ensemble_size = int(getattr(model, "ensemble_size", 1))
    if hasattr(model, "decoder") and hasattr(model.decoder, "ensemble_size"):
        ensemble_size = int(model.decoder.ensemble_size)

    if ensemble_size <= 1:
        return model.forward_rollout(x_ctx, float(dt_context), dt_list, member_idx=None)

    preds = []
    for m in range(ensemble_size):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        member_idx = torch.full((x_ctx.shape[0],), m, device=device, dtype=torch.long)
        pred_seq = model.forward_rollout(x_ctx, float(dt_context), dt_list, member_idx=member_idx)
        preds.append(pred_seq)

    return torch.stack(preds, dim=0).mean(dim=0)


def compute_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def find_best_channel(real_data, pre_seq, fine_seq, num_channels=30):
    step_indices = [5, 11]
    gt_indices = [1, 2]

    best_ch = 0
    best_improvement = -float("inf")
    channel_stats = []

    print("\nChannel-wise RMSE comparison:")
    print("-" * 70)
    print(f"{'Ch':>4} | {'Pre-RMSE':>10} | {'Fine-RMSE':>10} | {'Improv%':>10} | {'Status'}")
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
            f"{ch:>4} | {pre_rmse_avg:>10.4f} | {fine_rmse_avg:>10.4f} | "
            f"{improvement:>+10.2f}% | {status}"
        )

        channel_stats.append(
            {
                "ch": ch,
                "pre_rmse": pre_rmse_avg,
                "fine_rmse": fine_rmse_avg,
                "improvement": improvement,
            }
        )

        if improvement > best_improvement:
            best_improvement = improvement
            best_ch = ch

    print("-" * 70)
    print(f"Best channel: {best_ch} (improvement: {best_improvement:+.2f}%)")
    return best_ch, channel_stats


def visualize_comparison(real_data, pre_seq, fine_seq, channel_idx, save_path):
    step_indices = [5, 11]
    time_labels = ["6h", "12h"]
    gt_indices = [1, 2]
    row_labels = ["Ground Truth", "Pre-trained", "Fine-tuned"]

    fig, axes = plt.subplots(
        3, 2, figsize=(4.2, 4.0), gridspec_kw={"wspace": 0.05, "hspace": 0.15}
    )

    gt_imgs = [real_data[0, idx, channel_idx].cpu().numpy() for idx in gt_indices]
    pre_imgs = [pre_seq[0, idx, channel_idx].cpu().numpy() for idx in step_indices]
    fine_imgs = [fine_seq[0, idx, channel_idx].cpu().numpy() for idx in step_indices]

    all_imgs = gt_imgs + pre_imgs + fine_imgs
    vmin = min(img.min() for img in all_imgs)
    vmax = max(img.max() for img in all_imgs)

    cmap = "RdBu_r"
    im = None

    for i in range(2):
        ax = axes[0, i]
        img = gt_imgs[i]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(rf"$t$ = {time_labels[i]}", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    for i in range(2):
        ax = axes[1, i]
        pred_img = pre_imgs[i]
        gt_img = gt_imgs[i]
        ax.imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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

    for i in range(2):
        ax = axes[2, i]
        pred_img = fine_imgs[i]
        gt_img = gt_imgs[i]
        ax.imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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
        rf"Ensemble-Mean Temporal Generalization ($\Delta t$=1h) - Channel {channel_idx}",
        fontsize=10,
        y=0.98,
    )

    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Figure saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Experiment 5: Visualizing Temporal Consistency (Ensemble-Mean, dt=1h)")
    print("=" * 60)

    data = get_vis_data(device)
    x_ctx = data[:, 0:1]

    ckpt_pre_list = sorted(glob.glob("./uniphy/ckpt/*.pt"), key=os.path.getmtime)
    ckpt_align_list = sorted(glob.glob("./uniphy/align_ckpt/*.pt"), key=os.path.getmtime)
    ckpt_pre = ckpt_pre_list[-1] if ckpt_pre_list else None
    ckpt_align = ckpt_align_list[-1] if ckpt_align_list else None

    if not ckpt_pre or not ckpt_align:
        print("Checkpoints not found.")
        return

    print("Inferencing Pre-trained (ensemble mean)...")
    model_pre, cfg_pre = load_config_and_model(ckpt_pre, device)
    dt_context_pre = float(cfg_pre.get("dt_ref", DT_REF)) if cfg_pre is not None else DT_REF
    seq_pre = run_inference_sequence_ensemble_mean(
        model_pre, x_ctx, dt_step=1.0, total_steps=12, dt_context=dt_context_pre
    )
    del model_pre
    torch.cuda.empty_cache()

    print("Inferencing Fine-tuned (ensemble mean)...")
    model_align, cfg_align = load_config_and_model(ckpt_align, device)
    dt_context_align = float(cfg_align.get("dt_ref", DT_REF)) if cfg_align is not None else DT_REF
    seq_align = run_inference_sequence_ensemble_mean(
        model_align, x_ctx, dt_step=1.0, total_steps=12, dt_context=dt_context_align
    )
    del model_align
    torch.cuda.empty_cache()

    best_ch, _ = find_best_channel(data, seq_pre, seq_align, num_channels=30)

    save_name = f"forecast_comparison_dt1h_ensemblemean_ch{best_ch}.pdf"
    print(f"\nPlotting best channel: {best_ch}...")
    visualize_comparison(data, seq_pre, seq_align, channel_idx=best_ch, save_path=save_name)


if __name__ == "__main__":
    main()

