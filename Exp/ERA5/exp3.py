import os
import sys
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel


def load_config_and_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
            "dt_ref": 6.0,
            "sde_mode": "sde",
            "init_noise_scale": 0.0001,
            "max_growth_rate": 0.3,
            "ensemble_size": 4,
        }

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

    model = UniPhyModel(**filtered_cfg).to(device)
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model, model_cfg


def extract_block_timescales(block, dim, num_modes, h_p, w_p, device):
    eff_dim = min(dim, num_modes)
    probe = torch.eye(dim, device=device)[:eff_dim]
    probe_input = probe.view(eff_dim, dim, 1, 1).expand(-1, -1, h_p, w_p)

    h_prev = torch.zeros(eff_dim, h_p, w_p, dim, device=device)
    flux_prev = torch.zeros(eff_dim, dim, device=device)
    dt = torch.tensor(1.0, device=device)

    with torch.no_grad():
        z_out, _, _ = block.forward_step(probe_input, h_prev, dt, flux_prev)
        a_matrix = z_out.mean(dim=[-1, -2])[:, :eff_dim].cpu().numpy()

    try:
        l_matrix = scipy.linalg.logm(a_matrix)
    except Exception:
        l_matrix = a_matrix - np.eye(len(a_matrix))

    evals = np.linalg.eigvals(l_matrix)
    max_real = np.max(evals.real)
    if max_real > -1e-6:
        shift = max_real + 1e-5
        evals = evals - shift

    stable_evals = evals[evals.real < -1e-8]
    if len(stable_evals) == 0:
        return np.array([])

    decay_rates = np.abs(stable_evals.real)
    tau_days = (1.0 / decay_rates) * 0.25
    return tau_days


def analyze_memory_timescales(ckpt_path, save_path="memory_timescales.pdf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, cfg = load_config_and_model(ckpt_path, device)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Scanning all blocks for hierarchical memory structure...")

    all_taus = []
    block_taus = {}

    for i, block in enumerate(model.blocks):
        taus = extract_block_timescales(
            block,
            model.embed_dim,
            64,
            model.h_patches,
            model.w_patches,
            device,
        )
        all_taus.append(taus)
        block_taus[f"Block {i}"] = taus
        print(f"  Block {i}: Found {len(taus)} stable modes.")

    tau_days = np.concatenate(all_taus)

    if len(tau_days) == 0:
        print("No valid timescales extracted.")
        return

    tau_days_viz = np.clip(tau_days, 0, 60)
    total = len(tau_days)
    short = np.sum(tau_days < 5)
    medium = np.sum((tau_days >= 5) & (tau_days < 20))
    long_term = np.sum(tau_days >= 20)

    print(f"\nSystem Memory Distribution (All Layers):")
    print(f"  Short (< 5 days):   {short / total * 100:.1f}%")
    print(f"  Medium (5-20 days): {medium / total * 100:.1f}%")
    print(f"  Long (> 20 days):   {long_term / total * 100:.1f}%")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1 = axes[0]

    sns.kdeplot(
        tau_days_viz,
        fill=True,
        color="#2ca02c",
        alpha=0.3,
        linewidth=2.5,
        label="Learned Hierarchy",
        clip=(0, 60),
        bw_adjust=0.6,
        ax=ax1,
    )

    current_ylim = ax1.get_ylim()
    ax1.set_ylim(0, current_ylim[1] * 1.2)
    y_max = ax1.get_ylim()[1]

    ax1.axvline(x=5, color="#555555", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axvline(x=20, color="#555555", linestyle="--", alpha=0.5, linewidth=1)

    text_style = dict(
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
        fontweight="medium",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=0.85
        ),
    )
    ax1.text(2.5, y_max * 0.96, "Synoptic\n(< 5 days)", **text_style)
    ax1.text(12.5, y_max * 0.96, "Sub-seasonal\n(5-20 days)", **text_style)
    ax1.text(40, y_max * 0.96, "Long-term\n(> 20 days)", **text_style)

    ax1.set_xlabel("Characteristic Memory Timescale (Days)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(
        "(a) Spectral Distribution of Memory Timescales",
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlim(0, 65)
    ax1.set_xticks([0, 5, 10, 20, 30, 40, 50, 60])
    ax1.grid(True, linestyle=":", alpha=0.3)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    ax2 = axes[1]

    block_names = []
    block_data = []
    median_values = []

    all_values = []
    for name, taus in block_taus.items():
        if len(taus) > 0:
            all_values.extend(taus.tolist())

    p95 = np.percentile(all_values, 95)
    y_upper = min(p95 * 1.2, 60)

    for name, taus in block_taus.items():
        if len(taus) > 0:
            clipped_taus = np.clip(taus, 0, y_upper)
            block_names.append(name)
            block_data.append(clipped_taus)
            median_values.append(np.median(clipped_taus))

    colors = []
    cmap = plt.cm.RdYlGn_r
    norm_vals = np.array(median_values)
    norm_vals = (norm_vals - norm_vals.min()) / (
        norm_vals.max() - norm_vals.min() + 1e-8
    )

    for nv in norm_vals:
        colors.append(cmap(nv * 0.8 + 0.1))

    bp = ax2.boxplot(
        block_data,
        labels=block_names,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        whis=[5, 95],
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("#1a1a1a")
        median.set_linewidth(2)

    ax2.axhline(
        y=5,
        color="#888888",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="5-day threshold",
    )
    ax2.axhline(
        y=20,
        color="#888888",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
        label="20-day threshold",
    )

    ax2.set_xlabel("Network Depth")
    ax2.set_ylabel("Memory Timescale (Days)")
    ax2.set_title(
        "(b) Hierarchical Memory Structure by Layer",
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylim(0, y_upper * 1.1)
    ax2.grid(True, linestyle=":", alpha=0.3, axis="y")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, y_upper))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Median Timescale (Days)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")

    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"PNG version saved to {png_path}")

    plt.show()


if __name__ == "__main__":
    ckpt_file = "./uniphy/align_ckpt/align_epoch10.pt"
    analyze_memory_timescales(ckpt_file)

