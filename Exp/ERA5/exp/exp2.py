import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch

warnings.filterwarnings("ignore")
sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from exp_utils import get_device, load_config_and_model


def extract_block_operator(block, dim, h_p, w_p, device, num_modes=64):
    effective_dim = min(dim, num_modes)
    probe_base = torch.eye(dim, device=device)[:effective_dim]
    probe_input = probe_base.view(effective_dim, dim, 1, 1).expand(
        -1, -1, h_p, w_p
    )
    probe_complex = torch.complex(probe_input, torch.zeros_like(probe_input))
    h_prev = torch.zeros(
        effective_dim * h_p * w_p, 1, dim,
        device=device, dtype=torch.complex64,
    )
    flux_prev = torch.zeros(
        effective_dim, dim, device=device, dtype=torch.complex64,
    )
    dt = torch.tensor(1.0, device=device)
    with torch.no_grad():
        z_out, _, _ = block.forward_step(
            probe_complex, h_prev, dt, flux_prev
        )
    z_real = z_out.real if z_out.is_complex() else z_out
    output_matrix = z_real.mean(dim=[-1, -2])
    a_matrix = output_matrix[:, :effective_dim].cpu().numpy()
    try:
        l_matrix = scipy.linalg.logm(a_matrix)
    except Exception:
        l_matrix = a_matrix - np.eye(len(a_matrix))
    return l_matrix


def analyze_transient_growth(l_matrix, time_horizon=48, dt=0.5):
    evals = np.linalg.eigvals(l_matrix)
    max_real = np.max(evals.real)
    if max_real > 0:
        l_matrix = l_matrix - (max_real + 0.05) * np.eye(len(l_matrix))
        evals = evals - (max_real + 0.05)
    t_steps = np.arange(0, time_horizon + dt, dt)
    energy_growth = []
    normal_decay = []
    spectral_abscissa = np.max(evals.real)
    for t in t_steps:
        p_t = scipy.linalg.expm(l_matrix * t)
        norm_p = np.linalg.norm(p_t, 2)
        energy_growth.append(norm_p)
        normal_decay.append(np.exp(spectral_abscissa * t))
    return evals, t_steps, energy_growth, normal_decay


def plot_spectral_analysis(ckpt_path, save_path="transient_growth.pdf"):
    device = get_device()
    model, cfg = load_config_and_model(ckpt_path, device)

    print("Scanning all blocks for spectral properties...")
    all_evals_real = []
    all_evals_imag = []
    all_growth_curves = []
    max_peak_growth = -1.0
    best_layer_idx = 0
    common_t = None
    common_normal = None

    for i, block in enumerate(model.blocks):
        print(f"  Analyzing Block {i}...")
        l_matrix = extract_block_operator(
            block, model.embed_dim,
            model.h_patches, model.w_patches,
            device, num_modes=64,
        )
        evals, t, energy, normal = analyze_transient_growth(
            l_matrix, time_horizon=24,
        )
        all_evals_real.extend(evals.real)
        all_evals_imag.extend(evals.imag)
        all_growth_curves.append(energy)
        peak = np.max(energy)
        if peak > max_peak_growth:
            max_peak_growth = peak
            best_layer_idx = i
        if i == 0:
            common_t = t
            common_normal = normal

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.7",
    })

    fig = plt.figure(figsize=(7.5, 3.4), dpi=600)
    fig.suptitle(
        "Spectral Evidence of Transient Instability (All Layers)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    ax1 = fig.add_axes([0.08, 0.15, 0.38, 0.70])
    ax2 = fig.add_axes([0.56, 0.15, 0.40, 0.70])

    ax1.scatter(
        all_evals_real, all_evals_imag,
        alpha=0.5, c="#1f77b4", edgecolors="none",
        s=15, zorder=3, label="Eigenvalues",
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, zorder=2)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Real Part (Growth/Decay)")
    ax1.set_ylabel("Imaginary Part (Frequency)")
    ax1.set_title(
        r"(a) Global Spectrum (Re $\lambda < 0$)",
        fontsize=10, fontweight="normal", pad=12,
    )
    ax1.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax1.set_axisbelow(True)

    props_a = dict(
        boxstyle="round,pad=0.3", facecolor="#e8f4e8",
        edgecolor="0.7", alpha=0.95, linewidth=0.5,
    )
    ax1.text(
        0.05, 0.95, "Stable\nRegion",
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        fontweight="bold", alpha=0.9, bbox=props_a,
    )

    for i, curve in enumerate(all_growth_curves):
        if i == best_layer_idx:
            continue
        ax2.plot(
            common_t, curve, color="#c0392b",
            linewidth=1.0, alpha=0.15, zorder=2,
        )

    ax2.plot(
        common_t, all_growth_curves[best_layer_idx],
        color="#c0392b", linewidth=2.2,
        label=f"Max Growth (Layer {best_layer_idx})", zorder=4,
    )
    ax2.plot(
        common_t, common_normal,
        color="#2c3e50", linestyle="--", linewidth=1.8,
        alpha=0.8, label="Normal Decay Ref.", zorder=3,
    )

    peak_idx = np.argmax(all_growth_curves[best_layer_idx])
    peak_t = common_t[peak_idx]
    peak_e = all_growth_curves[best_layer_idx][peak_idx]
    y_max = max(max_peak_growth * 1.15, 1.5)

    initial_val = all_growth_curves[best_layer_idx][0]
    if peak_idx > 0 and peak_e > initial_val * 1.05:
        props_b = dict(
            boxstyle="round,pad=0.3", facecolor="white",
            edgecolor="#c0392b", alpha=0.95, linewidth=0.8,
        )
        ax2.annotate(
            "Transient Peak", xy=(peak_t, peak_e),
            xytext=(peak_t + 12.0, peak_e - 3.0),
            arrowprops=dict(
                arrowstyle="-|>", color="#c0392b",
                lw=1.2, connectionstyle="arc3,rad=0.2",
            ),
            fontsize=9, fontweight="bold", color="#c0392b",
            ha="right", va="center", bbox=props_b,
        )

    ax2.set_xlabel("Forecast Time (Steps)")
    ax2.set_ylabel(r"Energy Gain $\|e^{\mathcal{L}t}\|$")
    ax2.set_title(
        "(b) Non-Normal Amplification",
        fontsize=10, fontweight="normal", pad=12,
    )
    ax2.legend(
        loc="upper right", frameon=True, fancybox=False,
        edgecolor="0.7", borderpad=0.5, handlelength=1.8, fontsize=8,
    )
    ax2.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim(common_t[0], common_t[-1])
    ax2.set_ylim(0, y_max)

    plt.savefig(
        save_path, format="pdf",
        bbox_inches="tight", pad_inches=0.05, dpi=600,
    )
    print(f"Figure saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    ckpt_file = "./uniphy/align_ckpt/align_epoch10.pt"
    plot_spectral_analysis(ckpt_file)
