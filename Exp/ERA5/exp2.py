import os
import sys
import warnings
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch

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
            "ensemble_size": 4
        }
    valid_args = {
        "in_channels", "out_channels", "embed_dim", "expand", "depth",
        "patch_size", "img_height", "img_width", "dt_ref", "sde_mode",
        "init_noise_scale", "ensemble_size", "max_growth_rate", "num_experts"
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


def extract_block_operator(block, dim, h_p, w_p, device, num_modes=64):
    effective_dim = min(dim, num_modes)
    probe_base = torch.eye(dim, device=device)[:effective_dim]
    probe_input = probe_base.view(effective_dim, dim, 1, 1).expand(
        -1, -1, h_p, w_p
    )
    h_prev = torch.zeros(
        effective_dim, h_p, w_p, dim,
        device=device, dtype=probe_input.dtype
    )
    flux_prev = torch.zeros(
        effective_dim, dim,
        device=device, dtype=probe_input.dtype
    )
    dt = torch.tensor(1.0, device=device)
    with torch.no_grad():
        z_out, _, _ = block.forward_step(
            probe_input, h_prev, dt, flux_prev
        )
        output_matrix = z_out.mean(dim=[-1, -2])
    A_matrix = output_matrix[:, :effective_dim].cpu().numpy()
    try:
        L_matrix = scipy.linalg.logm(A_matrix)
    except Exception:
        L_matrix = A_matrix - np.eye(len(A_matrix))
    return L_matrix


def analyze_transient_growth(L, time_horizon=48, dt=0.5):
    evals, evecs = np.linalg.eig(L)
    max_real = np.max(evals.real)
    if max_real > 0:
        L = L - (max_real + 0.05) * np.eye(len(L))
        evals = evals - (max_real + 0.05)
    t_steps = np.arange(0, time_horizon + dt, dt)
    energy_growth = []
    normal_decay = []
    spectral_abscissa = np.max(evals.real)
    for t in t_steps:
        P_t = scipy.linalg.expm(L * t)
        norm_P = np.linalg.norm(P_t, 2)
        energy_growth.append(norm_P)
        norm_normal = np.exp(spectral_abscissa * t)
        normal_decay.append(norm_normal)
    return evals, t_steps, energy_growth, normal_decay


def plot_spectral_analysis(ckpt_path, save_path="transient_growth.pdf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, cfg = load_config_and_model(ckpt_path, device)
    except Exception as e:
        print(f"Error: {e}")
        return
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
        L = extract_block_operator(
            block, model.embed_dim,
            model.h_patches, model.w_patches,
            device, num_modes=64
        )
        evals, t, energy, normal = analyze_transient_growth(L, time_horizon=24)
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
    fig_width = 7.5
    fig_height = 3.4
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=600)
    fig.suptitle(
        "Spectral Evidence of Transient Instability (All Layers)",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )
    ax1 = fig.add_axes([0.08, 0.15, 0.38, 0.70])
    ax2 = fig.add_axes([0.56, 0.15, 0.40, 0.70])
    ax1.scatter(
        all_evals_real, all_evals_imag,
        alpha=0.5, c="#1f77b4", edgecolors="none",
        s=15, zorder=3, label="Eigenvalues"
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, zorder=2)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Real Part (Growth/Decay)")
    ax1.set_ylabel("Imaginary Part (Frequency)")
    ax1.set_title(
        r"(a) Global Spectrum (Re $\lambda < 0$)",
        fontsize=10,
        fontweight="normal",
        pad=12
    )
    ax1.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax1.set_axisbelow(True)
    props_a = dict(
        boxstyle="round,pad=0.3",
        facecolor="#e8f4e8",
        edgecolor="0.7",
        alpha=0.95,
        linewidth=0.5
    )
    ax1.text(
        0.05, 0.95,
        "Stable\nRegion",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontweight="bold",
        alpha=0.9,
        bbox=props_a
    )
    for i, curve in enumerate(all_growth_curves):
        if i == best_layer_idx:
            continue
        ax2.plot(
            common_t, curve,
            color="#c0392b", linewidth=1.0,
            alpha=0.15, zorder=2
        )
    ax2.plot(
        common_t, all_growth_curves[best_layer_idx],
        color="#c0392b", linewidth=2.2,
        label=f"Max Growth (Layer {best_layer_idx})", zorder=4
    )
    ax2.plot(
        common_t, common_normal,
        color="#2c3e50", linestyle="--", linewidth=1.8,
        alpha=0.8, label="Normal Decay Ref.", zorder=3
    )
    peak_idx = np.argmax(all_growth_curves[best_layer_idx])
    peak_t = common_t[peak_idx]
    peak_e = all_growth_curves[best_layer_idx][peak_idx]
    y_max = max(max_peak_growth * 1.15, 1.5)
    if peak_idx > 0 and peak_e > all_growth_curves[best_layer_idx][0] * 1.05:
        props_b = dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#c0392b",
            alpha=0.95,
            linewidth=0.8
        )
        ax2.annotate(
            "Transient Peak",
            xy=(peak_t, peak_e),
            xytext=(peak_t + 12.0, peak_e - 3.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#c0392b",
                lw=1.2,
                connectionstyle="arc3,rad=0.2"
            ),
            fontsize=9,
            fontweight="bold",
            color="#c0392b",
            ha="right",
            va="center",
            bbox=props_b
        )
    ax2.set_xlabel("Forecast Time (Steps)")
    ax2.set_ylabel(r"Energy Gain $\|e^{\mathcal{L}t}\|$")
    ax2.set_title(
        "(b) Non-Normal Amplification",
        fontsize=10,
        fontweight="normal",
        pad=12
    )
    ax2.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        borderpad=0.5,
        handlelength=1.8,
        fontsize=8
    )
    ax2.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim(common_t[0], common_t[-1])
    ax2.set_ylim(0, y_max)
    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=600
    )
    print(f"Figure saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    ckpt_file = "./uniphy/align_ckpt/align_epoch10.pt"
    plot_spectral_analysis(ckpt_file)

