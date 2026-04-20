import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        ),
    )

from Exp.ERA5.exp.exp_utils import (
    find_latest_checkpoint,
    get_device,
    load_config_and_model,
)


def visualize_latitude_metric_proxy(
    ckpt_path, save_path="positional_embedding_metric_proxy.pdf"
):
    device = get_device()
    model, cfg = load_config_and_model(ckpt_path, device)

    pos_emb = model.encoder.pos_emb.detach().cpu()
    _, dim, h_p, w_p = pos_emb.shape
    spatial_energy = torch.norm(pos_emb, dim=1).squeeze().numpy()
    spatial_energy_norm = (spatial_energy - spatial_energy.min()) / (
        spatial_energy.max() - spatial_energy.min()
    )
    zonal_mean = np.mean(spatial_energy, axis=1)
    zonal_mean_norm = (zonal_mean - zonal_mean.min()) / (
        zonal_mean.max() - zonal_mean.min()
    )
    latitudes = np.linspace(-90, 90, h_p)
    lat_rad = np.deg2rad(latitudes)
    metric_proxy = np.power(np.clip(np.cos(lat_rad), 1e-3, None), -0.5)
    metric_proxy_norm = (metric_proxy - metric_proxy.min()) / (
        metric_proxy.max() - metric_proxy.min()
    )
    correlation = np.corrcoef(zonal_mean_norm, metric_proxy_norm)[0, 1]
    print(f"Correlation with latitude metric proxy: {correlation:.4f}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.fontsize": 9,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.7",
            "text.usetex": False,
        }
    )

    fig = plt.figure(figsize=(7.5, 3.6), dpi=600)
    fig.suptitle(
        "Positional Embedding Latitude-Metric Proxy",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    ax1 = fig.add_axes([0.06, 0.13, 0.32, 0.70])
    cax = fig.add_axes([0.39, 0.13, 0.012, 0.70])
    ax2 = fig.add_axes([0.58, 0.13, 0.38, 0.70])

    im = ax1.imshow(
        spatial_energy_norm,
        cmap="magma",
        aspect="auto",
        extent=[-180, 180, -90, 90],
        origin="lower",
        interpolation="bilinear",
    )
    ax1.set_title(
        r"(a) Positional Embedding Energy",
        fontsize=10,
        fontweight="normal",
        pad=12,
    )
    ax1.set_xlabel("Longitude (\u00b0)")
    ax1.set_ylabel("Latitude (\u00b0)")
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_yticks([-90, -45, 0, 45, 90])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(
        r"$\|\mathbf{h}\|_2$ (norm.)",
        fontsize=9,
        rotation=270,
        labelpad=14,
    )

    color_theory = "#2c3e50"
    color_learned = "#c0392b"

    ax2.plot(
        metric_proxy_norm,
        latitudes,
        color=color_theory,
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=r"Implemented proxy: $(\cos\phi)^{-1/2}$",
        zorder=2,
    )
    ax2.plot(
        zonal_mean_norm,
        latitudes,
        color=color_learned,
        linewidth=2.2,
        label="Learned",
        zorder=3,
    )
    ax2.fill_betweenx(
        latitudes,
        zonal_mean_norm,
        0,
        color=color_learned,
        alpha=0.12,
        zorder=1,
    )

    textstr = f"$r = {correlation:.3f}$"
    props = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.6",
        alpha=0.95,
        linewidth=0.8,
    )
    ax2.text(
        0.94,
        0.06,
        textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        fontweight="bold",
    )

    ax2.set_title(
        "(b) Latitude Metric Proxy Match",
        fontsize=10,
        fontweight="normal",
        pad=12,
    )
    ax2.set_xlabel("Normalized Magnitude")
    ax2.set_ylabel("Latitude (\u00b0)")
    ax2.set_yticks([-90, -45, 0, 45, 90])
    ax2.set_xlim(-0.05, 1.10)
    ax2.set_ylim(-90, 90)
    ax2.grid(True, linestyle=":", alpha=0.5, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.64),
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        borderpad=0.5,
        handlelength=1.8,
    )

    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=600,
    )
    plt.savefig(
        save_path.replace(".pdf", ".png"),
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=600,
    )
    print(f"Figure saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    ckpt_file = find_latest_checkpoint(
        "./uniphy/align_ckpt/*.pt",
        "./**/align_ckpt/*.pt",
    )
    if ckpt_file is None:
        print("Fine-tuned checkpoint not found.")
    else:
        visualize_latitude_metric_proxy(ckpt_file)
