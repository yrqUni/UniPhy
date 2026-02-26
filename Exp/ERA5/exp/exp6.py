import glob
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5Dataset
from exp_utils import get_device, load_config_and_model


CKPT_GLOB = "./uniphy/ckpt/ckpt_epoch200.pt"
OUTDIR = "./exp6_gif"

COND_STEPS = 4
INFER_DT = 6.0
INFER_STEPS = 12

FPS = 1
PERCENTILE_LO = 1.0
PERCENTILE_HI = 99.0

DATA_INPUT_DIR = "/nfs/ERA5_data/data_norm"
DATA_YEAR_RANGE = [2009, 2009]
DATA_SAMPLING_MODE = "sequential"


SURFACE_VARS = ["TCWV", "U10", "V10", "T2", "MSLP", "SP"]
VAR = ["VV", "U", "V", "RH", "T", "Z"]
HEIGHTS = ["925", "850", "500", "100"]
PRESSURE_VARS = [v + h for v in VAR for h in HEIGHTS]
CHANNEL_NAMES = SURFACE_VARS + PRESSURE_VARS


def pick_ckpt_path():
    paths = sorted(glob.glob(CKPT_GLOB), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found: {CKPT_GLOB}")
    return paths[-1]


def load_model(device):
    ckpt_path = pick_ckpt_path()
    model, model_cfg = load_config_and_model(ckpt_path, device)
    dt_ref = float(model_cfg.get("dt_ref", 6.0))
    ensemble_size = int(model_cfg.get("ensemble_size", 1))
    return model, dt_ref, ensemble_size, ckpt_path


def get_sample(dt_ref, device):
    ds = ERA5Dataset(
        input_dir=DATA_INPUT_DIR,
        year_range=DATA_YEAR_RANGE,
        window_size=COND_STEPS + 1,
        sample_k=COND_STEPS + 1,
        look_ahead=0,
        is_train=False,
        dt_ref=float(dt_ref),
        sampling_mode=DATA_SAMPLING_MODE,
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
    )
    data, _ = next(iter(loader))
    return data.to(device, non_blocking=True).float(), ds


@torch.no_grad()
def rollout_prediction(model, x_ctx, dt_context):
    device = x_ctx.device
    b = x_ctx.shape[0]
    t_in = x_ctx.shape[1]
    dt_list = [
        torch.tensor(INFER_DT, device=device, dtype=torch.float32)
        for _ in range(int(INFER_STEPS))
    ]
    dt_ctx = torch.full(
        (b, t_in), dt_context, device=device, dtype=torch.float32,
    )
    pred_seq = model.forward_rollout(x_ctx, dt_ctx, dt_list)
    if pred_seq.is_complex():
        pred_seq = pred_seq.real
    return pred_seq


def fig_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def save_channel_gif(series, varname):
    frames = []

    for t in range(series.shape[0]):
        img = series[t]
        vmin = float(np.percentile(img, PERCENTILE_LO))
        vmax = float(np.percentile(img, PERCENTILE_HI))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1.0

        fig = plt.figure(figsize=(6.2, 3.2))
        plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        plt.axis("off")
        hours = float(t) * INFER_DT
        plt.title(f"{varname}  t={hours:.0f}h", fontsize=14)
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    out_path = os.path.join(OUTDIR, f"{varname}.gif")
    imageio.mimsave(out_path, frames, fps=int(FPS))


def main():
    device = get_device()
    os.makedirs(OUTDIR, exist_ok=True)

    model, dt_ref, ensemble_size, ckpt_path = load_model(device)
    print(f"Using ckpt: {ckpt_path}")
    print(f"dt_ref={dt_ref}, ensemble_size={ensemble_size}")

    data, ds = get_sample(dt_ref, device)
    x_ctx = data[:, :int(COND_STEPS)]

    pred_mean = rollout_prediction(model, x_ctx, dt_ref)
    pred = pred_mean[0].detach().cpu().numpy()

    n_channels = min(pred.shape[1], len(CHANNEL_NAMES))
    for c in range(n_channels):
        name = CHANNEL_NAMES[c]
        print("GIF:", name)
        save_channel_gif(pred[:, c], name)

    ds.cleanup()


if __name__ == "__main__":
    main()
