import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

import warnings
warnings.filterwarnings("ignore")


def load_stats(stats_dir, device):
    try:
        mean = np.load(os.path.join(stats_dir, "mean.npy"))
        std = np.load(os.path.join(stats_dir, "std.npy"))
        mean_tensor = torch.from_numpy(mean).to(device).view(1, -1, 1, 1).float()
        std_tensor = torch.from_numpy(std).to(device).view(1, -1, 1, 1).float()
        return mean_tensor, std_tensor
    except Exception as e:
        print(f"Warning: Could not load stats from {stats_dir}: {e}")
        return None, None


def denormalize(x, mean, std):
    if mean is None or std is None:
        return x
    return x * std + mean


def load_model(cfg, device):
    model = UniPhyModel(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        expand=cfg["model"]["expand"],
        depth=cfg["model"]["depth"],
        patch_size=cfg["model"]["patch_size"],
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
        max_growth_rate=cfg["model"]["max_growth_rate"],
    ).to(device)

    ckpt_path = cfg["inference"]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def visualize(input_frame, pred_frame, target_frame, step_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ch_idx = 0
    in_img = input_frame[0, ch_idx].cpu().numpy()
    pred_img = pred_frame[0, ch_idx].cpu().numpy()
    tgt_img = target_frame[0, ch_idx].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin = min(in_img.min(), pred_img.min(), tgt_img.min())
    vmax = max(in_img.max(), pred_img.max(), tgt_img.max())

    im0 = axes[0].imshow(in_img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Input (Step {step_idx})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction (Step {step_idx+1})")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(tgt_img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Ground Truth (Step {step_idx+1})")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"step_{step_idx+1:03d}.png"))
    plt.close()


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    mean, std = load_stats(cfg["data"]["stats_dir"], device)

    dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=[cfg["data"]["test_year"], cfg["data"]["test_year"]],
        window_size=cfg["data"]["window_size"],
        sample_k=cfg["inference"]["autoregressive_steps"] + 1,
        look_ahead=0,
        is_train=False,
        dt_ref=cfg["data"]["dt_ref"],
        sampling_mode="sequential",
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data, dt_data = next(iter(loader))

    curr_x = data[:, 0].to(device).float()
    targets = data[:, 1:].to(device).float()
    dt_steps = dt_data[:, :-1].to(device).float()

    predictions = []
    ensemble_size = cfg["inference"]["ensemble_size"]

    print(f"Starting inference for {cfg['inference']['autoregressive_steps']} steps...")

    with torch.no_grad():
        for step in range(targets.shape[1]):
            dt = dt_steps[:, step]
            member_preds = []
            for m in range(ensemble_size):
                member_idx = torch.tensor([m], device=device).repeat(curr_x.shape[0])
                pred_raw = model(curr_x, dt, member_idx=member_idx)
                if pred_raw.is_complex():
                    pred_raw = pred_raw.real
                member_preds.append(pred_raw)

            member_stack = torch.stack(member_preds, dim=0)
            pred_mean = member_stack.mean(dim=0)
            predictions.append(pred_mean)
            curr_x = pred_mean

            target_step = targets[:, step]
            pred_phys = denormalize(pred_mean, mean, std)
            target_phys = denormalize(target_step, mean, std)
            input_phys = denormalize(data[:, step].to(device), mean, std)

            rmse = torch.sqrt(((pred_phys - target_phys) ** 2).mean())
            print(f"Step {step+1} (dt={dt.item()}): RMSE = {rmse.item():.4f}")

            if cfg["inference"]["save_plot"]:
                visualize(
                    input_phys,
                    pred_phys,
                    target_phys,
                    step,
                    cfg["inference"]["output_dir"],
                )

    print("Inference completed.")


if __name__ == "__main__":
    with open("infer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg)

