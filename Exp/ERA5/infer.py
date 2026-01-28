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


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    train_cfg = checkpoint["cfg"]
    model_cfg = train_cfg["model"]

    valid_args = {
        "in_channels", "out_channels", "embed_dim", "expand", "depth",
        "patch_size", "img_height", "img_width", "dt_ref", "sde_mode",
        "init_noise_scale", "ensemble_size", "max_growth_rate"
    }
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in valid_args}

    model = UniPhyModel(**filtered_cfg).to(device)

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model, model_cfg


def visualize(input_frame, pred_frame, target_frame, step_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    ch_idx = 0

    in_img = input_frame[0, ch_idx].cpu().numpy()
    pred_img = pred_frame[0, ch_idx].cpu().numpy()
    tgt_img = target_frame[0, ch_idx].cpu().numpy()

    combined_data = np.concatenate([
        in_img.flatten(), pred_img.flatten(), tgt_img.flatten()
    ])
    vmin = np.percentile(combined_data, 2)
    vmax = np.percentile(combined_data, 98)

    cmap = "turbo"

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    im0 = axes[0].imshow(in_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Input (Step {step_idx})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction (Step {step_idx+1})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(tgt_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Ground Truth (Step {step_idx+1})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"step_{step_idx+1:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg["inference"]["ckpt_path"]
    model, model_cfg = load_model(ckpt_path, device)
    mean, std = load_stats(cfg["data"]["stats_dir"], device)

    cond_steps = cfg["inference"]["condition_steps"]
    pred_steps = cfg["inference"]["autoregressive_steps"]
    user_dt = cfg["inference"]["dt"]
    dt_ref = model_cfg["dt_ref"]

    dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=[cfg["data"]["test_year"], cfg["data"]["test_year"]],
        window_size=cfg["data"]["window_size"],
        sample_k=cond_steps + pred_steps + 1,
        look_ahead=0,
        is_train=False,
        dt_ref=dt_ref,
        sampling_mode="sequential",
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data, dt_data = next(iter(loader))

    x_all = data.to(device).float()
    x_ctx = x_all[:, :cond_steps]
    x_tgt = x_all[:, cond_steps:]

    dt_ctx = user_dt
    
    dt_list = [
        torch.tensor(user_dt, device=device, dtype=torch.float32)
        for _ in range(pred_steps)
    ]

    trained_ensemble_size = model_cfg["ensemble_size"]
    requested_ensemble_size = cfg["inference"]["ensemble_size"]

    print(f"Inference Config:")
    print(f"  Condition Steps: {cond_steps}")
    print(f"  Forecast Steps: {pred_steps}")
    print(f"  DT (Time Step): {user_dt} hours")
    print(f"  Trained Ensemble Size: {trained_ensemble_size}")
    print(f"  Requested Ensemble Size: {requested_ensemble_size}")

    if requested_ensemble_size > trained_ensemble_size:
        print(f"Warning: Requested size {requested_ensemble_size} exceeds "
              f"trained size {trained_ensemble_size}. Clamping to trained size.")
        ensemble_size = trained_ensemble_size
    else:
        ensemble_size = requested_ensemble_size

    with torch.no_grad():
        member_preds_list = []
        for m in range(ensemble_size):
            preds = model.forward_rollout(
                x_ctx, dt_ctx, dt_list
            )

            if preds.is_complex():
                preds = preds.real
            member_preds_list.append(preds)

        member_stack = torch.stack(member_preds_list, dim=0)
        pred_mean = member_stack.mean(dim=0)

        for step in range(pred_steps):
            if step >= x_tgt.shape[1]:
                break

            pred_step = pred_mean[:, step]
            target_step = x_tgt[:, step]

            pred_phys = denormalize(pred_step, mean, std)
            target_phys = denormalize(target_step, mean, std)
            input_phys = denormalize(x_ctx[:, -1], mean, std)

            rmse = torch.sqrt(((pred_phys - target_phys) ** 2).mean())
            print(
                f"Step {step+1} (+{(step+1)*user_dt}h): "
                f"RMSE = {rmse.item():.4f}"
            )

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

