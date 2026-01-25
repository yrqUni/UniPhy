import argparse
import os
import sys
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, H, 1)


def compute_metrics(pred, target, lat_weights):
    if pred.is_complex():
        pred = pred.real
    if target.is_complex():
        target = target.real

    pred = pred.to(target.device)

    min_t = min(pred.shape[1], target.shape[1])
    pred = pred[:, :min_t]
    target = target[:, :min_t]

    error = pred - target
    mse = error ** 2

    if lat_weights is not None:
        if lat_weights.device != pred.device:
            lat_weights = lat_weights.to(pred.device)
        mse = mse * lat_weights

    rmse = torch.sqrt(mse.mean()).item()
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if "inference" in cfg:
        cfg = cfg["inference"]

    device = torch.device(cfg.get("device", "cuda"))

    if cfg.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg["save_dir"], exist_ok=True)

    checkpoint = torch.load(cfg["ckpt_path"], map_location="cpu")

    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
    else:
        model_args = {
            "in_channels": 30,
            "out_channels": 30,
            "embed_dim": 512,
            "expand": 4,
            "num_experts": 8,
            "depth": 8,
            "patch_size": 32,
            "img_height": 721,
            "img_width": 1440,
            "dt_ref": cfg.get("dt_ref", 6.0),
            "sde_mode": cfg.get("sde_mode", "sde"),
            "init_noise_scale": cfg.get("noise_scale", 0.01),
            "max_growth_rate": 0.3,
        }

    model = UniPhyModel(**model_args).to(device)

    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from {cfg['ckpt_path']}")
    print(f"Model parameters: {total_params:,}")

    total_frames = cfg["cond_frames"] + cfg["pred_frames"]

    test_ds = ERA5_Dataset(
        input_dir=cfg["data_root"],
        year_range=cfg["year_range"],
        window_size=total_frames,
        sample_k=total_frames,
        look_ahead=2,
        is_train=False,
        dt_ref=cfg["dt_ref"],
    )

    if cfg["eval_sample_num"] > 0 and cfg["eval_sample_num"] < len(test_ds):
        total_len = len(test_ds)
        indices = np.linspace(0, total_len - 1, cfg["eval_sample_num"], dtype=int)
        test_ds = Subset(test_ds, indices)
        print(f"Subsampled dataset to {len(test_ds)} samples (from {total_len}).")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    lat_weights = get_lat_weights(
        model_args["img_height"], model_args["img_width"], device
    )

    all_rmses = []
    all_rmses_per_step = [[] for _ in range(cfg["pred_frames"])]

    print(f"Starting inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for i, (data, dt_data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(device).float()

            x_cond = data[:, :cfg["cond_frames"]]
            x_target = data[:, cfg["cond_frames"]:]

            dt_cond = torch.ones(x_cond.shape[0], cfg["cond_frames"], device=device) * cfg["dt_ref"]
            dt_future = torch.ones(x_cond.shape[0], cfg["pred_frames"], device=device) * cfg["dt_ref"]

            if cfg.get("num_ensemble", 1) > 1:
                ensemble_preds = []
                for _ in range(cfg["num_ensemble"]):
                    pred = model.forecast(x_cond, dt_cond, cfg["pred_frames"], dt_future)
                    if pred.is_complex():
                        pred = pred.real
                    ensemble_preds.append(pred)
                pred_final = torch.stack(ensemble_preds, dim=0).mean(dim=0)
            else:
                pred_final = model.forecast(x_cond, dt_cond, cfg["pred_frames"], dt_future)
                if pred_final.is_complex():
                    pred_final = pred_final.real

            pred_final = pred_final.to(device)

            for t in range(min(cfg["pred_frames"], x_target.shape[1], pred_final.shape[1])):
                rmse_t = compute_metrics(
                    pred_final[:, t:t+1], x_target[:, t:t+1], lat_weights
                )
                all_rmses_per_step[t].append(rmse_t)

            rmse = compute_metrics(pred_final, x_target, lat_weights)
            all_rmses.append(rmse)

            if i % 10 == 0:
                print(f"Batch {i}: RMSE = {rmse:.4f}")

    mean_rmse = np.mean(all_rmses)
    std_rmse = np.std(all_rmses)

    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    print(f"Overall RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
    print("\nRMSE per forecast step:")
    for t in range(cfg["pred_frames"]):
        if all_rmses_per_step[t]:
            step_rmse = np.mean(all_rmses_per_step[t])
            print(f"  Step {t+1} ({(t+1)*cfg['dt_ref']:.0f}h): {step_rmse:.4f}")

    result_path = os.path.join(cfg["save_dir"], "metric_summary.txt")
    with open(result_path, "w") as f:
        f.write(f"Checkpoint: {cfg['ckpt_path']}\n")
        f.write(f"Year Range: {cfg['year_range']}\n")
        f.write(f"Mean RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}\n")
        f.write("\nRMSE per step:\n")
        for t in range(cfg["pred_frames"]):
            if all_rmses_per_step[t]:
                f.write(f"  Step {t+1}: {np.mean(all_rmses_per_step[t]):.4f}\n")

    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
    