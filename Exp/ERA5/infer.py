import argparse
import os
import random
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


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(pred, target, lat_weight):
    if lat_weight is not None:
        if lat_weight.device != pred.device:
            lat_weight = lat_weight.to(pred.device)
    error = pred - target
    mse = error**2
    if lat_weight is not None:
        mse = mse * lat_weight
        rmse = torch.sqrt(mse.mean()).item()
    else:
        rmse = torch.sqrt(mse.mean()).item()
    return rmse


def get_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, H, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args_cli = parser.parse_args()
    with open(args_cli.config, "r") as f:
        cfg = yaml.safe_load(f)["inference"]
    if cfg["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_random_seed(1017)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"])
    checkpoint = torch.load(cfg["ckpt_path"], map_location="cpu")
    model_args = checkpoint["model_args"]
    if "sde_mode" in cfg:
        model_args["sde_mode"] = cfg["sde_mode"]
    if "noise_scale" in cfg:
        model_args["noise_scale"] = cfg["noise_scale"]
    model = UniPhyModel(**model_args).to(device)
    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    print(f"Loaded model from {cfg['ckpt_path']}")
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
    print(f"Start Inference on {len(test_loader)} batches...")
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(device).float()
            x_cond = data[:, : cfg["cond_frames"]]
            x_target = data[:, cfg["cond_frames"] :]
            dt_cond = (
                torch.ones(x_cond.shape[0], cfg["cond_frames"], device=device)
                * cfg["dt_ref"]
            )
            dt_future = (
                torch.ones(x_cond.shape[0], cfg["pred_frames"], device=device)
                * cfg["dt_ref"]
            )
            if cfg["num_ensemble"] > 1:
                ensemble_preds = []
                for _ in range(cfg["num_ensemble"]):
                    pred = model.forecast(
                        x_cond, dt_cond, cfg["pred_frames"], dt_future
                    )
                    ensemble_preds.append(pred)
                pred_final = torch.stack(ensemble_preds, dim=0).mean(dim=0)
            else:
                pred_final = model.forecast(
                    x_cond, dt_cond, cfg["pred_frames"], dt_future
                )
            rmse = compute_metrics(pred_final, x_target, lat_weights)
            all_rmses.append(rmse)
            if i % 10 == 0:
                print(f"Step {i}: RMSE = {rmse:.4f}")
    mean_rmse = np.mean(all_rmses)
    print(f"\nFinal Test RMSE: {mean_rmse:.4f}")
    result_path = os.path.join(cfg["save_dir"], "metric_summary.txt")
    with open(result_path, "w") as f:
        f.write(f"Checkpoint: {cfg['ckpt_path']}\n")
        f.write(f"Year Range: {cfg['year_range']}\n")
        f.write(f"Mean RMSE: {mean_rmse:.4f}\n")


if __name__ == "__main__":
    main()
    