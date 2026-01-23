import argparse
import datetime
import logging
import os
import random
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

warnings.filterwarnings("ignore")

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
    mse = (error ** 2)
    if lat_weight is not None:
        mse = mse * lat_weight
        rmse = torch.sqrt(mse.mean(dim=(0, 2, 3))).mean().item()
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/nfs/ERA5_data/data_norm")
    parser.add_argument("--save_dir", type=str, default="./uniphy/results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--year_range", type=int, nargs='+', default=[2021, 2021])
    parser.add_argument("--cond_frames", type=int, default=2)
    parser.add_argument("--pred_frames", type=int, default=8)
    parser.add_argument("--dt_ref", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_ensemble", type=int, default=1)
    parser.add_argument("--eval_sample_num", type=int, default=100)
    args = parser.parse_args()

    set_random_seed(1017)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model_args = checkpoint["model_args"]

    model = UniPhyModel(
        in_channels=model_args["in_channels"],
        out_channels=model_args["out_channels"],
        embed_dim=model_args["embed_dim"],
        expand=model_args["expand"],
        num_experts=model_args.get("num_experts", 4),
        depth=model_args["depth"],
        patch_size=model_args["patch_size"],
        img_height=model_args["img_height"],
        img_width=model_args["img_width"]
    ).to(device)

    sd = checkpoint["model"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    total_frames = args.cond_frames + args.pred_frames
    test_ds = ERA5_Dataset(
            input_dir=args.data_root,
            year_range=args.year_range,
            window_size=total_frames,
            sample_k=total_frames,
            look_ahead=2,
            is_train=False,
            dt_ref=args.dt_ref
            )
    
    if args.eval_sample_num > 0 and args.eval_sample_num < len(test_ds):
        total_len = len(test_ds)
        indices = np.linspace(0, total_len - 1, args.eval_sample_num, dtype=int)
        test_ds = Subset(test_ds, indices)
        print(f"Subsampled dataset to {len(test_ds)} samples (from {total_len}).")

    test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
            )

    lat_weights = get_lat_weights(model_args["img_height"], model_args["img_width"], device)
    all_rmses = []

    print(f"Start Inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for i, (data, dt) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(device).float()
            
            x_cond = data[:, :args.cond_frames]
            x_target = data[:, args.cond_frames:]
            
            dt_cond = torch.ones(x_cond.shape[0], args.cond_frames, device=device) * args.dt_ref
            dt_future = torch.ones(x_cond.shape[0], args.pred_frames, device=device) * args.dt_ref
            
            if args.num_ensemble > 1:
                ensemble_preds = []
                for _ in range(args.num_ensemble):
                    pred = model.forecast(x_cond, dt_cond, args.pred_frames, dt_future)
                    ensemble_preds.append(pred)
                pred_final = torch.stack(ensemble_preds, dim=0).mean(dim=0)
            else:
                pred_final = model.forecast(x_cond, dt_cond, args.pred_frames, dt_future)
            
            rmse = compute_metrics(pred_final, x_target, lat_weights)
            all_rmses.append(rmse)
            
            if i % 10 == 0:
                print(f"Step {i}: RMSE = {rmse:.4f}")

    mean_rmse = np.mean(all_rmses)
    print(f"\nFinal Test RMSE: {mean_rmse:.4f}")
    
    result_path = os.path.join(args.save_dir, "metric_summary.txt")
    with open(result_path, "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Year Range: {args.year_range}\n")
        f.write(f"Mean RMSE: {mean_rmse:.4f}\n")

if __name__ == "__main__":
    main()
    