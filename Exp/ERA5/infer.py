import argparse
import os
import sys
import yaml
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
plt.switch_backend('Agg')

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

def save_visualization(target, pred, save_dir, index):
    os.makedirs(save_dir, exist_ok=True)
    
    if target.is_complex(): target = target.real
    if pred.is_complex(): pred = pred.real

    target = target.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, axes = plt.subplots(2, target.shape[0], figsize=(target.shape[0] * 4, 8))
    
    for t in range(target.shape[0]):
        im1 = axes[0, t].imshow(target[t, 0], cmap='RdBu_r')
        axes[0, t].set_title(f"Target T={t}")
        axes[0, t].axis('off')
        
        im2 = axes[1, t].imshow(pred[t, 0], cmap='RdBu_r')
        axes[1, t].set_title(f"Pred T={t}")
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{index}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="infer.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["inference"]["device"])
    
    model = UniPhyModel(**cfg["model"]).to(device)
    
    ckpt_path = cfg["inference"]["ckpt_path"]
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    total_frames = cfg["inference"]["cond_frames"] + cfg["inference"]["pred_frames"]
    
    test_dataset = ERA5_Dataset(
        input_dir=cfg["inference"]["data_root"],
        year_range=cfg["inference"]["year_range"],
        window_size=total_frames,
        sample_k=total_frames,
        look_ahead=0,
        is_train=False,
        dt_ref=cfg["inference"]["dt_ref"]
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    save_dir = cfg["inference"]["save_dir"]
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    lat_weights = get_lat_weights(
        cfg["model"]["img_height"],
        cfg["model"]["img_width"],
        device
    )

    all_rmses = []
    all_rmses_per_step = [[] for _ in range(cfg["inference"]["pred_frames"])]
    num_vis_saved = 0
    max_vis = cfg["inference"]["max_vis_samples"]
    
    cond_frames = cfg["inference"]["cond_frames"]
    pred_frames = cfg["inference"]["pred_frames"]
    dt_ref = cfg["inference"]["dt_ref"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Inference...", total=len(dataloader))

        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= cfg["inference"]["eval_sample_num"]:
                    break

                data = data.to(device)
                
                x_init = data[:, cond_frames - 1]
                x_target = data[:, cond_frames : cond_frames + pred_frames]

                dt_list = [dt_ref] * pred_frames
                
                pred_final = model.forward_rollout(x_init, dt_list)

                for t in range(min(pred_frames, pred_final.shape[1])):
                    rmse_t = compute_metrics(
                        pred_final[:, t:t+1], 
                        x_target[:, t:t+1], 
                        lat_weights
                    )
                    all_rmses_per_step[t].append(rmse_t)

                rmse = compute_metrics(pred_final, x_target, lat_weights)
                all_rmses.append(rmse)

                if num_vis_saved < max_vis:
                    save_visualization(x_target[0], pred_final[0], save_dir, i)
                    num_vis_saved += 1

                progress.advance(task)

    mean_rmse = np.mean(all_rmses)
    std_rmse = np.std(all_rmses)

    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    print(f"Overall RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
    print("\nRMSE per forecast step:")
    for t in range(pred_frames):
        if all_rmses_per_step[t]:
            step_rmse = np.mean(all_rmses_per_step[t])
            hours = (t + 1) * dt_ref
            print(f"  Step {t+1} ({hours:.0f}h): {step_rmse:.4f}")

    result_path = os.path.join(save_dir, "metric_summary.txt")
    with open(result_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Year Range: {cfg['inference']['year_range']}\n")
        f.write(f"Overall RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}\n")
        f.write("\nRMSE per step:\n")
        for t in range(pred_frames):
            if all_rmses_per_step[t]:
                step_rmse = np.mean(all_rmses_per_step[t])
                hours = (t + 1) * dt_ref
                f.write(f"  Step {t+1} ({hours:.0f}h): {step_rmse:.4f}\n")

if __name__ == "__main__":
    main()
    