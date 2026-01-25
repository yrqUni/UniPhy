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
    if pred.is_complex():
        pred = pred.real
    if target.is_complex():
        target = target.real

    error = pred - target
    mse = error ** 2

    if lat_weight is not None:
        if lat_weight.device != pred.device:
            lat_weight = lat_weight.to(pred.device)
        mse = mse * lat_weight

    rmse = torch.sqrt(mse.mean()).item()
    return rmse


def get_lat_weights(H, W, device):
    lat = torch.linspace(-90, 90, H, device=device)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    return weights.view(1, 1, 1, H, 1)


def load_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
    else:
        model_args = None

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    return model_args, state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    model_cfg = full_cfg.get("model", {})
    cfg = full_cfg.get("inference", {})

    if cfg.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_random_seed(1017)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(cfg.get("device", "cuda"))

    model_args, state_dict = load_checkpoint(cfg["ckpt_path"], device)

    if model_args is None:
        model_args = {
            "in_channels": model_cfg.get("in_channels", 30),
            "out_channels": model_cfg.get("out_channels", 30),
            "embed_dim": model_cfg.get("embed_dim", 512),
            "expand": model_cfg.get("expand", 4),
            "num_experts": model_cfg.get("num_experts", 8),
            "depth": model_cfg.get("depth", 8),
            "patch_size": model_cfg.get("patch_size", 32),
            "img_height": model_cfg.get("img_height", 721),
            "img_width": model_cfg.get("img_width", 1440),
            "dt_ref": model_cfg.get("dt_ref", 6.0),
            "sde_mode": model_cfg.get("sde_mode", "sde"),
            "init_noise_scale": model_cfg.get("init_noise_scale", 1.0),
            "max_growth_rate": model_cfg.get("max_growth_rate", 0.3),
        }

    model = UniPhyModel(**model_args).to(device)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"Model loaded from {cfg['ckpt_path']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_dataset = ERA5_Dataset(
        input_dir=cfg["data_root"],
        year_range=cfg["year_range"],
        window_size=cfg["cond_frames"] + cfg["pred_frames"],
        sample_k=cfg["cond_frames"] + cfg["pred_frames"],
        look_ahead=cfg["pred_frames"],
        is_train=False,
        dt_ref=cfg["dt_ref"],
    )

    eval_sample_num = min(cfg.get("eval_sample_num", 100), len(test_dataset))
    indices = list(range(0, len(test_dataset), max(1, len(test_dataset) // eval_sample_num)))[:eval_sample_num]
    test_subset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
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
        for i, (data, dt) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(device)
            dt = dt.to(device)

            x_cond = data[:, :cfg["cond_frames"]]
            x_target = data[:, cfg["cond_frames"]:]

            x_cond_complex = torch.complex(x_cond, torch.zeros_like(x_cond))

            dt_cond = dt[:, :cfg["cond_frames"]] if dt.ndim > 1 else dt[:cfg["cond_frames"]]
            if dt_cond.ndim == 1:
                dt_cond = dt_cond.unsqueeze(0).expand(x_cond.shape[0], -1)

            dt_future = torch.ones(
                x_cond.shape[0], cfg["pred_frames"], device=device
            ) * cfg["dt_ref"]

            if cfg.get("num_ensemble", 1) > 1:
                ensemble_preds = []
                for _ in range(cfg["num_ensemble"]):
                    pred = model.forecast(
                        x_cond_complex, dt_cond, cfg["pred_frames"], dt_future
                    )
                    if pred.is_complex():
                        pred = pred.real
                    ensemble_preds.append(pred)
                pred_final = torch.stack(ensemble_preds, dim=0).mean(dim=0)
            else:
                pred_final = model.forecast(
                    x_cond_complex, dt_cond, cfg["pred_frames"], dt_future
                )
                if pred_final.is_complex():
                    pred_final = pred_final.real

            pred_final = pred_final.to(device)

            for t in range(min(cfg["pred_frames"], x_target.shape[1], pred_final.shape[1])):
                rmse_t = compute_metrics(pred_final[:, t], x_target[:, t], lat_weights)
                all_rmses_per_step[t].append(rmse_t)

            rmse = compute_metrics(pred_final, x_target[:, :pred_final.shape[1]], lat_weights)
            all_rmses.append(rmse)

            if i % 10 == 0:
                print(f"Batch {i}: RMSE = {rmse:.4f}")

    mean_rmse = np.mean(all_rmses)
    std_rmse = np.std(all_rmses)

    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    print(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print("\nRMSE per forecast step:")
    for t in range(cfg["pred_frames"]):
        if all_rmses_per_step[t]:
            step_rmse = np.mean(all_rmses_per_step[t])
            print(f"  Step {t+1} ({(t+1)*cfg['dt_ref']:.0f}h): {step_rmse:.4f}")

    results = {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "rmse_per_step": [np.mean(s) if s else 0 for s in all_rmses_per_step],
        "config": cfg,
    }

    results_path = os.path.join(cfg["save_dir"], "inference_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
    