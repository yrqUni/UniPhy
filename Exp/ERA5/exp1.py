import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

import warnings
warnings.filterwarnings("ignore")


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


def compute_batch_rmse(pred, target):
    mse = (pred - target) ** 2
    return mse.mean(dim=(0, 2, 3, 4))


def run_comparison():
    console = Console()
    with open("infer.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = cfg["inference"]["ckpt_path"]
    model, model_cfg = load_model(ckpt_path, device)

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

    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dt_list = [
        torch.tensor(user_dt, device=device, dtype=torch.float32)
        for _ in range(pred_steps)
    ]
    dt_ctx = user_dt

    mse_accum_uniphy = torch.zeros(pred_steps, device=device)
    mse_accum_pers = torch.zeros(pred_steps, device=device)
    mse_accum_clim = torch.zeros(pred_steps, device=device)
    sample_count = 0

    trained_ensemble_size = model_cfg["ensemble_size"]
    requested_ensemble_size = cfg["inference"]["ensemble_size"]
    if requested_ensemble_size > trained_ensemble_size:
        ensemble_size = trained_ensemble_size
    else:
        ensemble_size = requested_ensemble_size

    console.print(f"[bold green]Starting Comparison Experiment...[/bold green]")
    console.print(f"Model: UniPhy (Ensemble={ensemble_size}) vs Persistence vs Climatology")
    console.print(f"Steps: {pred_steps} x {user_dt}h")

    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device).float()
            
            x_ctx = data[:, :cond_steps]
            x_tgt = data[:, cond_steps:cond_steps+pred_steps]
            
            B, T_tgt, C, H, W = x_tgt.shape
            
            x_last = x_ctx[:, -1]
            pers_pred = x_last.unsqueeze(1).expand(-1, T_tgt, -1, -1, -1)
            
            clim_pred = torch.zeros_like(x_tgt)

            member_preds_list = []
            for m in range(ensemble_size):
                preds = model.forward_rollout(
                    x_ctx, dt_ctx, dt_list
                )
                if preds.is_complex():
                    preds = preds.real
                member_preds_list.append(preds)
            
            uniphy_pred = torch.stack(member_preds_list, dim=0).mean(dim=0)

            uniphy_pred = uniphy_pred[:, :T_tgt]
            pers_pred = pers_pred[:, :T_tgt]
            clim_pred = clim_pred[:, :T_tgt]

            mse_accum_uniphy += compute_batch_rmse(uniphy_pred, x_tgt) * B
            mse_accum_pers += compute_batch_rmse(pers_pred, x_tgt) * B
            mse_accum_clim += compute_batch_rmse(clim_pred, x_tgt) * B
            
            sample_count += B

            if i % 10 == 0:
                print(f"Processed batch {i}...")

    rmse_uniphy = torch.sqrt(mse_accum_uniphy / sample_count).cpu().numpy()
    rmse_pers = torch.sqrt(mse_accum_pers / sample_count).cpu().numpy()
    rmse_clim = torch.sqrt(mse_accum_clim / sample_count).cpu().numpy()

    table = Table(title="UniPhy Performance Benchmark (Normalized RMSE)")

    table.add_column("Step", justify="right", style="cyan", no_wrap=True)
    table.add_column("Time", justify="right", style="magenta")
    table.add_column("UniPhy", justify="right", style="green")
    table.add_column("Persistence", justify="right", style="yellow")
    table.add_column("Climatology", justify="right", style="red")
    table.add_column("Skill (vs Pers)", justify="right", style="bold white")

    for t in range(pred_steps):
        time_str = f"+{(t+1)*user_dt:.0f}h"
        
        u_val = rmse_uniphy[t]
        p_val = rmse_pers[t]
        c_val = rmse_clim[t]
        
        skill = (p_val - u_val) / p_val * 100
        skill_str = f"{skill:+.1f}%"
        skill_style = "green" if skill > 0 else "red"

        table.add_row(
            f"{t+1}",
            time_str,
            f"{u_val:.4f}",
            f"{p_val:.4f}",
            f"{c_val:.4f}",
            f"[{skill_style}]{skill_str}[/{skill_style}]"
        )

    console.print(table)


if __name__ == "__main__":
    run_comparison()

