import os
import sys
import torch
import yaml
import numpy as np
from rich.console import Console
from rich.progress import Progress

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

def load_model(cfg, device):
    model = UniPhyModel(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        expand=cfg["model"]["expand"],
        num_experts=cfg["model"]["num_experts"],
        depth=cfg["model"]["depth"],
        patch_size=cfg["model"]["patch_size"],
        img_height=cfg["model"]["img_height"],
        img_width=cfg["model"]["img_width"],
        dt_ref=cfg["model"]["dt_ref"],
        sde_mode=cfg["model"]["sde_mode"],
        init_noise_scale=cfg["model"]["init_noise_scale"],
        max_growth_rate=cfg["model"]["max_growth_rate"],
    ).to(device)

    ckpt_path = cfg["inference"]["ckpt"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        print("Warning: No checkpoint loaded.")
    
    model.eval()
    return model

def infer_rollout(cfg, model, device, console):
    test_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["test_year_range"],
        window_size=cfg["data"]["input_steps"],
        sample_k=cfg["data"]["input_steps"],
        look_ahead=cfg["data"]["pred_steps"],
        is_train=False,
        dt_ref=cfg["data"]["dt_ref"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    results = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("Rollout Inference...", total=len(test_loader))
        
        for batch in test_loader:
            data, dt_data = batch
            data = data.to(device).float()
            dt_data = dt_data.to(device).float()

            x_init = data[:, 0]
            dt_list = []
            
            for i in range(cfg["data"]["pred_steps"]):
                dt_list.append(dt_data[:, i])

            with torch.no_grad():
                preds = model.forward_rollout(x_init, dt_list)
            
            targets = data[:, 1:]
            
            error = (preds.real - targets).pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
            results.append(error.cpu())
            
            progress.advance(task)

    overall_rmse = torch.stack(results).mean(dim=0)
    
    console.print("\n" + "="*60)
    console.print("Rollout Inference Results")
    console.print("="*60)
    
    for i, rmse in enumerate(overall_rmse):
        hours = (i + 1) * 6
        console.print(f"Step {i+1} ({hours}h): RMSE {rmse:.4f}")
    
    console.print("="*60)

def infer_dt_comparison(cfg, model, device, console):
    test_dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["test_year_range"],
        window_size=1,
        sample_k=1,
        look_ahead=1, 
        is_train=False,
        dt_ref=cfg["data"]["dt_ref"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    strategies = [
        {"name": "dt=1.0 x 6", "dt": 1.0, "steps": 6},
        {"name": "dt=2.0 x 3", "dt": 2.0, "steps": 3},
        {"name": "dt=3.0 x 2", "dt": 3.0, "steps": 2},
        {"name": "dt=6.0 x 1", "dt": 6.0, "steps": 1},
    ]

    rmse_stats = {s["name"]: [] for s in strategies}

    with Progress(console=console) as progress:
        task = progress.add_task("DT Comparison...", total=len(test_loader))

        for batch in test_loader:
            data, _ = batch
            data = data.to(device).float()
            
            x_init = data[:, 0]
            x_target = data[:, 1] 

            for s in strategies:
                dt_val = s["dt"]
                steps = s["steps"]
                dt_list = [torch.tensor(dt_val, device=device)] * steps

                with torch.no_grad():
                    preds = model.forward_rollout(x_init, dt_list)
                
                final_pred = preds[:, -1].real
                mse = (final_pred - x_target).pow(2).mean()
                rmse = torch.sqrt(mse)
                
                rmse_stats[s["name"]].append(rmse.item())

            progress.advance(task)

    console.print("\n" + "="*60)
    console.print("DT Comparison Results (Target: +6h)")
    console.print("="*60)

    for s in strategies:
        name = s["name"]
        avg_rmse = np.mean(rmse_stats[name])
        std_rmse = np.std(rmse_stats[name])
        console.print(f"Strategy [{name}]: RMSE {avg_rmse:.4f} +/- {std_rmse:.4f}")
    
    console.print("="*60)

def main():
    with open("infer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()
    
    model = load_model(cfg, device)

    if cfg["inference"].get("dt_comparison", False):
        infer_dt_comparison(cfg, model, device, console)
    else:
        infer_rollout(cfg, model, device, console)

if __name__ == "__main__":
    main()
    