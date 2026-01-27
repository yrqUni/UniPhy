import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel

def setup_model(cfg, device):
    model = UniPhyModel(**cfg["model"]).to(device)
    ckpt_path = cfg["inference"]["ckpt"]
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        clean_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        model.load_state_dict(clean_dict)
    model.eval()
    return model

def get_dataloader(cfg):
    i_len = cfg["inference"]["input_len"]
    f_hz = cfg["inference"]["forecast_horizon"]
    total_frames = i_len + int(f_hz / 6.0)
    dataset = ERA5_Dataset(
        input_dir=cfg["data"]["input_dir"],
        year_range=cfg["data"]["test_year_range"],
        window_size=total_frames,
        sample_k=1,
        look_ahead=0,
        is_train=False,
        dt_ref=cfg["data"]["dt_ref"]
    )
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4
    )

def save_visualization(tensor, output_dir, sample_idx, step_idx):
    os.makedirs(output_dir, exist_ok=True)
    img = tensor.cpu().numpy()
    if img.ndim == 3: img = img[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='RdBu_r')
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_step_{step_idx:02d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def run_inference(cfg, model, device, console):
    loader = get_dataloader(cfg)
    strategies = cfg["inference"]["strategies"]
    
    input_len = cfg["inference"]["input_len"]
    input_dt = cfg["inference"]["input_dt"]
    horizon = cfg["inference"]["forecast_horizon"]
    output_base_dir = cfg["inference"]["output_dir"]
    
    ensemble_mode = cfg["inference"].get("ensemble_mode", False)
    ensemble_size = cfg["inference"].get("ensemble_size", 10) if ensemble_mode else 1
    max_samples = cfg["inference"].get("max_samples", None)
    
    processed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Multi-Strategy Ensemble", total=min(len(loader), max_samples) if max_samples else len(loader))

        for batch_idx, batch in enumerate(loader):
            if max_samples and processed_count >= max_samples:
                break
            
            data = batch[0].to(device).float()
            x_context = data[:, :input_len]
            
            for s in strategies:
                dt_val = s["dt"]
                strat_name = s["name"].replace(" ", "_").replace("=", "_").replace("(", "").replace(")", "")
                num_steps = int(horizon / dt_val)
                dt_list = [torch.tensor(dt_val, device=device).float()] * num_steps
                
                strat_preds = []
                
                for m in range(ensemble_size):
                    seed = 42 + batch_idx * 100 + m
                    torch.manual_seed(seed)
                    
                    with torch.no_grad():
                        preds = model.forward_rollout(x_context, input_dt, dt_list)
                        strat_preds.append(preds)
                    
                    member_dir = os.path.join(output_base_dir, strat_name, f"member_{m:02d}")
                    for t in range(num_steps):
                        save_visualization(preds[0, t], member_dir, processed_count, t + 1)
                
                ensemble_tensor = torch.stack(strat_preds, dim=0)
                mean_pred = torch.mean(ensemble_tensor, dim=0)
                
                mean_dir = os.path.join(output_base_dir, strat_name, "mean")
                for t in range(num_steps):
                    save_visualization(mean_pred[0, t], mean_dir, processed_count, t + 1)

            processed_count += 1
            progress.advance(task)

def main():
    with open("infer.yaml", "r") as f: cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()
    
    console.print(f"[bold blue]UniPhy Inference: {len(cfg['inference']['strategies'])} Strategies, {cfg['inference']['ensemble_size']} Members[/bold blue]")
    model = setup_model(cfg, device)
    run_inference(cfg, model, device, console)

if __name__ == "__main__":
    main()
    