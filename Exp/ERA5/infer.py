import os
import sys
import yaml
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

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
        batch_size=cfg["inference"]["batch_size"], 
        shuffle=False, 
        num_workers=4
    )

def execute_inference(cfg, model, device, console):
    loader = get_dataloader(cfg)
    strategies = cfg["inference"]["strategies"]
    results = {s["name"]: [] for s in strategies}
    
    input_len = cfg["inference"]["input_len"]
    input_dt = cfg["inference"]["input_dt"]
    horizon = cfg["inference"]["forecast_horizon"]
    max_samples = cfg["inference"].get("max_samples")
    
    processed_count = 0
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )

    with progress:
        task_id = progress.add_task("Evaluating Rollout", total=len(loader))
        
        for batch in loader:
            if max_samples and processed_count >= max_samples:
                break
                
            data = batch[0].to(device).float()
            x_context = data[:, :input_len]
            x_target = data[:, -1]
            
            for s in strategies:
                dt_val = s["dt"]
                num_steps = int(horizon / dt_val)
                dt_list = [torch.tensor(dt_val, device=device).float()] * num_steps
                
                preds = model.forward_rollout(x_context, input_dt, dt_list)
                
                rmse = torch.sqrt((preds[:, -1].real - x_target).pow(2).mean(dim=(1, 2, 3)))
                results[s["name"]].extend(rmse.cpu().tolist())
            
            processed_count += data.shape[0]
            progress.advance(task_id)

    return results

def display_results(results, console):
    table = Table(title="UniPhy Forecast Performance (12h Horizon)", show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="dim", width=25)
    table.add_column("Mean RMSE", justify="right", style="green")
    table.add_column("Std Dev", justify="right", style="cyan")
    table.add_column("Samples", justify="right")

    for name, values in results.items():
        if values:
            table.add_row(
                name, 
                f"{np.mean(values):.4f}", 
                f"{np.std(values):.4f}", 
                str(len(values))
            )
    
    console.print(Panel(table, expand=False, border_style="blue"))

def main():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("infer.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    console.print(f"[bold blue]Initializing UniPhy Inference on {device}...[/bold blue]")
    
    model = setup_model(cfg, device)
    raw_results = execute_inference(cfg, model, device, console)
    display_results(raw_results, console)

if __name__ == "__main__":
    main()
    