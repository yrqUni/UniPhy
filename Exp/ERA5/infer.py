import os
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ERA5 import ERA5_Dataset
from ModelUniPhy import UniPhyModel


def setup_model(ckpt_path, device, console):
    if not os.path.exists(ckpt_path):
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "cfg" not in ckpt or "model" not in ckpt["cfg"]:
        console.print("[bold red]Error:[/bold red] Missing model configuration in ckpt.")
        sys.exit(1)
    model_cfg = ckpt["cfg"]["model"]
    model = UniPhyModel(**model_cfg).to(device)
    state_dict = ckpt["model"]
    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.eval()
    return model


def get_dataloader(cfg, console):
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
    if len(dataset) == 0:
        console.print("[bold red]Error:[/bold red] No samples found.")
        sys.exit(1)
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )


def save_visualization(tensor, output_dir, sample_idx, step_idx):
    os.makedirs(output_dir, exist_ok=True)
    img = tensor.cpu().numpy()
    if img.ndim == 3:
        img = img[0]
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='RdBu_r')
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_step_{step_idx:02d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def print_metrics(results, console):
    table = Table(title="Inference Metrics (RMSE @ Final Step)", show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Mean RMSE", style="green")
    table.add_column("Std Dev", style="magenta")
    table.add_column("Samples", style="white")

    for strat_name, errors in results.items():
        if errors:
            mean_rmse = np.mean(errors)
            std_rmse = np.std(errors)
            table.add_row(
                strat_name,
                f"{mean_rmse:.4f}",
                f"{std_rmse:.4f}",
                str(len(errors))
            )
        else:
            table.add_row(strat_name, "N/A", "N/A", "0")
    
    console.print(Panel(table, expand=False))


def run_inference(cfg, model, device, console):
    loader = get_dataloader(cfg, console)
    strategies = cfg["inference"]["strategies"]
    input_len = cfg["inference"]["input_len"]
    input_dt = cfg["inference"]["input_dt"]
    horizon = cfg["inference"]["forecast_horizon"]
    output_base_dir = cfg["inference"]["output_dir"]
    ensemble_mode = cfg["inference"].get("ensemble_mode", False)
    ensemble_size = cfg["inference"].get("ensemble_size", 10) if ensemble_mode else 1
    max_samples = cfg["inference"].get("max_samples", None)
    
    results = {s["name"]: [] for s in strategies}
    processed_count = 0
    total_to_process = min(len(loader), max_samples) if max_samples else len(loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Inference", total=total_to_process)

        for batch_idx, batch in enumerate(loader):
            if max_samples and processed_count >= max_samples:
                break
            
            data = batch[0].to(device).float()
            x_context = data[:, :input_len]
            x_target = data[:, -1]
            
            for s in strategies:
                dt_val = s["dt"]
                strat_name_clean = s["name"].replace(" ", "_").replace("=", "_")
                num_steps = int(horizon / dt_val)
                dt_list = [torch.tensor(dt_val, device=device).float()] * num_steps
                
                strat_preds = []
                for m in range(ensemble_size):
                    torch.manual_seed(42 + batch_idx * 100 + m)
                    with torch.no_grad():
                        preds = model.forward_rollout(x_context, input_dt, dt_list)
                        strat_preds.append(preds)
                    
                    member_dir = os.path.join(output_base_dir, strat_name_clean, f"member_{m:02d}")
                    save_visualization(preds[0, -1], member_dir, processed_count, num_steps)
                
                ensemble_tensor = torch.stack(strat_preds, dim=0)
                mean_pred = torch.mean(ensemble_tensor, dim=0)
                
                mean_dir = os.path.join(output_base_dir, strat_name_clean, "mean")
                save_visualization(mean_pred[0, -1], mean_dir, processed_count, num_steps)
                
                final_rmse = torch.sqrt((mean_pred[0, -1] - x_target).pow(2).mean()).item()
                results[s["name"]].append(final_rmse)

            processed_count += 1
            progress.advance(task)

    return results


def main():
    config_path = "infer.yaml"
    if "--config" in sys.argv:
        try:
            config_path = sys.argv[sys.argv.index("--config") + 1]
        except IndexError:
            pass

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()
    
    console.print(f"[bold blue]UniPhy Inference[/bold blue]")
    ckpt_path = cfg["inference"]["ckpt"]
    model = setup_model(ckpt_path, device, console)
    
    metrics = run_inference(cfg, model, device, console)
    print_metrics(metrics, console)


if __name__ == "__main__":
    main()
    