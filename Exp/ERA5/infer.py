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
    model = UniPhyModel(**cfg["model"]).to(device)
    if os.path.exists(cfg["inference"]["ckpt"]):
        ckpt = torch.load(cfg["inference"]["ckpt"], map_location=device)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()})
    model.eval()
    return model

def run_inference(cfg, model, device, console):
    i_len, i_dt, f_hz = cfg["inference"]["input_len"], cfg["inference"]["input_dt"], cfg["inference"]["forecast_horizon"]
    t_frames = i_len + int(f_hz / 6.0)
    dataset = ERA5_Dataset(input_dir=cfg["data"]["input_dir"], year_range=cfg["data"]["test_year_range"], window_size=t_frames, sample_k=1, look_ahead=0, is_train=False, dt_ref=cfg["data"]["dt_ref"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["inference"]["batch_size"], shuffle=False, num_workers=4)
    
    strategies = cfg["inference"]["strategies"]
    results = {s["name"]: [] for s in strategies}
    max_s, count = cfg["inference"].get("max_samples"), 0

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating Strategies...", total=min(len(loader), (max_s // cfg["inference"]["batch_size"] + 1) if max_s else len(loader)))
        for batch in loader:
            if max_s and count >= max_s: break
            data = batch[0].to(device).float()
            x_ctx, x_tgt = data[:, :i_len], data[:, -1]
            count += data.shape[0]
            for s in strategies:
                dt_s = s["dt"]
                steps = int(f_hz / dt_s)
                dt_l = [torch.tensor(dt_s, device=device).float()] * steps
                preds = model.forward_rollout(x_ctx, i_dt, dt_l)
                results[s["name"]].extend(torch.sqrt((preds[:, -1].real - x_tgt).pow(2).mean(dim=(1, 2, 3))).cpu().tolist())
            progress.advance(task)

    console.print("\n" + "="*50 + "\nForecast Results (Horizon: 12h)\n" + "="*50)
    for s in strategies:
        res = results[s["name"]]
        if len(res) > 0:
            console.print(f"{s['name']}: RMSE {np.mean(res):.4f} +/- {np.std(res):.4f}")
        else:
            console.print(f"{s['name']}: No results")

if __name__ == "__main__":
    with open("infer.yaml", "r") as f: cfg = yaml.safe_load(f)
    run_inference(cfg, load_model(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.device("cuda" if torch.cuda.is_available() else "cpu"), Console())
    