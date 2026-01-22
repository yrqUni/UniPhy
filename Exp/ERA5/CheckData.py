import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from ERA5 import ERA5_Dataset
from rich.console import Console
from rich.table import Table

console = Console()

def test_speed(num_workers):
    input_dir = '/nfs/ERA5_data/data_norm'
    batch_size = 2
    steps = 40

    dataset = ERA5_Dataset(input_dir=input_dir, year_range=[2000, 2000], sample_len=8)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, persistent_workers=True
    )

    latencies = []
    it = iter(loader)
    next(it) 
    
    for _ in range(steps):
        t0 = time.perf_counter()
        next(it)
        latencies.append(time.perf_counter() - t0)
    
    avg_lat = np.mean(latencies)
    fps = batch_size / avg_lat
    return avg_lat, fps

if __name__ == "__main__":
    table = Table(title="Dataloader Optimization Test", header_style="bold magenta")
    table.add_column("Workers", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Throughput (samples/s)", justify="right")

    for w in [2, 4, 8]:
        lat, fps = test_speed(w)
        table.add_row(str(w), f"{lat*1000:.1f}", f"{fps:.2f}")
    
    console.print(table)

