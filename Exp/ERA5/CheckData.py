import os
import time
from torch.utils.data import DataLoader
from ERA5 import ERA5_Dataset
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

def get_shm_status(shm_root):
    if not os.path.exists(shm_root):
        return 0, []
    files = []
    total_size = 0
    for root, _, filenames in os.walk(shm_root):
        for f in filenames:
            if f.endswith(".npy"):
                fp = os.path.join(root, f)
                size = os.path.getsize(fp) / (1024**3)
                total_size += size
                files.append(f)
    return total_size, sorted(files)

def run_test():
    os.environ["LOCAL_RANK"] = "0"
    input_dir = "/nfs/ERA5_data/data_monthly"
    year_range = [2000, 2001]
    batch_size = 2
    num_workers = 4
    test_steps = 100

    dataset = ERA5_Dataset(
        input_dir=input_dir, 
        year_range=year_range, 
        sample_len=8, 
        look_ahead=2
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )

    loader_iter = iter(loader)
    latencies = []
    
    table = Table(title="Prefetching Real-time Monitor")
    table.add_column("Step")
    table.add_column("Latency (ms)")
    table.add_column("SHM Size (GB)")
    table.add_column("Cached Months")

    with Live(table, refresh_per_second=4):
        for i in range(test_steps):
            t0 = time.perf_counter()
            try:
                _ = next(loader_iter)
            except StopIteration:
                break
            t1 = time.perf_counter()
            
            latency = (t1 - t0) * 1000
            latencies.append(latency)
            
            shm_size, cached_files = get_shm_status(dataset.shm_root)
            
            table.add_row(
                str(i),
                f"{latency:.2f}",
                f"{shm_size:.1f}",
                ", ".join(cached_files[-3:]) if cached_files else "Waiting..."
            )
            
            time.sleep(0.5)

    avg_speed = batch_size / (sum(latencies[10:]) / len(latencies[10:]) / 1000)
    console.print(f"\n[bold green]Test Completed.[/bold green]")
    console.print(f"Stable Throughput: [cyan]{avg_speed:.2f} samples/s[/cyan]")
    console.print(f"Final SHM Usage: [cyan]{get_shm_status(dataset.shm_root)[0]:.1f} GB[/cyan]")

if __name__ == "__main__":
    run_test()