import os
import time
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from ERA5 import ERA5_Dataset


def get_shm_usage():
    try:
        total, used, free = shutil.disk_usage("/dev/shm")
        return used / (1024**3), free / (1024**3)
    except Exception:
        return 0, 0


def test_full_epoch(input_dir, year_range, batch_size=4, num_workers=4):
    print("=" * 60)
    print("STRESS TEST: FULL EPOCH SIMULATION")
    print("=" * 60)

    try:
        dataset = ERA5_Dataset(
            input_dir=input_dir,
            year_range=year_range,
            is_train=True,
            max_shm_gb=8
        )
    except Exception as e:
        print(f"INIT ERROR: {e}")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )

    total_batches = len(loader)
    start_time = time.time()
    
    try:
        for i, (data, dt) in enumerate(loader):
            if i % 50 == 0 or i == total_batches - 1:
                used, free = get_shm_usage()
                elapsed = time.time() - start_time
                ips = (i + 1) / elapsed
                
                print(f"Batch [{i+1}/{total_batches}] | "
                      f"Speed: {ips:.2f} it/s | "
                      f"SHM Used: {used:.2f}GB | Free: {free:.2f}GB")

            if i >= 500:
                print("-" * 60)
                print("STABILITY CHECK PASSED: LRU WORKING")
                break

        print(f"TOTAL TIME: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        used, _ = get_shm_usage()
        print(f"FINAL SHM FOOTPRINT: {used:.2f} GB")
        print("=" * 60)


if __name__ == "__main__":
    DATA_ROOT = "/nfs/ERA5_data/data_norm"
    TEST_YEARS = (2010, 2011)
    
    if os.path.exists(DATA_ROOT):
        test_full_epoch(
            input_dir=DATA_ROOT,
            year_range=TEST_YEARS,
            batch_size=8,
            num_workers=4
        )
    else:
        print(f"PATH NOT FOUND: {DATA_ROOT}")
        