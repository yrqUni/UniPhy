import os
import hashlib
import shutil
import sys
import numpy as np
import torch
from ERA5 import ERA5_Dataset


def get_shm_path(input_dir):
    input_hash = hashlib.md5(str(input_dir).encode("utf-8")).hexdigest()
    return os.path.join("/dev/shm", "era5_cache", input_hash)


def check_nfs_source(input_dir, year_range):
    print(f"Checking NFS source: {input_dir}")
    total_files = 0
    total_frames = 0
    missing_years = []

    for year in range(int(year_range[0]), int(year_range[1]) + 1):
        y_dir = os.path.join(input_dir, str(year))
        if not os.path.isdir(y_dir):
            missing_years.append(year)
            continue

        months = sorted([f for f in os.listdir(y_dir) if f.endswith(".npy")])
        for m in months:
            file_path = os.path.join(y_dir, m)
            try:
                arr = np.load(file_path, mmap_mode="r")
                total_frames += arr.shape[0]
                total_files += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"NFS Status: {total_files} files found, {total_frames} total frames.")
    if missing_years:
        print(f"Missing years: {missing_years}")
    return total_frames


def check_era5_dataset_logic(input_dir, year_range):
    print(f"Testing ERA5_Dataset class from ERA5.py")
    try:
        ds = ERA5_Dataset(
            input_dir=input_dir,
            year_range=year_range,
            window_size=16,
            sample_k=4,
            look_ahead=2
        )
        
        ds_len = len(ds)
        print(f"Dataset length: {ds_len}")
        print(f"Total internal frames: {ds.total_frames}")
        print(f"File offsets: {ds.file_frame_offsets}")
        
        if ds_len > 0:
            print("Testing __getitem__ (this may trigger SHM caching)...")
            data, dt = ds[0]
            print(f"Sample data shape: {data.shape}")
            print(f"Sample dt_step: {dt}")
            print("Dataset __getitem__ test passed.")
        else:
            print("Warning: Dataset is empty.")
            
    except Exception as e:
        print(f"Dataset Logic Error: {e}")
        sys.exit(1)


def check_shm_status(input_dir):
    shm_path = get_shm_path(input_dir)
    print(f"Checking SHM cache: {shm_path}")

    if not os.path.exists(shm_path):
        print("SHM cache directory does not exist.")
        return

    files = [
        os.path.join(shm_path, f) 
        for f in os.listdir(shm_path) 
        if f.endswith(".npy")
    ]
    total_size = sum(os.path.getsize(f) for f in files)
    
    print(f"SHM Status: {len(files)} files cached.")
    print(f"Total SHM usage: {total_size / (1024**3):.2f} GB")

    try:
        total, used, free = shutil.disk_usage("/dev/shm")
        print(f"/dev/shm System Total: {total / (1024**3):.2f} GB")
        print(f"/dev/shm System Free: {free / (1024**3):.2f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    DATA_DIR = "/nfs/ERA5_data/data_norm"
    YEARS = (2010, 2020)

    print("=" * 50)
    check_nfs_source(DATA_DIR, YEARS)
    print("-" * 50)
    check_era5_dataset_logic(DATA_DIR, YEARS)
    print("-" * 50)
    check_shm_status(DATA_DIR)
    print("=" * 50)
