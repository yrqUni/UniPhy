import os
import hashlib
import shutil
import numpy as np


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


def check_shm_status(input_dir):
    shm_path = get_shm_path(input_dir)
    print(f"Checking SHM cache: {shm_path}")

    if not os.path.exists(shm_path):
        print("SHM cache directory does not exist.")
        return

    files = [os.path.join(shm_path, f) for f in os.listdir(shm_path) if f.endswith(".npy")]
    total_size = sum(os.path.getsize(f) for f in files)
    
    print(f"SHM Status: {len(files)} files cached.")
    print(f"Total SHM usage: {total_size / (1024**3):.2f} GB")

    try:
        total, used, free = shutil.disk_usage("/dev/shm")
        print(f"/dev/shm System Free: {free / (1024**3):.2f} GB")
    except Exception:
        pass


def clean_shm_cache(input_dir):
    shm_path = get_shm_path(input_dir)
    if os.path.exists(shm_path):
        print(f"Cleaning cache: {shm_path}")
        shutil.rmtree(shm_path)
        print("Cache cleared.")
    else:
        print("No cache found to clean.")


if __name__ == "__main__":
    DATA_DIR = "/nfs/ERA5_data/data_norm"
    YEARS = (2010, 2020)

    check_nfs_source(DATA_DIR, YEARS)
    print("-" * 30)
    check_shm_status(DATA_DIR)
    clean_shm_cache(DATA_DIR)
