import os
import shutil
import random
import hashlib
import fcntl
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset


class ERA5_Dataset(Dataset):
    def __init__(
        self,
        input_dir,
        year_range,
        window_size=16,
        sample_k=4,
        look_ahead=2,
        is_train=True,
        frame_hours=6.0,
        sampling_mode="mixed",
        max_shm_gb=16,
    ):
        self.input_root = str(input_dir)
        self.window_size = int(window_size)
        self.sample_k = int(sample_k)
        self.look_ahead = int(look_ahead)
        self.is_train = bool(is_train)
        self.frame_hours = float(frame_hours)
        self.sampling_mode = "mixed" if sampling_mode in {"mix", "mixed"} else str(sampling_mode)

        input_hash = hashlib.md5(self.input_root.encode("utf-8")).hexdigest()
        self.shm_root = os.path.join("/dev/shm", "era5_cache", input_hash)
        os.makedirs(self.shm_root, exist_ok=True)

        self.all_info = []
        for year in range(int(year_range[0]), int(year_range[1]) + 1):
            y_dir = os.path.join(self.input_root, str(year))
            if not os.path.isdir(y_dir):
                continue
            months = sorted([f for f in os.listdir(y_dir) if f.endswith(".npy")])
            for m in months:
                full_path = os.path.join(y_dir, m)
                file_hash = hashlib.md5(full_path.encode("utf-8")).hexdigest()
                self.all_info.append({
                    "nfs_path": full_path,
                    "shm_path": os.path.join(self.shm_root, f"{file_hash}.npy"),
                    "id": f"{year}_{m}",
                })

        self.file_frame_offsets = [0]
        for info in self.all_info:
            arr = np.load(info["nfs_path"], mmap_mode="r")
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + int(arr.shape[0]))
        self.total_frames = int(self.file_frame_offsets[-1])

        self._mmap_cache = OrderedDict()
        self._max_mmap_handles = 8
        self.max_shm_bytes = max_shm_gb * 1024 * 1024 * 1024

    def __len__(self):
        n = self.total_frames - self.window_size - self.look_ahead
        return max(0, int(n))

    def _locate_file_and_index(self, global_idx):
        lo = 0
        hi = len(self.file_frame_offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.file_frame_offsets[mid + 1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        return int(lo), int(global_idx - self.file_frame_offsets[lo])

    def _get_current_cache_size(self):
        files = [
            os.path.join(self.shm_root, f) 
            for f in os.listdir(self.shm_root) 
            if f.endswith(".npy")
        ]
        return sum(os.path.getsize(f) for f in files)

    def _manage_shm_space(self, required_bytes):
        try:
            _, _, free_phys = shutil.disk_usage("/dev/shm")
            current_cache_usage = self._get_current_cache_size()

            while (current_cache_usage + required_bytes > self.max_shm_bytes) or \
                  (free_phys < required_bytes + 1024**3):
                files = [
                    os.path.join(self.shm_root, f)
                    for f in os.listdir(self.shm_root)
                    if f.endswith(".npy")
                ]
                if not files:
                    break
                files.sort(key=lambda x: os.path.getatime(x))
                target = files[0]
                try:
                    os.remove(target)
                    lock_f = target + ".lock"
                    if os.path.exists(lock_f):
                        os.remove(lock_f)
                except OSError:
                    pass
                current_cache_usage = self._get_current_cache_size()
                _, _, free_phys = shutil.disk_usage("/dev/shm")
        except Exception:
            pass

    def _ensure_file_in_shm(self, info):
        shm_path = info["shm_path"]
        if os.path.exists(shm_path):
            try:
                os.utime(shm_path, None)
            except OSError:
                pass
            return shm_path

        nfs_path = info["nfs_path"]
        lock_path = shm_path + ".lock"

        with open(lock_path, "w") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if os.path.exists(shm_path):
                    return shm_path
                
                file_size = os.path.getsize(nfs_path)
                self._manage_shm_space(file_size)
                
                temp_path = f"{shm_path}.{os.getpid()}.{random.randint(0, 1000)}.tmp"
                shutil.copy2(nfs_path, temp_path)
                os.rename(temp_path, shm_path)
                return shm_path
            except (OSError, IOError):
                return nfs_path
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _get_array_mmap(self, file_idx):
        info = self.all_info[file_idx]
        if info["id"] in self._mmap_cache:
            self._mmap_cache.move_to_end(info["id"])
            return self._mmap_cache[info["id"]]

        path_to_use = self._ensure_file_in_shm(info)
        try:
            arr = np.load(path_to_use, mmap_mode="r")
        except Exception:
            arr = np.load(info["nfs_path"], mmap_mode="r")

        self._mmap_cache[info["id"]] = arr
        while len(self._mmap_cache) > self._max_mmap_handles:
            self._mmap_cache.popitem(last=False)
        return arr

    def _sample_offsets(self):
        if not self.is_train:
            return list(range(self.sample_k))
        
        if self.sampling_mode == "sequential":
            start = random.randint(0, self.window_size - self.sample_k)
            return list(range(start, start + self.sample_k))
        
        if self.sampling_mode == "mixed" and random.random() < 0.5:
            start = random.randint(0, self.window_size - self.sample_k)
            return list(range(start, start + self.sample_k))
            
        return sorted(random.sample(range(self.window_size), self.sample_k))

    def __getitem__(self, idx):
        base = int(idx)
        offsets = self._sample_offsets()
        
        frames = []
        for off in offsets:
            f_idx, l_idx = self._locate_file_and_index(base + off)
            arr = self._get_array_mmap(f_idx)
            frames.append(torch.from_numpy(arr[l_idx].copy()))

        data = torch.stack(frames, dim=0)
        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        dt_step = (offsets_t[1:] - offsets_t[:-1]) * self.frame_hours
        dt_padded = torch.cat([dt_step, dt_step[-1:]], dim=0)
        
        return data, dt_padded
    