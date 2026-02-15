import os
import random
import shutil
import threading
import uuid
from collections import OrderedDict

import numpy as np
import torch
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
    ):
        self.input_root = str(input_dir)
        self.window_size = int(window_size)
        self.sample_k = int(sample_k)
        self.look_ahead = int(look_ahead)
        self.is_train = bool(is_train)
        self.frame_hours = float(frame_hours)
        self.sampling_mode = "mixed" if sampling_mode in {"mix", "mixed"} else str(sampling_mode)

        self.shm_root = f"/dev/shm/era5_cache/{uuid.uuid4().hex}"
        self.all_info = []
        for year in range(int(year_range[0]), int(year_range[1]) + 1):
            y_dir = os.path.join(self.input_root, str(year))
            if not os.path.isdir(y_dir):
                continue
            months = sorted([f for f in os.listdir(y_dir) if f.endswith(".npy")])
            for m in months:
                self.all_info.append(
                    {
                        "nfs_path": os.path.join(y_dir, m),
                        "shm_path": os.path.join(self.shm_root, str(year), m),
                        "id": f"{year}_{m}",
                    }
                )

        self.file_frame_offsets = [0]
        for info in self.all_info:
            arr = np.load(info["nfs_path"], mmap_mode="r")
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + int(arr.shape[0]))
        self.total_frames = int(self.file_frame_offsets[-1])

        self.cache_lock = threading.Lock()
        self.local_cache = OrderedDict()
        self.max_cache_files = 8

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
        file_idx = int(lo)
        local_idx = int(global_idx - self.file_frame_offsets[file_idx])
        return file_idx, local_idx

    def _ensure_file_in_shm(self, info):
        shm_path = info["shm_path"]
        os.makedirs(os.path.dirname(shm_path), exist_ok=True)
        if not os.path.exists(shm_path):
            shutil.copy2(info["nfs_path"], shm_path)
        return shm_path

    def _get_array_mmap(self, file_idx):
        info = self.all_info[file_idx]
        with self.cache_lock:
            arr = self.local_cache.get(info["id"])
            if arr is not None:
                self.local_cache.move_to_end(info["id"])
                return arr
        shm_path = self._ensure_file_in_shm(info)
        arr = np.load(shm_path, mmap_mode="r")
        with self.cache_lock:
            self.local_cache[info["id"]] = arr
            while len(self.local_cache) > self.max_cache_files:
                self.local_cache.popitem(last=False)
        return arr

    def _sample_offsets(self):
        if self.sampling_mode == "sequential":
            return list(range(self.sample_k))
        return sorted(random.sample(range(self.window_size), self.sample_k))

    def __getitem__(self, idx):
        base = int(idx)
        offsets = self._sample_offsets()
        frame_indices = [base + o for o in offsets]

        frames = []
        for gidx in frame_indices:
            file_idx, local_idx = self._locate_file_and_index(gidx)
            arr = self._get_array_mmap(file_idx)
            frames.append(torch.from_numpy(arr[local_idx].copy()))
        data = torch.stack(frames, dim=0)

        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        dt_step = (offsets_t[1:] - offsets_t[:-1]) * self.frame_hours
        return data, dt_step

    def cleanup(self):
        shutil.rmtree(self.shm_root, ignore_errors=True)

    def __del__(self):
        self.cleanup()
