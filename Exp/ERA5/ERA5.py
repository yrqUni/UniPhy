import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mmap
import shutil
from collections import OrderedDict
import torch.distributed as dist
import time
import fcntl

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir, year_range, is_train=True, sample_len=8, cache_size_gb=700):
        self.input_root = input_dir
        self.sample_len = sample_len
        self.shm_root = "/dev/shm/era5_cache"
        self.cache_limit = (cache_size_gb * 1024**3) // (15 * 1024**2 * 1024)
        
        self.all_info = []
        for year in range(year_range[0], year_range[1] + 1):
            y_dir = os.path.join(input_dir, str(year))
            if os.path.isdir(y_dir):
                months = sorted([f for f in os.listdir(y_dir) if f.endswith('.npy')])
                for m in months:
                    self.all_info.append({
                        "nfs_path": os.path.join(y_dir, m),
                        "shm_path": os.path.join(self.shm_root, str(year), m),
                        "year": str(year),
                        "filename": m
                    })

        self.file_shapes = []
        self.file_frame_offsets = [0]
        for info in self.all_info:
            data = np.load(info["nfs_path"], mmap_mode='r')
            self.file_shapes.append(data.shape)
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + data.shape[0])
        
        self.total_frames = self.file_frame_offsets[-1]
        self._mmap_handle_cache = OrderedDict()
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if not os.path.exists(self.shm_root):
                os.makedirs(self.shm_root, exist_ok=True)

    def __len__(self):
        return self.total_frames - self.sample_len + 1

    def _ensure_in_shm(self, file_idx):
        info = self.all_info[file_idx]
        if os.path.exists(info["shm_path"]):
            return info["shm_path"]

        lock_path = info["shm_path"] + ".lock"
        os.makedirs(os.path.dirname(info["shm_path"]), exist_ok=True)
        
        with open(lock_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if not os.path.exists(info["shm_path"]):
                self._manage_shm_space()
                shutil.copy(info["nfs_path"], info["shm_path"])
            fcntl.flock(f, fcntl.LOCK_UN)
        return info["shm_path"]

    def _manage_shm_space(self):
        shm_files = []
        for root, _, files in os.walk(self.shm_root):
            for f in files:
                if f.endswith(".npy"):
                    p = os.path.join(root, f)
                    shm_files.append((p, os.path.getatime(p)))
        
        if len(shm_files) >= self.cache_limit:
            shm_files.sort(key=lambda x: x[1])
            num_to_delete = len(shm_files) - self.cache_limit + 1
            for i in range(num_to_delete):
                try:
                    os.remove(shm_files[i][0])
                    if os.path.exists(shm_files[i][0] + ".lock"):
                        os.remove(shm_files[i][0] + ".lock")
                except:
                    pass

    def _get_mmap(self, file_idx):
        shm_path = self._ensure_in_shm(file_idx)
        if file_idx in self._mmap_handle_cache:
            self._mmap_handle_cache.move_to_end(file_idx)
            return self._mmap_handle_cache[file_idx]
        
        data = np.load(shm_path, mmap_mode='r')
        self._mmap_handle_cache[file_idx] = data
        if len(self._mmap_handle_cache) > 32:
            self._mmap_handle_cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_frame_offsets, idx, side='right') - 1
        f_start = self.file_frame_offsets[file_idx]
        f_len = self.file_shapes[file_idx][0]
        off = idx - f_start
        
        if off + self.sample_len <= f_len:
            res = self._get_mmap(file_idx)[off : off + self.sample_len]
        else:
            frames = []
            needed = self.sample_len
            curr_ptr, curr_f = idx, file_idx
            while needed > 0:
                s, l = self.file_frame_offsets[curr_f], self.file_shapes[curr_f][0]
                o = curr_ptr - s
                take = min(l - o, needed)
                frames.append(self._get_mmap(curr_f)[o : o + take])
                needed -= take
                curr_ptr += take
                curr_f += 1
            res = np.concatenate(frames, axis=0)
            
        return torch.from_numpy(res.copy())
    