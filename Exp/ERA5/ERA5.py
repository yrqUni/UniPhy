import os
import numpy as np
import torch
from torch.utils.data import Dataset
import shutil
from collections import OrderedDict
import threading
import time
import random

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir, year_range, window_size=16, sample_k=4, look_ahead=2, is_train=True, dt_ref=6.0):
        self.input_root = input_dir
        self.window_size = window_size
        self.sample_k = sample_k
        self.look_ahead = look_ahead
        self.is_train = is_train
        self.dt_ref = dt_ref
        self.shm_root = "/dev/shm/era5_cache"
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        self.all_info = []
        for year in range(year_range[0], year_range[1] + 1):
            y_dir = os.path.join(input_dir, str(year))
            if os.path.isdir(y_dir):
                months = sorted([f for f in os.listdir(y_dir) if f.endswith('.npy')])
                for m in months:
                    self.all_info.append({
                        "nfs_path": os.path.join(y_dir, m),
                        "shm_path": os.path.join(self.shm_root, str(year), m),
                        "id": f"{year}_{m}"
                    })

        self.file_frame_offsets = [0]
        self.file_shapes = []
        for info in self.all_info:
            data = np.load(info["nfs_path"], mmap_mode='r')
            self.file_shapes.append(data.shape)
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + data.shape[0])
        
        self.total_frames = self.file_frame_offsets[-1]
        self._mmap_cache = OrderedDict()
        self.current_file_idx = 0
        
        if self.local_rank == 0:
            if not os.path.exists(self.shm_root):
                os.makedirs(self.shm_root, exist_ok=True)
            threading.Thread(target=self._prefetch_worker, daemon=True).start()

    def _prefetch_worker(self):
        while True:
            for i in range(self.current_file_idx, min(self.current_file_idx + self.look_ahead + 1, len(self.all_info))):
                info = self.all_info[i]
                if not os.path.exists(info["shm_path"]):
                    os.makedirs(os.path.dirname(info["shm_path"]), exist_ok=True)
                    tmp_shm = info["shm_path"] + ".tmp"
                    try:
                        shutil.copy(info["nfs_path"], tmp_shm)
                        os.rename(tmp_shm, info["shm_path"])
                    except:
                        pass
            
            existing_files = []
            for root, _, files in os.walk(self.shm_root):
                for f in files:
                    if f.endswith(".npy"):
                        existing_files.append(os.path.join(root, f))
            
            if len(existing_files) > 40:
                existing_files.sort(key=os.path.getatime)
                for f in existing_files[:-30]:
                    try: os.remove(f)
                    except: pass
            
            time.sleep(5)

    def __len__(self):
        return max(0, self.total_frames - self.window_size + 1)

    def _get_data_ptr(self, file_idx):
        info = self.all_info[file_idx]
        self.current_file_idx = file_idx
        
        target_path = info["shm_path"] if os.path.exists(info["shm_path"]) else info["nfs_path"]
        
        if file_idx in self._mmap_cache:
            self._mmap_cache.move_to_end(file_idx)
            return self._mmap_cache[file_idx]
        
        data = np.load(target_path, mmap_mode='r')
        self._mmap_cache[file_idx] = data
        if len(self._mmap_cache) > 16:
            self._mmap_cache.popitem(last=False)
        return data

    def _get_single_frame(self, global_idx):
        f_idx = np.searchsorted(self.file_frame_offsets, global_idx, side='right') - 1
        off = global_idx - self.file_frame_offsets[f_idx]
        return self._get_data_ptr(f_idx)[off]

    def __getitem__(self, idx):
        if self.is_train:
            offsets = sorted(random.sample(range(self.window_size), self.sample_k))
        else:
            offsets = list(range(self.sample_k))
            
        frames = []
        start_global = idx + offsets[0]
        f_idx = np.searchsorted(self.file_frame_offsets, start_global, side='right') - 1
        
        current_file_data = self._get_data_ptr(f_idx)
        file_start_global = self.file_frame_offsets[f_idx]
        file_len = self.file_shapes[f_idx][0]
        
        for off in offsets:
            curr_global = idx + off
            local_off = curr_global - file_start_global
            
            if 0 <= local_off < file_len:
                frames.append(torch.from_numpy(current_file_data[local_off].copy()))
            else:
                frame_data = self._get_single_frame(curr_global)
                frames.append(torch.from_numpy(frame_data.copy()))
        
        data = torch.stack(frames, dim=0)
        
        offsets_tensor = torch.tensor(offsets, dtype=torch.float32)
        dt = (offsets_tensor[1:] - offsets_tensor[:-1]) * self.dt_ref
        
        return data, dt
    