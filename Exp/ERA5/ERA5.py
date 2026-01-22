import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mmap
import shutil
from collections import OrderedDict
import torch.distributed as dist
import threading
import time

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir, year_range, is_train=True, sample_len=8, look_ahead=2):
        self.input_root = input_dir
        self.sample_len = sample_len
        self.shm_root = "/dev/shm/era5_cache"
        self.look_ahead = look_ahead
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
        return self.total_frames - self.sample_len + 1

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

    def __getitem__(self, idx):
        f_idx = np.searchsorted(self.file_frame_offsets, idx, side='right') - 1
        off = idx - self.file_frame_offsets[f_idx]
        
        if off + self.sample_len <= self.file_shapes[f_idx][0]:
            res = self._get_data_ptr(f_idx)[off : off + self.sample_len]
        else:
            frames = []
            needed = self.sample_len
            c_ptr, c_f = idx, f_idx
            while needed > 0:
                s, l = self.file_frame_offsets[c_f], self.file_shapes[c_f][0]
                o = c_ptr - s
                take = min(l - o, needed)
                frames.append(self._get_data_ptr(c_f)[o : o + take])
                needed -= take
                c_ptr += take
                c_f += 1
            res = np.concatenate(frames, axis=0)
            
        return torch.from_numpy(res.copy())
    