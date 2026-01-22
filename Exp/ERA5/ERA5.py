import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mmap
from collections import OrderedDict

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir, year_range, is_train=True, sample_len=8, eval_sample=1):
        self.input_dir = input_dir
        self.sample_len = sample_len
        self.all_files = []
        
        for year in range(year_range[0], year_range[1] + 1):
            year_dir = os.path.join(input_dir, str(year))
            if os.path.isdir(year_dir):
                months = sorted([f for f in os.listdir(year_dir) if f.endswith('.npy')])
                self.all_files.extend([os.path.join(year_dir, m) for m in months])
        
        self.file_shapes = []
        self.file_frame_offsets = [0]
        
        for path in self.all_files:
            data = np.load(path, mmap_mode='r')
            self.file_shapes.append(data.shape)
            self.file_frame_offsets.append(self.file_frame_offsets[-1] + data.shape[0])
            
            try:
                fd = os.open(path, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                os.close(fd)
            except:
                pass
        
        self.total_frames = self.file_frame_offsets[-1]
        self._mmap_cache = OrderedDict()
        self.cache_limit = 64

    def __len__(self):
        return self.total_frames - self.sample_len + 1

    def _get_mmap(self, file_idx):
        if file_idx in self._mmap_cache:
            self._mmap_cache.move_to_end(file_idx)
            return self._mmap_cache[file_idx]
        
        data = np.load(self.all_files[file_idx], mmap_mode='r')
        self._mmap_cache[file_idx] = data
        if len(self._mmap_cache) > self.cache_limit:
            self._mmap_cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_frame_offsets, idx, side='right') - 1
        
        f_start = self.file_frame_offsets[file_idx]
        f_len = self.file_shapes[file_idx][0]
        offset_in_file = idx - f_start
        
        if offset_in_file + self.sample_len <= f_len:
            res = self._get_mmap(file_idx)[offset_in_file : offset_in_file + self.sample_len]
        else:
            frames = []
            needed = self.sample_len
            curr_ptr = idx
            curr_f_idx = file_idx
            
            while needed > 0:
                s = self.file_frame_offsets[curr_f_idx]
                l = self.file_shapes[curr_f_idx][0]
                off = curr_ptr - s
                take = min(l - off, needed)
                
                frames.append(self._get_mmap(curr_f_idx)[off : off + take])
                needed -= take
                curr_ptr += take
                curr_f_idx += 1
            res = np.concatenate(frames, axis=0)
            
        return torch.from_numpy(res.copy())
    