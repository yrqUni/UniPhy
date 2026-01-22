import numpy as np
import os
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir='/nfs/ERA5_data/data_norm', 
                 year_range=[2000, 2021], 
                 is_train=True, 
                 sample_len=16, 
                 eval_sample=400, 
                 **kwargs):
        
        self.input_dir = input_dir
        self.sample_len = sample_len
        self.file_list = []
        years = sorted(list(range(year_range[0], year_range[1] + 1)))
        
        for year in years:
            year_dir = os.path.join(input_dir, str(year))
            if not os.path.exists(year_dir): continue
            files = sorted([
                os.path.join(year_dir, f) 
                for f in os.listdir(year_dir) 
                if f.endswith('.npy')
            ])
            self.file_list.extend(files)

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npy files found in {input_dir}")

        if is_train:
            self.file_list = self.file_list[:-eval_sample]
        else:
            self.file_list = self.file_list[-eval_sample:]
            
        try:
            tmp = np.load(self.file_list[0], mmap_mode='r')
            self.T_per_file, self.C, self.H, self.W = tmp.shape
        except Exception as e:
            raise RuntimeError(f"Failed to read {self.file_list[0]}: {e}")

        self.total_frames = len(self.file_list) * self.T_per_file
        self.total_samples = max(0, self.total_frames - self.sample_len + 1)
        self.file_cache = OrderedDict()
        self.cache_limit = 512 

    def _get_mmap(self, file_idx):
        if file_idx in self.file_cache:
            self.file_cache.move_to_end(file_idx)
            return self.file_cache[file_idx]
        
        mmap_obj = np.load(self.file_list[file_idx], mmap_mode='r')
        self.file_cache[file_idx] = mmap_obj
        if len(self.file_cache) > self.cache_limit:
            self.file_cache.popitem(last=False)
        return mmap_obj

    def __len__(self):
        return self.total_samples

    def __getitem__(self, global_idx):
        start_file_idx = global_idx // self.T_per_file
        start_frame_offset = global_idx % self.T_per_file
        out = np.empty((self.sample_len, self.C, self.H, self.W), dtype=np.float32)
        
        frames_read = 0
        curr_file_idx = start_file_idx
        curr_offset = start_frame_offset
        
        while frames_read < self.sample_len:
            if curr_file_idx >= len(self.file_list): break
            
            mmap_obj = self._get_mmap(curr_file_idx)
            frames_available = self.T_per_file - curr_offset
            frames_to_take = min(frames_available, self.sample_len - frames_read)
            
            out[frames_read : frames_read + frames_to_take] = \
                mmap_obj[curr_offset : curr_offset + frames_to_take]
            
            frames_read += frames_to_take
            curr_file_idx += 1
            curr_offset = 0
            
        return out

