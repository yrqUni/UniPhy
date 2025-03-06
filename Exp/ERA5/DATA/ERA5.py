
from joblib import dump, load
from collections import OrderedDict
import numpy as np
import os
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import math
import time

'''all keys:
['TCWV', 'U10', 'V10', 'T2', 'MSLP', 'SP', 
'VV50', 'U50', 'V50', 'RH50', 'T50', 
'VV100', 'U100', 'V100', 'RH100', 'T100', 
'VV150', 'U150', 'V150', 'RH150', 'T150', 
'VV200', 'U200', 'V200', 'RH200', 'T200', 
'VV250', 'U250', 'V250', 'RH250', 'T250', 
'VV300', 'U300', 'V300', 'RH300', 'T300', 
'VV400', 'U400', 'V400', 'RH400', 'T400', 
'VV500', 'U500', 'V500', 'RH500', 'T500', 
'VV600', 'U600', 'V600', 'RH600', 'T600', 
'VV700', 'U700', 'V700', 'RH700', 'T700', 
'VV850', 'U850', 'V850', 'RH850', 'T850', 
'VV925', 'U925', 'V925', 'RH925', 'T925', 
'VV1000', 'U1000', 'V1000', 'RH1000', 'T1000', 
'TIME1', 'TIME2', 'POS1', 'POS2', 'POS3']
'''

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir='/work2/esm10/ERA5_data/data_norm' , year_range=[2000,2021],is_train=True,sample_len=16,eval_sample=400,max_cache_size=2,rank=0,gpus=8):
        self.file_paths = []
        self.sample_len = sample_len
        self.rank = rank
        self.gpus = gpus
        self.local_file_paths = []
        self.is_train = is_train
        years = list(range(year_range[0],year_range[1]+1))
        for year in years: 
            subdir_name = f"{year}"
            subdir_path = os.path.join(input_dir, subdir_name)
            if os.path.exists(subdir_path):
                
                for file_name in os.listdir(subdir_path):
                    
                    if os.path.isfile(os.path.join(subdir_path, file_name)) and file_name.isdigit() and len(file_name) == 8:
                        file_path = os.path.join(subdir_path, file_name)
                        self.file_paths.append(file_path)
        self.file_paths = sorted(self.file_paths)
        # ================
        if self.is_train:
            self.file_paths = self.file_paths[:-eval_sample]
        else:
            self.file_paths = self.file_paths[-eval_sample:]
        additional_files_num = math.ceil((self.sample_len - 1) / 4)
        part_size = self.calculate_max_segment_size(len(self.file_paths),self.gpus,additional_files_num)
        self.local_file_paths = self.file_paths[self.rank * part_size:(self.rank + 1) * part_size + additional_files_num]
        self.local_sample_num = (len(self.local_file_paths) * 4 - self.sample_len + 1)
        # ================
        self.var = ['VV', 'U', 'V', 'RH', 'T', 'Z']
        self.heights = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925','1000']
        self.time_pos_var_list = ['TIME1', 'TIME2', 'POS1', 'POS2', 'POS3']
        self.surface_var_list = ['TCWV', 'U10', 'V10', 'T2', 'MSLP', 'SP']
        self.sample_count = 0
        self.sample_count_threshold = 20000
        self.cache = OrderedDict() 
        self.max_cache_size = max_cache_size
    
    def calculate_max_segment_size(self,n, m, x):
        total_length = n - x
        y = total_length // m
        return y

    def __getitem__(self, index):
        local_index = index // self.gpus
        samples = []
        elapsed_time = 0
        for step in range(local_index, local_index + self.sample_len):
            file_idx = step // 4
            file_idx_rel = step % 4
            if file_idx not in self.cache.keys():
                data_dict = load(self.local_file_paths[file_idx])
                self.cache[file_idx] = data_dict
                if len(self.cache) > self.max_cache_size:
                    min_key = min(self.cache.keys())
                    self.cache.pop(min_key)
            
            start_time = time.time()
            data_dict = self.cache[file_idx]
            ch_num = len(self.var) * len(self.heights) + len(self.time_pos_var_list + self.surface_var_list)
            data = np.empty((ch_num, 721, 1440), dtype=np.float32) 
            ch = 0 
            height_keys = [var + height for var in self.var for height in self.heights]
            for key in height_keys:
                data[ch] = data_dict[key][file_idx_rel,:,:].copy()
                ch += 1
            for key in self.time_pos_var_list:
                data[ch] = data_dict[key][file_idx_rel,:,:].copy()
                ch += 1
            for key in self.surface_var_list:
                data[ch] = data_dict[key][file_idx_rel,:,:].copy()
                ch += 1
            samples.append(data)
            end_time = time.time()
            elapsed_time += (end_time - start_time)
        samples = np.array(samples)
        return samples

    def __len__(self):
        return self.local_sample_num * self.gpus

    def __str__(self):
        return (f"=========ERA5_Dataset======\n"
                f"Number of file paths: {len(self.file_paths)}\n"
                f"is_train: {self.is_train}\n"
                f"Rank: {self.rank}\n"
                f"Number of files in rank: {len(self.local_file_paths)}\n"
                f"Sample len: {self.sample_len}\n"
                f"Total indices: {self.__len__()}\n"
                f"Cache file number in each rank: {self.max_cache_size}\n"
                f"Pressure Variables: {self.var}\n"
                f"Heights: {self.heights}\n"
                f"Surface Variables: {self.surface_var_list}\n"
                f"Time and Pos Variables: {self.time_pos_var_list}\n"
                f"=========ERA5_Dataset======\n")
    
    def release_memory(self):
        pass

# if __name__ == "__main__":
#     train_dataset = ERA5_Dataset(input_dir='/work2/esm10/ERA5_data/data_norm', year_range=[2016, 2021], is_train=True, sample_len=4, eval_sample=365,max_cache_size=4)
#     train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
#     print(train_dataset)
#     start_time = time.time()  
#     for i, samples in enumerate(train_data_loader):
#         print(f"Sample {i} shapes: {samples.shape}")
#         end_time = time.time()  
#         print(f"Time taken for sample {i}: {end_time - start_time} seconds")
#         start_time = time.time()  
#         if i == 10:  
#             break
    
#     val_dataset = ERA5_Dataset(input_dir='/work2/esm10/ERA5_data/data_norm', year_range=[2016, 2021], is_train=False, sample_len=4, eval_sample=365,max_cache_size=2)
#     val__data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#     print(val_dataset)

