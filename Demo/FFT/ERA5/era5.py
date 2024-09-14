from joblib import dump, load
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class ERA5_Dataset(Dataset):
    def __init__(self, input_dir='./data/data_norm', year_range=[2000, 2021], is_train=True, sample_len=16, label_len=16, eval_sample=400):
        self.file_paths = []
        self.sample_len = sample_len
        self.label_len = label_len
        self.var = ['U', 'V', 'RH', 'T']
        self.heights = ['150', '200', '250', '300', '400', '500', '600', '700']
        self.sample_count = 0
        self.sample_count_threshold = 20000
        
        years = list(range(year_range[0], year_range[1] + 1))
        for year in years:
            subdir_name = f"{year}"
            subdir_path = os.path.join(input_dir, subdir_name)
            if os.path.exists(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if os.path.isfile(os.path.join(subdir_path, file_name)) and file_name.isdigit() and len(file_name) == 8:
                        file_path = os.path.join(subdir_path, file_name)
                        self.file_paths.append(file_path)
        self.file_paths = sorted(self.file_paths)
        if is_train:
            self.file_paths = self.file_paths[:-eval_sample]
        else:
            self.file_paths = self.file_paths[-eval_sample:]

    def __getitem__(self, index):
        self.sample_count += 1
        if self.sample_count >= self.sample_count_threshold:
            self.release_memory()

        samples = []
        labels = []

        for step in range(index, index + self.sample_len + self.label_len):
            file_idx = step // 4
            file_idx_rel = step % 4
            data_dict = load(self.file_paths[file_idx])
            ch_num = len(self.var) * len(self.heights)
            data = np.empty((ch_num, 64, 64), dtype=np.float32)
            ch = 0
            for var_idx, var in enumerate(self.var):
                for height_idx, height in enumerate(self.heights):
                    key = var + height
                    data[ch] = data_dict[key][file_idx_rel, :, :]
                    ch += 1
            if step < index + self.sample_len:
                samples.append(data)
            else:
                labels.append(data)

        samples = np.array(samples)
        labels = np.array(labels)
        return samples, labels

    def __len__(self):
        return len(self.file_paths) * 4 - self.sample_len - self.label_len + 1

    def __str__(self):
        return (f"=========ERA5_Dataset======\n"
                f"Number of file paths: {len(self.file_paths)}\n"
                f"Total indices: {self.__len__()}\n"
                f"Variables: {self.var}\n"
                f"Heights: {self.heights}\n"
                f"=========ERA5_Dataset======\n")

    def release_memory(self):
        torch.cuda.empty_cache()
        # print("Memory released")

# if __name__ == "__main__":
#     dataset = ERA5_Dataset(input_dir='/data/lzh/jshasyh/yrqUni/Data/ERA5_norm', year_range=[2000, 2021], is_train=True, sample_len=16, label_len=16, eval_sample=400)
#     data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
#     print(dataset)
#     for i, (samples, labels) in enumerate(data_loader):
#         print(f"Sample {i} shapes: {samples.shape}, {labels.shape}")
#         if i == 5:
#             break

