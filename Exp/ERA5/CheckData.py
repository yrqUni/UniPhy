import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from ERA5 import ERA5_Dataset

def check_dataset_consistency():
    input_dir = '/nfs/ERA5_data/data_norm'
    sample_len = 8
    batch_size = 2
    
    dataset = ERA5_Dataset(
        input_dir=input_dir,
        year_range=[2000, 2001],
        is_train=True,
        sample_len=sample_len
    )
    
    print(f"Dataset Length: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    for i, batch in enumerate(loader):
        print(f"Batch {i} shape: {batch.shape}")
        
        if torch.isnan(batch).any():
            print("Error: NaN detected")
        
        if i == 0:
            first_sample = batch[0]
            second_sample = batch[1]
            diff = (first_sample[1:] - second_sample[:-1]).abs().sum()
            print(f"Sequential overlap diff: {diff.item()}")
            
        if i >= 5:
            break

def check_ddp_sampling():
    if not torch.cuda.is_available():
        return
        
    world_size = 2
    input_dir = '/nfs/ERA5_data/data_norm'
    
    dataset = ERA5_Dataset(
        input_dir=input_dir,
        year_range=[2000, 2002],
        sample_len=4
    )
    
    indices_rank0 = []
    sampler0 = DistributedSampler(dataset, num_replicas=world_size, rank=0, shuffle=False, drop_last=True)
    for idx in sampler0:
        indices_rank0.append(idx)
        
    indices_rank1 = []
    sampler1 = DistributedSampler(dataset, num_replicas=world_size, rank=1, shuffle=False, drop_last=True)
    for idx in sampler1:
        indices_rank1.append(idx)
        
    intersection = set(indices_rank0).intersection(set(indices_rank1))
    print(f"Rank 0 indices count: {len(indices_rank0)}")
    print(f"Rank 1 indices count: {len(indices_rank1)}")
    print(f"Overlapping indices: {len(intersection)}")
    
    if len(intersection) == 0:
        print("DDP Sampler check passed")
    else:
        print("DDP Sampler check failed")

if __name__ == "__main__":
    check_dataset_consistency()
    check_ddp_sampling()

