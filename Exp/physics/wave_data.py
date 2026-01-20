import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WaveDataset(Dataset):
    def __init__(self, num_samples=1000, N=64, dt=1.0, c=0.2):
        self.N = N
        self.data = []
        x = np.linspace(0, 1, N, endpoint=False)
        
        for _ in range(num_samples):
            center = np.random.rand()
            width = np.random.uniform(50, 150)
            u0 = np.exp(-width * (x - center)**2)
            
            shift = c * dt
            u_next = np.exp(-width * ((x - (center + shift)) % 1.0 - 0.5 + 0.5)**2)
            
            u_t = u0
            u_tp1 = np.exp(-width * (np.mod(x - center - shift + 0.5, 1.0) - 0.5)**2)
            
            self.data.append((u_t, u_tp1))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.float32), \
               torch.tensor(self.data[idx][1], dtype=torch.float32)

def get_wave_loader(batch_size=64):
    ds = WaveDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

