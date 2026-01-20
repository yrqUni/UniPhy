import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SWESphereDataset(Dataset):
    """
    模拟球面浅水方程数据
    在球面上设置一个高斯山包（Mountain），观察流体受地形影响产生的形变
    """
    def __init__(self, h=64, w=128, num_samples=1000, dt=0.1):
        self.h, self.w = h, w
        self.num_samples = num_samples
        
        lats = np.linspace(-90, 90, h)
        lons = np.linspace(0, 360, w)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        mountain = 1.0 * np.exp(-((lat_grid - 45)**2 / 100 + (lon_grid - 180)**2 / 400))
        self.topography = torch.tensor(mountain, dtype=torch.float32).unsqueeze(0)
        
        data = []
        for i in range(num_samples + 1):
            phase = i * dt
            field = np.sin(np.radians(lat_grid) * 2) * np.cos(np.radians(lon_grid) - phase)
            field = field * (1.0 + 0.5 * mountain) 
            data.append(field)
            
        data = np.array(data)
        self.samples = torch.tensor(data, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx], self.samples[idx+1], self.topography

def get_geo_data(batch_size=32):
    ds = SWESphereDataset()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader

