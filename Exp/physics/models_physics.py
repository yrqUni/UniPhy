import torch
import torch.nn as nn
import sys
import os

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

class DiscreteBaseline(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, N)
        )
    def forward(self, x):
        return x + self.net(x) * 0.1

class UniPhyPhysicsAdapter(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.model = UniPhyModel(
            in_channels=1, out_channels=1,
            embed_dim=64, depth=2, patch_size=1,
            img_height=1, img_width=N
        )
        self.N = N

    def forward(self, x, dt):
        x_5d = x.view(x.shape[0], 1, 1, 1, self.N)
        out_5d = self.model(x_5d, dt) 
        return out_5d.view(x.shape[0], self.N)

