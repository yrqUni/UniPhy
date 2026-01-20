import torch
import torch.nn as nn
import sys
import os

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel

print("✅ Successfully imported UniPhyModel from /nfs/UniPhy/Model/UniPhy")

class DeterministicBaseline(nn.Module):
    def __init__(self, N=40, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N)
        )

    def forward(self, x):
        return x + self.net(x) * 0.01

class UniPhyLorenzAdapter(nn.Module):
    def __init__(self, N=40):
        super().__init__()
        
        self.model = UniPhyModel(
            in_channels=1,
            out_channels=1,     
            embed_dim=64,
            depth=2,
            patch_size=1,
            img_height=1,       
            img_width=N,        
            dropout=0.0
        )
        
        self.N = N
        self.logvar_head = nn.Parameter(torch.zeros(1, N) - 5.0)

    def forward(self, x):
        B = x.shape[0]
        
        x_5d = x.view(B, 1, 1, 1, self.N)
        
        dt = torch.ones(B, 1, device=x.device) * 0.01
        
        out_5d = self.model(x_5d, dt)
        
        mu = out_5d.view(B, self.N)
        
        logvar = self.logvar_head.expand(B, self.N)
        logvar = torch.clamp(logvar, min=-10.0, max=-2.0)
        
        return mu, logvar

    def sample(self, x, n_samples=1):
        samples = []
        for _ in range(n_samples):
            mu, _ = self.forward(x)
            samples.append(mu)
            
        if n_samples == 1:
            return samples[0]
        else:
            return torch.stack(samples)

