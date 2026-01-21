import torch
import torch.nn as nn
import sys
import os

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from UniPhyOps import RiemannianCliffordConv2d

class CNNBaseline(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_ch, 3, padding=1)
        )
    def forward(self, x, topo):
        return x + self.net(x) * 0.1

class UniPhyGeoAdapter(nn.Module):
    def __init__(self, h, w, in_ch=1, out_ch=1, hidden=64):
        super().__init__()
        self.geo_conv = RiemannianCliffordConv2d(
            in_channels=in_ch, 
            out_channels=hidden, 
            kernel_size=3, 
            padding=1, 
            img_height=h, 
            img_width=w
        )
        self.out_conv = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, x, topo):
        feat = torch.relu(self.geo_conv(x))
        return x + self.out_conv(feat) * 0.1

