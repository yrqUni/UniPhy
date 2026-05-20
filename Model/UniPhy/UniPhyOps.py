import torch
import torch.nn as nn


class RealMultiScaleSpatialMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch_local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.branch_regional = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim, bias=False),
            nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.branch_large = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim, bias=False),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        mid = max(1, dim // 4)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(dim, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, 3, 1),
        )
        nn.init.zeros_(self.mix_gate[-1].weight)
        nn.init.zeros_(self.mix_gate[-1].bias)
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y1 = self.branch_local(x)
        y2 = self.branch_regional(x)
        y3 = self.branch_large(x)
        weights = torch.softmax(self.mix_gate(y1 + y2 + y3), dim=1)
        delta = y1 * weights[:, 0:1] + y2 * weights[:, 1:2] + y3 * weights[:, 2:3]
        return x + delta * self.output_scale
