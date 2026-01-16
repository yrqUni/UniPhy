import torch
import torch.nn as nn
import torch.nn.functional as F

class HouseholderMixing(nn.Module):
    def __init__(self, channels, num_reflections=4):
        super().__init__()
        self.channels = channels
        self.num_reflections = num_reflections
        self.vs = nn.Parameter(torch.randn(num_reflections, channels, 1, 1))

    def forward(self, x):
        for i in range(self.num_reflections):
            v = self.vs[i]
            v = F.normalize(v, p=2, dim=0)

            x_real = x.real
            x_imag = x.imag

            dot_r = (x_real * v).sum(dim=1, keepdim=True)
            dot_i = (x_imag * v).sum(dim=1, keepdim=True)

            proj_r = v * dot_r
            proj_i = v * dot_i

            x_real = x_real - 2 * proj_r
            x_imag = x_imag - 2 * proj_i

            x = torch.complex(x_real, x_imag)
        return x

class PhasePotentialNet(nn.Module):
    def __init__(self, in_ch, hidden_ratio=4):
        super().__init__()
        hidden_dim = in_ch * hidden_ratio

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_ch, 1)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_mag):
        return self.net(x_mag)

class UniPhyParaPool(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        self.dim = dim
        self.potential_net = PhasePotentialNet(dim, hidden_ratio=expansion_factor)
        self.channel_mixer = HouseholderMixing(dim, num_reflections=4)
        self.scale_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        amplitude = x.abs()
        phase_shift = self.potential_net(amplitude)
        phase_shift = phase_shift * self.scale_factor

        modulator = torch.polar(torch.ones_like(amplitude), phase_shift)
        x_modulated = x * modulator

        x_mixed = self.channel_mixer(x_modulated)

        return x_mixed

