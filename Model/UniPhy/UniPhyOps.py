import torch
import torch.nn as nn
import torch.fft
import math
from PScan import PScanTriton

class MetricAwareCliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, img_height):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        lat_indices = torch.arange(img_height)
        lat_rad = (lat_indices / (img_height - 1)) * math.pi - (math.pi / 2)
        metric_factor = torch.sqrt(torch.cos(lat_rad).abs() + 1e-6)
        metric_factor = metric_factor / metric_factor.mean()
        self.register_buffer('metric_factor', metric_factor.view(1, 1, -1, 1))

    def forward(self, x):
        x_scaled = x * self.metric_factor
        out = self.conv(x_scaled)
        out = out / self.metric_factor
        return out

class SymplecticPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dt_ref = dt_ref
        self.skew_kernel = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def get_operators(self, dt):
        K = self.skew_kernel.triu(1)
        A = K - K.t()
        H_matrix = 1j * A
        L_real, V = torch.linalg.eigh(H_matrix)
        L = -1j * L_real
        V_inv = V.conj().t()

        if dt.ndim == 2:
            dt_cast = dt.unsqueeze(-1)
        elif dt.ndim == 1:
            dt_cast = dt.view(-1, 1, 1)
        else:
            dt_cast = dt.unsqueeze(-1)

        evo_diag = torch.exp(L.view(1, 1, -1) * dt_cast)
        return V, V_inv, evo_diag

class SpectralStep(nn.Module):
    def __init__(self, dim, h, w, viscosity=1e-4):
        super().__init__()
        self.dim = dim
        self.viscosity = viscosity
        self.weight = nn.Parameter(torch.randn(dim, h, w, dtype=torch.cfloat) * 0.02)
        kx = torch.fft.fftfreq(w, d=1.0)
        ky = torch.fft.fftfreq(h, d=1.0)
        k_x, k_y = torch.meshgrid(kx, ky, indexing='xy')
        k_sq = k_x ** 2 + k_y ** 2
        self.register_buffer('k_sq', k_sq)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')
        original_dc = x_fft[..., 0, 0].clone()
        x_fft = x_fft * self.weight
        dissipation = torch.exp(-self.viscosity * self.k_sq * (H * W))
        x_fft = x_fft * dissipation.unsqueeze(0).unsqueeze(0)
        x_fft[..., 0, 0] = original_dc
        out = torch.fft.ifft2(x_fft, s=(H, W), norm='ortho')
        return out

