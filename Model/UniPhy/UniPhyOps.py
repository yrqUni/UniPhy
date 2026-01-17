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

class NoiseInjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.global_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, dt):
        x_perm = x.permute(0, 2, 1, 3, 4)
        
        x_in = torch.cat([x_perm.real, x_perm.imag], dim=1)
        
        sigma = self.net(x_in) 
        sigma = sigma.permute(0, 2, 1, 3, 4)
        
        dt_shape = dt.view(dt.shape[0], dt.shape[1], 1, 1, 1)
        
        noise_std = torch.randn_like(x)
        
        diffusion = sigma * self.global_scale * noise_std * torch.sqrt(dt_shape.clamp(min=0.0))
        return diffusion

class SymplecticPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, stochastic=True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dt_ref = dt_ref
        
        self.frequencies = nn.Parameter(torch.randn(dim) * 1.0)
        self.basis_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
        self.stochastic = stochastic
        if stochastic:
            self.noise_injector = NoiseInjector(dim)
        else:
            self.noise_injector = None

    def get_orthogonal_basis(self):
        S = self.basis_generator.triu(1)
        S = S - S.t()
        I = torch.eye(self.dim, device=S.device)
        Q = torch.linalg.solve(I - S, I + S)
        return Q

    def get_operators(self, dt):
        Q = self.get_orthogonal_basis()
        V = Q.to(dtype=torch.cfloat)
        V_inv = V.T 

        L = 1j * self.frequencies

        if dt.ndim == 2:
            dt_cast = dt.unsqueeze(-1)
        elif dt.ndim == 1:
            dt_cast = dt.view(-1, 1, 1)
        else:
            dt_cast = dt.unsqueeze(-1)

        evo_diag = torch.exp(L.view(1, 1, -1) * dt_cast)
        
        return V, V_inv, evo_diag

    def inject_noise(self, x, dt):
        if self.stochastic and self.noise_injector is not None:
            return self.noise_injector(x, dt)
        return torch.zeros_like(x)

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

