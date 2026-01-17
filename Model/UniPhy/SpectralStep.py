import torch
import torch.nn as nn
import torch.fft

class SpectralStep(nn.Module):
    def __init__(self, dim, h, w, viscosity=1e-4):
        super().__init__()
        self.dim = dim
        self.viscosity = viscosity
        self.h = h
        self.w = w
        
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

