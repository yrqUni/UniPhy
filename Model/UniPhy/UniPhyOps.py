import torch
import torch.nn as nn
from UniPhyKernels import (
    CliffordConv2d,
    FusedHamiltonian,
    AdvectionStep,
    SpectralStep,
    StreamFunctionMixing
)

class UniPhyLayer(nn.Module):
    def __init__(self, emb_ch, input_shape, rank=32):
        super().__init__()
        H, W = input_shape
        self.emb_ch = emb_ch

        self.clifford_in = CliffordConv2d(emb_ch, 3, 1)
        self.hamiltonian = FusedHamiltonian.apply
        self.h_params_r = nn.Parameter(torch.randn(emb_ch, H, W//2+1) * 0.01)
        self.h_params_i = nn.Parameter(torch.randn(emb_ch, H, W//2+1) * 0.01)
        self.sigma = nn.Parameter(torch.tensor(0.02))

        self.advection = AdvectionStep(emb_ch)
        self.spectral = SpectralStep(emb_ch, rank=rank, w_freq=W//2+1)

        self.stream_fix = StreamFunctionMixing(emb_ch, H, W)
        self.out_proj = nn.Conv2d(emb_ch, emb_ch, 1)
        self.norm = nn.GroupNorm(4, emb_ch)

    def forward(self, x, h_prev, dt):
        B, C, H, W = x.shape
        dt_flat = dt.view(-1)

        if h_prev is None:
            h_prev = torch.zeros_like(x)

        state = h_prev + x

        state = self.advection(state, dt_flat)
        state = self.spectral(state, dt_flat)

        h_geo = self.clifford_in(state)
        h_geo_f = torch.fft.rfft2(h_geo, norm='ortho')
        
        z_real = h_geo_f.real.contiguous()
        z_imag = h_geo_f.imag.contiguous()
        
        hr, hi = self.hamiltonian(z_real, z_imag, self.h_params_r, self.h_params_i, dt_flat, self.sigma)
        h_geo_next = torch.fft.irfft2(torch.complex(hr, hi), s=(H, W), norm='ortho')

        state = h_geo_next
        state_clean = self.stream_fix(state, dt_flat)

        out = self.norm(self.out_proj(state_clean))
        return out, state_clean

