import torch
import torch.nn as nn
from UniPhyKernels import (
    CliffordConv2d,
    AdvectionStep,
    SpectralStep,
    HelmholtzProjection,
    StreamFunctionMixing
)

class UniPhyLayer(nn.Module):
    def __init__(self, emb_ch, input_shape, rank=32):
        super().__init__()
        H, W = input_shape
        self.emb_ch = emb_ch

        self.transport_op = AdvectionStep(emb_ch)

        self.interaction_op = nn.Sequential(
            CliffordConv2d(emb_ch, 3, 1),
            nn.SiLU(),
            nn.Conv2d(emb_ch, emb_ch, 1)
        )
        nn.init.zeros_(self.interaction_op[-1].weight)
        nn.init.eye_(self.interaction_op[-1].weight[:, :, 0, 0])
        nn.init.zeros_(self.interaction_op[-1].bias)

        self.dispersion_op = SpectralStep(emb_ch, rank=rank, w_freq=W//2+1)
        
        self.stream_mixing_op = StreamFunctionMixing(emb_ch, H, W)
        self.projection_op = HelmholtzProjection(H, W)
        
        self.norm = nn.GroupNorm(4, emb_ch)

    def forward(self, x, h_prev, dt):
        B, C, H, W = x.shape
        dt_flat = dt.view(-1)

        if h_prev is None:
            h_prev = torch.zeros_like(x)

        u = h_prev + x
        
        u = self.transport_op(u, dt_flat)
        
        u_interaction = self.interaction_op(u)
        u = u + u_interaction

        u_spec_delta = self.dispersion_op(u, dt_flat)
        u = u + u_spec_delta

        u = self.stream_mixing_op(u, dt_flat, projection_op=self.projection_op)
        u = self.projection_op(u)

        out = self.norm(u)
        
        return out, u

