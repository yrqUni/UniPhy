import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from CliffordConv2d import CliffordConv2d
from HelmholtzProjection import HelmholtzProjection
from PScan import PScanTriton

class UniPhyLayer(nn.Module):
    def __init__(self, emb_ch, input_shape):
        super().__init__()
        self.emb_ch = emb_ch
        self.h, self.w = input_shape
        self.fw = self.w // 2 + 1
        
        self.gen_proj = nn.Linear(emb_ch, emb_ch * 2)
        self.clifford_op = CliffordConv2d(emb_ch, kernel_size=3, padding=1)
        self.projection_op = HelmholtzProjection(self.h, self.w)
        self.norm = nn.GroupNorm(4, emb_ch)
        self.pscan = PScanTriton.apply

    def forward(self, x_seq, dt_seq):
        B, T, C, H, W = x_seq.shape
        dt = dt_seq.view(B, T, 1, 1, 1)

        x_fft = torch.fft.rfft2(x_seq, dim=(-2, -1), norm="ortho")

        stats = x_seq.mean(dim=(-1, -2))
        params = self.gen_proj(stats)
        nu, omega = torch.chunk(params, 2, dim=-1)
        
        lambda_c = torch.complex(-torch.abs(nu), omega).view(B, T, C, 1, 1)

        A_scan = torch.exp(lambda_c * dt)

        eps = 1e-6
        lambda_abs = torch.abs(lambda_c)
        is_small = lambda_abs < eps
        
        safe_lambda = torch.where(is_small, torch.full_like(lambda_c, eps), lambda_c)
        
        exp_lambda_dt = torch.exp(safe_lambda * dt)
        X_term = torch.where(
            is_small,
            dt.to(torch.complex64),
            (exp_lambda_dt - 1.0) / safe_lambda
        )
        
        X_scan = x_fft * X_term

        A_input = A_scan.expand(B, T, C, H, self.fw).to(torch.complex64)
        X_input = X_scan.to(torch.complex64)
        
        h_fft = self.pscan(A_input, X_input)

        h_spatial = torch.fft.irfft2(h_fft, s=(H, W), dim=(-2, -1), norm="ortho")
        
        h_spatial_flat = h_spatial.reshape(B * T, C, H, W)
        
        u = h_spatial_flat + self.clifford_op(h_spatial_flat)
        u = self.projection_op(u)
        
        out = self.norm(u)
        
        return out.view(B, T, C, H, W)

