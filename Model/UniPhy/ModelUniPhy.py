import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import triton
import triton.language as tl

from PScan import pscan
from GridSamplePScan import GridSamplePScan, warp_common

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 4}, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 8}, num_stages=4),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 8}, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 8}, num_stages=4),
    ],
    key=["B", "L", "R", "C", "H", "Wf"],
)
@triton.jit
def koopman_A_kernel(
    nu_ptr,
    theta_ptr,
    dt_ptr,
    out_real_ptr,
    out_imag_ptr,
    B,
    L,
    R,
    C,
    H,
    Wf,
    stride_nu_b,
    stride_nu_l,
    stride_nu_r,
    stride_nu_c,
    stride_nu_wf,
    stride_dt_b,
    stride_dt_l,
    stride_o_b,
    stride_o_l,
    stride_o_r,
    stride_o_c,
    stride_o_h,
    stride_o_wf,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = B * L * R * C * H * Wf
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    idx_wf = offs % Wf
    tmp = offs // Wf
    idx_h = tmp % H
    tmp = tmp // H
    idx_c = tmp % C
    tmp = tmp // C
    idx_r = tmp % R
    tmp = tmp // R
    idx_l = tmp % L
    idx_b = tmp // L

    dt = tl.load(dt_ptr + idx_b * stride_dt_b + idx_l * stride_dt_l, mask=mask, other=0.0)

    nu_off = (
        idx_b * stride_nu_b
        + idx_l * stride_nu_l
        + idx_r * stride_nu_r
        + idx_c * stride_nu_c
        + idx_wf * stride_nu_wf
    )
    nu = tl.load(nu_ptr + nu_off, mask=mask, other=0.0)
    th = tl.load(theta_ptr + nu_off, mask=mask, other=0.0)

    decay = tl.exp(-nu * dt)
    ang = th * dt
    c = tl.cos(ang)
    s = tl.sin(ang)

    out_r = decay * c
    out_i = decay * s

    o_off = (
        idx_b * stride_o_b
        + idx_l * stride_o_l
        + idx_r * stride_o_r
        + idx_c * stride_o_c
        + idx_h * stride_o_h
        + idx_wf * stride_o_wf
    )
    tl.store(out_real_ptr + o_off, out_r, mask=mask)
    tl.store(out_imag_ptr + o_off, out_i, mask=mask)

def get_safe_groups(channels: int, target: int = 4) -> int:
    if channels % target == 0:
        return target
    return 1

def warp_image_step(x: torch.Tensor, flow: torch.Tensor, mode: str = 'bilinear', padding_mode: str = 'border') -> torch.Tensor:
    B, C, H, W = x.shape
    grid = warp_common(flow, B, H, W)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, L, H, W = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
            y = super().forward(y)
            return y.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return super().forward(x)

class DeformConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False, groups: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.groups = int(groups)
        
        offset_groups = self.groups
        self.conv_offset = nn.Conv2d(
            int(in_channels),
            int(2 * offset_groups * kernel_size * kernel_size + offset_groups * kernel_size * kernel_size),
            kernel_size=int(kernel_size),
            padding=int(padding),
            groups=self.groups
        )
        self.conv_dcn = torchvision.ops.DeformConv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(padding),
            bias=bool(bias),
            groups=self.groups
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.conv_dcn(x, offset, mask)

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, conv_type: str = "conv"):
        super().__init__()
        self.pad_sp = int(kernel_size) // 2
        self.conv_type = str(conv_type)
        
        if self.conv_type == "dcn":
            self.spatial_conv = DeformConv2d(
                int(in_channels), 
                int(in_channels), 
                kernel_size=int(kernel_size), 
                padding=0, 
                bias=False, 
                groups=int(in_channels)
            )
        else:
            self.spatial_conv = nn.Conv2d(
                int(in_channels), 
                int(in_channels), 
                kernel_size=int(kernel_size), 
                padding=0, 
                bias=False, 
                groups=int(in_channels)
            )
        
        self.depth_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        
        x_sp = F.pad(x_reshaped, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        
        return self.depth_conv(x_sp).view(B, L, -1, H, W).permute(0, 2, 1, 3, 4)

class EfficientSpatialGating(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(channels, mid_channels, 1),
            nn.SiLU(),
            nn.Conv1d(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        self.lat_mlp = nn.Sequential(
            nn.Conv1d(channels, mid_channels, 1),
            nn.SiLU(),
            nn.Conv1d(mid_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        
        g = self.global_pool(x_flat).view(B * L, C, 1)
        g_gate = self.global_mlp(g).view(B, L, C, 1, 1).permute(0, 2, 1, 3, 4)
        
        lat = x_flat.mean(dim=3).view(B * L, C, H)
        lat_gate = self.lat_mlp(lat).view(B, L, C, H, 1).permute(0, 2, 1, 3, 4)
        
        return x * g_gate * lat_gate

class SpectralMixer(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = int(rank)
        self.mix_linear = nn.Linear(self.rank * 2, self.rank * 2)
        nn.init.zeros_(self.mix_linear.weight)
        nn.init.zeros_(self.mix_linear.bias)
        with torch.no_grad():
            self.mix_linear.weight.view(self.rank * 2, self.rank * 2).diagonal().fill_(1.0)

    def forward(self, h_freq: torch.Tensor) -> torch.Tensor:
        h_real = h_freq.real
        h_imag = h_freq.imag
        h_stacked = torch.cat([h_real, h_imag], dim=-1)
        h_mixed = self.mix_linear(h_stacked)
        h_real_out, h_imag_out = torch.chunk(h_mixed, 2, dim=-1)
        return torch.complex(h_real_out, h_imag_out)

class LowFreqSpectralMixer(nn.Module):
    def __init__(self, channels: int, modes_h: int = 12, modes_w: int = 12, residual_scale: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        self.modes_h = int(modes_h)
        self.modes_w = int(modes_w)
        self.residual_scale = float(residual_scale)
        self.mix = nn.Linear(self.channels * 2, self.channels * 2)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        with torch.no_grad():
            self.mix.weight.view(self.channels * 2, self.channels * 2).diagonal().fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)
        xf = torch.fft.rfft2(x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W).float(), norm="ortho")
        xlow = xf[:, :, :mh, :mw]
        re = xlow.real
        im = xlow.imag
        z = torch.cat([re, im], dim=1).permute(0, 2, 3, 1).contiguous()
        z = self.mix(z).permute(0, 3, 1, 2).contiguous()
        re2, im2 = torch.chunk(z, 2, dim=1)
        xlow2 = torch.complex(re2, im2)
        xf2 = xf.clone()
        xf2[:, :, :mh, :mw] = xlow2
        y = torch.fft.irfft2(xf2, s=(H, W), norm="ortho").to(x.dtype)
        y = y.reshape(B, L, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x + self.residual_scale * y

class DynamicsParameterEstimator(nn.Module):
    def __init__(self, in_ch: int, emb_ch: int, rank: int, w_freq: int):
        super().__init__()
        self.w_freq = int(w_freq)
        self.emb_ch = int(emb_ch)
        self.rank = int(rank)
        ch = int(in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(ch, ch)
        self.pool = nn.AdaptiveAvgPool2d((1, self.w_freq))
        
        bottleneck = max(32, self.rank * 4)
        output_dim = self.emb_ch * self.rank * 3
        
        self.head = nn.Sequential(
            nn.Conv2d(ch, bottleneck, 1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, output_dim, 1)
        )
        
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.view(self.emb_ch, self.rank, 3)[:, :, 0].fill_(1.0)
            nn.init.uniform_(self.head[-1].bias.view(self.emb_ch, self.rank, 3)[:, :, 1], -0.01, 0.01)
            self.head[-1].bias.view(self.emb_ch, self.rank, 3)[:, :, 2].fill_(-5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        feat = self.conv(x)
        glob = self.global_pool(feat).flatten(1)
        glob = torch.sigmoid(self.global_fc(glob)).view(B, C, 1, 1)
        feat = feat * glob
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 3, 1).reshape(B, 1, self.w_freq, C).permute(0, 3, 1, 2)
        
        params = self.head(feat)
        params = params.permute(0, 2, 3, 1).view(B, 1, self.w_freq, self.emb_ch, self.rank, 3)
        params = params.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        nu_rate = F.softplus(params[..., 0])
        theta_rate = torch.tanh(params[..., 1]) * math.pi
        sigma = torch.sigmoid(params[..., 2])
        return nu_rate, theta_rate, sigma

class SpectralDynamics(nn.Module):
    def __init__(self, channels: int, rank: int, w_freq: int):
        super().__init__()
        self.channels = int(channels)
        self.rank = int(rank)
        self.w_freq = int(w_freq)
        self.estimator = DynamicsParameterEstimator(self.channels, self.channels, self.rank, self.w_freq)
        self.mixer = SpectralMixer(self.rank)

    def compute_params(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(B * L, C, H, W)
        nu_rate, theta_rate, sigma = self.estimator(x_flat)
        _, _, C_emb, Wf, R = nu_rate.shape
        nu_rate = nu_rate.view(B, L, 1, C_emb, Wf, R)
        theta_rate = theta_rate.view(B, L, 1, C_emb, Wf, R)
        sigma = sigma.view(B, L, 1, C_emb, Wf, R)
        return nu_rate, theta_rate, sigma

    def build_A_koop(self, nu_rate: torch.Tensor, theta_rate: torch.Tensor, dt_seq: torch.Tensor, H: int) -> torch.Tensor:
        B, L = dt_seq.shape
        nu = nu_rate.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()
        th = theta_rate.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()
        B2, L2, R, C, Wf = nu.shape
        if B2 != B or L2 != L:
            raise ValueError(f"nu shape {nu.shape} incompatible with dt {dt_seq.shape}")
        dt = dt_seq.contiguous()

        out_real = torch.empty((B, L, R, C, H, Wf), device=dt.device, dtype=dt.dtype)
        out_imag = torch.empty_like(out_real)

        grid = lambda META: (triton.cdiv(B * L * R * C * H * Wf, META["BLOCK_SIZE"]),)
        koopman_A_kernel[grid](
            nu,
            th,
            dt,
            out_real,
            out_imag,
            B,
            L,
            R,
            C,
            H,
            Wf,
            *nu.stride(),
            *dt.stride(),
            *out_real.stride(),
        )
        return torch.complex(out_real, out_imag)

class TransportParameterEstimator(nn.Module):
    def __init__(self, in_ch: int, emb_ch: int):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU()
        )
        self.flow_head = nn.Conv2d(in_ch, 2, 3, 1, 1)
        self.forcing_head = nn.Conv2d(in_ch, emb_ch, 3, 1, 1)
        
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        flow = torch.tanh(self.flow_head(feat))
        forcing = self.forcing_head(feat)
        return flow, forcing

def _match_dt_seq(dt_seq: torch.Tensor, L: int) -> torch.Tensor:
    if dt_seq.dim() != 2:
        raise ValueError(f"listT must be 2D [B,L], got {tuple(dt_seq.shape)}")
    if dt_seq.size(1) == L:
        return dt_seq
    if dt_seq.size(1) == 1:
        return dt_seq.repeat(1, L)
    if dt_seq.size(1) > L:
        return dt_seq[:, :L]
    pad = dt_seq[:, -1:].repeat(1, L - dt_seq.size(1))
    return torch.cat([dt_seq, pad], dim=1)

class ParallelPhysicalRecurrentLayer(nn.Module):
    def __init__(
        self,
        emb_ch: int,
        input_shape: Tuple[int, int],
        rank: int = 64,
        dt_ref: float = 1.0,
        inj_k: float = 2.0,
        use_noise: bool = False,
        noise_scale: float = 0.1,
        dynamics_mode: str = "spectral",
        interpolation_mode: str = "bilinear",
        **kwargs
    ):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.H, self.W = int(input_shape[0]), int(input_shape[1])
        self.Wf = self.W // 2 + 1
        self.rank = int(rank)
        self.dt_ref = float(dt_ref) if float(dt_ref) > 0 else 1.0
        self.inj_k = max(float(inj_k), 0.0)
        self.use_noise = bool(use_noise)
        self.noise_scale = float(noise_scale)
        self.dynamics_mode = str(dynamics_mode).lower()
        self.interpolation_mode = str(interpolation_mode).lower()

        self.norm = SpatialGroupNorm(get_safe_groups(self.emb_ch), self.emb_ch)
        self.gate = EfficientSpatialGating(self.emb_ch)

        if self.dynamics_mode == "spectral":
            self.koopman = SpectralDynamics(self.emb_ch, self.rank, self.Wf)
            self.proj_out = nn.Linear(self.rank, 1)
            self.post_ifft_proj = nn.Conv2d(self.emb_ch, self.emb_ch, kernel_size=1)
            self.rank_scale = nn.Parameter(torch.ones(self.rank) * 0.001)
        elif self.dynamics_mode == "advection":
            self.estimator = TransportParameterEstimator(self.emb_ch, self.emb_ch)
            self.flow_scale = nn.Parameter(torch.tensor(0.01))
            
            pscan_use_decay = kwargs.get("pscan_use_decay", True)
            pscan_use_residual = kwargs.get("pscan_use_residual", True)
            pscan_chunk_size = kwargs.get("pscan_chunk_size", 32)
            
            self.advection_pscan = GridSamplePScan(
                mode=self.interpolation_mode,
                channels=self.emb_ch,
                use_decay=pscan_use_decay,
                use_residual=pscan_use_residual,
                chunk_size=pscan_chunk_size
            )
        else:
            raise ValueError(f"Unknown dynamics_mode: {self.dynamics_mode}")

    def _build_injection_freq(self, x: torch.Tensor, dt_seq: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        dt_ref = torch.tensor(self.dt_ref, device=x.device, dtype=x.dtype).clamp_min(1e-6)
        inj_k = torch.tensor(self.inj_k, device=x.device, dtype=x.dtype)

        dt_scaled = (dt_seq / dt_ref).clamp_min(0.0)
        g = 1.0 - torch.exp(-dt_scaled * inj_k)
        g = g.view(B, L, 1, 1, 1, 1)

        x_bl = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W).float()
        xf = torch.fft.rfft2(x_bl, norm="ortho")
        xf = xf.reshape(B, L, C, H, self.Wf).to(torch.complex64)

        rs = self.rank_scale.to(x.device, x.dtype).view(1, 1, self.rank, 1, 1, 1)
        xf = xf.unsqueeze(2).expand(B, L, self.rank, C, H, self.Wf)
        xinj = xf * rs.to(torch.complex64)

        return xinj * g.to(torch.complex64)

    def _init_state_freq(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((B, self.rank, self.emb_ch, self.H, self.Wf), device=device, dtype=torch.complex64)

    def forward_spectral(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor],
        dt_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L, H, W = x.shape
        
        x_perm = x.permute(0, 2, 1, 3, 4).contiguous()
        nu_rate, theta_rate, sigma = self.koopman.compute_params(x_perm)
        A_koop = self.koopman.build_A_koop(nu_rate, theta_rate, dt_seq, H)
        X_inj = self._build_injection_freq(x, dt_seq)

        if self.use_noise and self.training:
             sig = sigma.squeeze(2).permute(0, 1, 4, 2, 3)
             sig = sig.unsqueeze(-2)
             dt_sqrt = torch.sqrt(dt_seq.view(B, L, 1, 1, 1, 1).clamp_min(1e-6))
             
             noise_real = torch.randn_like(X_inj.real)
             noise_imag = torch.randn_like(X_inj.imag)
             noise = torch.complex(noise_real, noise_imag)
             
             X_inj = X_inj + sig * self.noise_scale * dt_sqrt * noise

        A_flat = A_koop.permute(0, 1, 2, 3, 4, 5).reshape(B, L, -1).contiguous()
        X_flat = X_inj.permute(0, 1, 2, 3, 4, 5).reshape(B, L, -1).contiguous()

        if last_hidden_in is None:
            H0 = self._init_state_freq(B, x.device, x.dtype)
        else:
            if last_hidden_in.dtype.is_complex:
                H0 = last_hidden_in
            else:
                h0 = last_hidden_in.permute(0, 4, 1, 2, 3).contiguous()
                H0 = torch.fft.rfft2(h0.float(), norm="ortho").to(torch.complex64)

        A_cum = torch.cumprod(A_koop, dim=1)
        H_natural = A_cum * H0.unsqueeze(1)

        Y_forced_flat = pscan(A_flat.clone(), X_flat.clone())
        Y_forced = Y_forced_flat.view(B, L, self.rank, C, H, self.Wf)
        
        Y = Y_forced + H_natural

        h_space = torch.fft.irfft2(Y, s=(H, W), norm="ortho").to(x.dtype)
        h_stack = h_space.permute(0, 3, 1, 4, 5, 2).contiguous()
        h_last = h_stack[:, :, -1].contiguous()

        out = self.proj_out(h_stack).squeeze(-1)
        out = out.permute(0, 1, 4, 2, 3).reshape(B * L, C, H, W)
        out = self.post_ifft_proj(out)
        out = out.reshape(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        
        out = self.norm(out)
        out = self.gate(out)
        return x + out, h_last

    def forward_advection(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor],
        dt_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        
        flow_raw, forcing_raw = self.estimator(x_flat)
        dt_scale = (dt_seq / self.dt_ref).view(B, L, 1, 1, 1).to(x.device, x.dtype)

        flow = flow_raw.view(B, L, 2, H, W) * self.flow_scale * dt_scale
        forcing = forcing_raw.view(B, L, C, H, W) * dt_scale

        if self.use_noise and self.training:
            noise = torch.randn_like(forcing) * self.noise_scale
            noise = noise * torch.sqrt(dt_scale.clamp_min(1e-6))
            forcing = forcing + noise

        if last_hidden_in is not None:
            h0 = last_hidden_in
        else:
            h0 = torch.zeros(B, self.emb_ch, H, W, device=x.device, dtype=x.dtype)

        if last_hidden_in is not None:
            flow_step = flow[:, 0] 
            forcing_step = forcing[:, 0]
            
            h_warped = warp_image_step(h0, flow_step, mode=self.interpolation_mode, padding_mode="border")
            h_new = h_warped + forcing_step
            
            out = h_new.unsqueeze(2)
            out = self.norm(out)
            out = self.gate(out)
            return x + out, h_new

        h_forced = self.advection_pscan(flow, forcing)
        
        flow_cum = torch.cumsum(flow, dim=1)
        h_natural_list = []
        for t in range(L):
            flow_t = flow_cum[:, t]
            h_nat_t = warp_image_step(h0, flow_t, mode=self.interpolation_mode, padding_mode="border")
            h_natural_list.append(h_nat_t)
        h_natural = torch.stack(h_natural_list, dim=1)
        
        h_seq = h_forced + h_natural
        
        h_seq_perm = h_seq.permute(0, 2, 1, 3, 4)
        out = self.norm(h_seq_perm)
        out = self.gate(out)
        
        return x + out, h_seq[:, -1]

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor] = None,
        listT: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (x.shape[-2], x.shape[-1]) != (self.H, self.W):
            raise ValueError(f"input spatial {(x.shape[-2], x.shape[-1])} must match layer {(self.H, self.W)}")

        dt_seq_in = torch.ones(x.size(0), x.size(2), device=x.device, dtype=x.dtype) if listT is None else listT.to(x.device, x.dtype)
        dt_seq = _match_dt_seq(dt_seq_in, x.size(2))

        if self.dynamics_mode == "spectral":
            return self.forward_spectral(x, last_hidden_in, dt_seq)
        else:
            return self.forward_advection(x, last_hidden_in, dt_seq)

class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], kernel_size: int = 7, cond_channels: Optional[int] = None, conv_type: str = "conv"):
        super().__init__()
        self.dw_conv = PeriodicConv2d(int(channels), int(channels), kernel_size=int(kernel_size), conv_type=str(conv_type))
        self.norm = SpatialGroupNorm(get_safe_groups(int(channels)), int(channels))
        self.pw_conv_in = nn.Linear(int(channels), int(channels) * 2)
        self.pw_conv_out = nn.Linear(int(channels), int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.pw_conv_in(x)
        x_val, x_gate = torch.chunk(x, 2, dim=-1)
        x = x_val * F.sigmoid(x_gate)
        x = self.pw_conv_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], ffn_ratio: float = 4.0, conv_type: str = "conv"):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.ffn_ratio = float(ffn_ratio)
        self.hidden_dim = int(self.emb_ch * self.ffn_ratio)
        self.hidden_size = (int(input_shape[0]), int(input_shape[1]))
        self.conv_type = str(conv_type)
        self.c_in = nn.Linear(self.emb_ch, self.hidden_dim)
        self.block = GatedConvBlock(self.hidden_dim, self.hidden_size, kernel_size=7, conv_type=self.conv_type)
        self.c_out = nn.Linear(self.hidden_dim, self.emb_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 2, 3, 4, 1)
        x = self.c_in(x)
        x = self.act(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.block(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.c_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x

class UniPhyBlock(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], prl_args: Dict[str, Any], ffn_args: Dict[str, Any]):
        super().__init__()
        self.prl_layer = ParallelPhysicalRecurrentLayer(emb_ch, input_shape, **prl_args)
        self.feed_forward = FeedForwardNetwork(emb_ch, input_shape, **ffn_args)

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor] = None,
        listT: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mid, h_out = self.prl_layer(x, last_hidden_in, listT=listT)
        x_out = self.feed_forward(x_mid)
        return x_out, h_out

class BottleneckAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(dim) // int(num_heads)
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(int(dim), int(dim) * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(int(dim), int(dim))
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B * L, H * W, C)
        BN, N, Cc = x_flat.shape
        shortcut = x_flat
        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm).reshape(BN, N, 3, self.num_heads, Cc // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
        x_attn = x_attn.transpose(1, 2).reshape(BN, N, Cc)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        out = shortcut + x_attn
        out = out.view(B, L, H, W, Cc).permute(0, 4, 1, 2, 3).contiguous()
        return out

class CrossScaleGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.conv_g = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.conv_l = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(ch, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.norm = SpatialGroupNorm(get_safe_groups(ch), ch)

    def forward(self, local_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = local_x.shape
        loc = local_x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        glob = global_x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        
        g = self.conv_g(glob)
        l = self.conv_l(loc)
        psi = self.relu(g + l)
        attn = self.sigmoid(self.psi(psi))
        
        out = loc * attn
        return self.norm(out.view(B, L, C, H, W).permute(0, 2, 1, 3, 4))

class ShuffleDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(int(channels) * 4, int(channels), kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def pixel_unshuffle(input, downscale_factor):
            c = input.shape[1]
            kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor], device=input.device)
            for y in range(downscale_factor):
                for x in range(downscale_factor):
                    kernel[x + y * downscale_factor :: downscale_factor*downscale_factor, 0, y, x] = 1
            return F.conv2d(input, kernel, stride=downscale_factor, groups=c)
        
        B, C, L, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_down = pixel_unshuffle(x_reshaped, 2)
        x_down = self.proj(x_down)
        return x_down.view(B, L, -1, H // 2, W // 2).permute(0, 2, 1, 3, 4)

class UniPhyBackbone(nn.Module):
    def __init__(
        self,
        emb_ch: int,
        input_shape: Tuple[int, int],
        num_layers: int,
        arch_mode: str,
        down_mode: str,
        prl_args: Dict[str, Any],
        ffn_args: Dict[str, Any],
        spectral_modes_h: int = 12,
        spectral_modes_w: int = 12,
    ):
        super().__init__()
        self.arch_mode = str(arch_mode).lower()
        self.use_unet = self.arch_mode != "no_unet"
        layers = int(num_layers)
        self.down_mode = str(down_mode).lower()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.csa_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        C = int(emb_ch)
        H, W = int(input_shape[0]), int(input_shape[1])

        if not self.use_unet:
            self.uniphy_blocks = nn.ModuleList([UniPhyBlock(C, (H, W), prl_args, ffn_args) for _ in range(layers)])
            self.upsample = None
            self.fusion = None
            self.mid_attention = None
            self.mid_spectral = None
        else:
            curr_H, curr_W = H, W
            encoder_res: List[Tuple[int, int]] = []
            for i in range(layers):
                self.down_blocks.append(UniPhyBlock(C, (curr_H, curr_W), prl_args, ffn_args))
                encoder_res.append((curr_H, curr_W))
                if i < layers - 1:
                    if self.down_mode == "conv":
                        self.downsamples.append(nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1))
                    elif self.down_mode == "shuffle":
                        self.downsamples.append(ShuffleDownsample(C))
                    else:
                        self.downsamples.append(nn.AvgPool2d(kernel_size=2, stride=2))
                    if curr_H % 2 != 0: curr_H += 1
                    if curr_W % 2 != 0: curr_W += 1
                    curr_H = max(1, curr_H // 2)
                    curr_W = max(1, curr_W // 2)

            heads = 8
            for h in [8, 4, 2, 1]:
                if C % h == 0:
                    heads = h
                    break
            self.mid_attention = BottleneckAttention(C, num_heads=heads)
            self.mid_spectral = LowFreqSpectralMixer(C, modes_h=spectral_modes_h, modes_w=spectral_modes_w, residual_scale=1.0)

            for i in range(layers - 2, -1, -1):
                h_up, w_up = encoder_res[i]
                self.up_blocks.append(UniPhyBlock(C, (h_up, w_up), prl_args, ffn_args))
                self.csa_blocks.append(CrossScaleGate(C))

            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.fusion = nn.Conv2d(C * 2, C, 1)
            self.uniphy_blocks = None

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_ins: Optional[List[torch.Tensor]] = None,
        listT: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            last_hidden_outs: List[torch.Tensor] = []
            assert self.uniphy_blocks is not None
            for idx, blk in enumerate(self.uniphy_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None and idx < len(last_hidden_ins)) else None
                x, h_out = blk(x, h_in, listT=listT)
                last_hidden_outs.append(h_out)
            return x, last_hidden_outs

        skips: List[torch.Tensor] = []
        last_hidden_outs: List[torch.Tensor] = []
        num_down = len(self.down_blocks)
        hs_in_down = last_hidden_ins[:num_down] if last_hidden_ins is not None else [None] * num_down
        hs_in_up = last_hidden_ins[num_down:] if last_hidden_ins is not None else [None] * len(self.up_blocks)

        for i, blk in enumerate(self.down_blocks):
            x, h_out = blk(x, hs_in_down[i], listT=listT)
            last_hidden_outs.append(h_out)
            if i < len(self.down_blocks) - 1:
                skips.append(x)
                x_s = x
                pad_h = x_s.shape[-2] % 2
                pad_w = x_s.shape[-1] % 2
                if pad_h > 0 or pad_w > 0:
                    x_s = F.pad(x_s, (0, pad_w, 0, pad_h))
                
                B, C, L, H, W = x_s.shape
                x_flat = x_s.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
                
                if self.down_mode == "shuffle":
                    x = self.downsamples[i](x_s) 
                else:
                    x_down = self.downsamples[i](x_flat)
                    _, _, H_d, W_d = x_down.shape
                    x = x_down.view(B, L, C, H_d, W_d).permute(0, 2, 1, 3, 4)

        assert self.mid_attention is not None and self.mid_spectral is not None
        x = self.mid_attention(x)
        x = self.mid_spectral(x)

        for i, blk in enumerate(self.up_blocks):
            B, C, L, H, W = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
            x_up = self.upsample(x_flat)
            _, _, H_u, W_u = x_up.shape
            x = x_up.view(B, L, C, H_u, W_u).permute(0, 2, 1, 3, 4)

            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(-2) - x.size(-2)
                diffX = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            skip = self.csa_blocks[i](skip, x)
            x = torch.cat([x, skip], dim=1)
            
            B, C2, L, H, W = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C2, H, W)
            x_fused = self.fusion(x_flat)
            x = x_fused.view(B, L, -1, H, W).permute(0, 2, 1, 3, 4)

            x, h_out = blk(x, hs_in_up[i], listT=listT)
            last_hidden_outs.append(h_out)

        return x, last_hidden_outs

class FeatureEmbedding(nn.Module):
    def __init__(self, input_ch: int, input_size: Tuple[int, int], emb_ch: int, hidden_factor: Tuple[int, int] = (2, 2), **kwargs):
        super().__init__()
        self.input_ch = int(input_ch)
        self.input_size = tuple(input_size)
        self.emb_ch = int(emb_ch)
        self.emb_hidden_ch = self.emb_ch
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])
        self.input_ch_total = self.input_ch + 4
        self.patch_embed = nn.Conv2d(
            self.input_ch_total,
            self.emb_hidden_ch,
            kernel_size=(self.rH + 2, self.rW + 2),
            stride=(self.rH, self.rW),
            padding=(1, 1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch_total, int(self.input_size[0]), int(self.input_size[1]))
            out_dummy = self.patch_embed(dummy)
            _, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, int(H), int(W))
        self.register_buffer("grid_embed", self._make_grid(self.input_size), persistent=False)
        self.hidden_size = (int(self.input_downsp_shape[1]), int(self.input_downsp_shape[2]))
        
        self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_size, kernel_size=7)])
        self.c_out = nn.Conv2d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.norm = SpatialGroupNorm(get_safe_groups(self.emb_ch), self.emb_ch)

    def _make_grid(self, input_size: Tuple[int, int]) -> torch.Tensor:
        H, W = tuple(input_size)
        lat = torch.linspace(-math.pi / 2, math.pi / 2, int(H))
        lon = torch.linspace(0, 2 * math.pi, int(W))
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
        emb = torch.stack([torch.sin(grid_lat), torch.cos(grid_lat), torch.sin(grid_lon), torch.cos(grid_lon)], dim=0)
        return emb.unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Embedding expects [B,L,C,H,W], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, L, H, W = x.shape
        grid = self.grid_embed.expand(B, -1, L, -1, -1).to(x.device, x.dtype)
        x = torch.cat([x, grid], dim=1)
        
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, -1, H, W)
        x_emb = self.patch_embed(x_flat)
        x_emb = self.activation(x_emb)
        
        _, C_emb, H_emb, W_emb = x_emb.shape
        x = x_emb.view(B, L, C_emb, H_emb, W_emb).permute(0, 2, 1, 3, 4)

        for layer in self.c_hidden:
            x = layer(x)
            
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C_emb, H_emb, W_emb)
        x_out = self.c_out(x_flat)
        x = x_out.view(B, L, -1, H_emb, W_emb).permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        return x

class ProbabilisticDecoder(nn.Module):
    def __init__(self, out_ch: int, emb_ch: int, dist_mode: str = "gaussian", hidden_factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.dist_mode = str(dist_mode).lower()
        self.output_ch = int(out_ch)
        self.emb_ch = int(emb_ch)
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])
        out_ch_after_up = self.emb_ch
        
        self.pre_shuffle_conv = nn.Conv2d(
            in_channels=self.emb_ch,
            out_channels=out_ch_after_up * self.rH * self.rW,
            kernel_size=3,
            padding=1,
        )
        
        self.final_out_ch = self.output_ch * 2
        if self.dist_mode == "mdn":
            self.num_mixtures = 3
            self.final_out_ch = self.output_ch * 3 * self.num_mixtures

        self.c_out = nn.Conv2d(out_ch_after_up, self.final_out_ch, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        
        x_flat = self.pre_shuffle_conv(x_flat)
        
        B_flat, C_tot, H_in, W_in = x_flat.shape
        C_out = C_tot // (self.rH * self.rW)
        x_flat = x_flat.view(B_flat, C_out, self.rH, self.rW, H_in, W_in)
        x_flat = x_flat.permute(0, 1, 4, 2, 5, 3)
        x_flat = x_flat.reshape(B_flat, C_out, H_in * self.rH, W_in * self.rW)
        
        x_flat = self.activation(x_flat)
        x_flat = self.c_out(x_flat)
        
        _, C_out, H_out, W_out = x_flat.shape
        x = x_flat.view(B, L, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

        if self.dist_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
                sigma = F.softplus(log_sigma.float()).to(mu.dtype) + 1e-3
            return torch.cat([mu, sigma], dim=1)

        if self.dist_mode == "laplace":
            mu, log_b = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_b = torch.clamp(log_b, min=-5.0, max=5.0)
                b = F.softplus(log_b.float()).to(mu.dtype) + 1e-3
            return torch.cat([mu, b], dim=1)

        return x

@dataclass
class RevINStats:
    mean: torch.Tensor
    stdev: torch.Tensor

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features, 1, 1))
        self._last_stats: Optional[RevINStats] = None

    def stats(self, x: torch.Tensor) -> RevINStats:
        dim2reduce = (3, 4)
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            mean = torch.mean(x_fp32, dim=dim2reduce, keepdim=True).to(x.dtype)
            stdev = torch.sqrt(torch.var(x_fp32, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).to(x.dtype)
        return RevINStats(mean=mean.detach(), stdev=stdev.detach())

    def normalize(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        y = (x - stats.mean) / stats.stdev
        if self.affine:
            y = y * self.affine_weight + self.affine_bias
        return y

    def denormalize(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        y = x
        if self.affine:
            y = (y - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return y * stats.stdev + stats.mean

    def forward(self, x: torch.Tensor, mode: str, stats: Optional[RevINStats] = None) -> torch.Tensor:
        if mode == "norm":
            st = stats if stats is not None else self.stats(x)
            self._last_stats = st
            return self.normalize(x, st)
        if mode == "denorm":
            st = stats if stats is not None else self._last_stats
            if st is None:
                st = self.stats(x)
                self._last_stats = st
            return self.denormalize(x, st)
        raise NotImplementedError

class UniPhy(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        input_ch = int(getattr(args, "input_ch", 1))
        input_size = tuple(getattr(args, "input_size", (64, 64)))
        emb_ch = int(getattr(args, "emb_ch", 64))
        hidden_factor = getattr(args, "hidden_factor", (2, 2))

        self.embedding = FeatureEmbedding(input_ch, input_size, emb_ch, hidden_factor)

        num_blocks = int(getattr(args, "convlru_num_blocks", 2))
        arch = getattr(args, "Arch", "unet")
        down_mode = getattr(args, "down_mode", "avg")

        rank = int(getattr(args, "lru_rank", 64))
        dt_ref = float(getattr(args, "dt_ref", 1.0))
        inj_k = float(getattr(args, "inj_k", 2.0))
        
        koopman_use_noise = bool(getattr(args, "koopman_use_noise", False))
        koopman_noise_scale = float(getattr(args, "koopman_noise_scale", 1.0))
        
        dynamics_mode = getattr(args, "dynamics_mode", "spectral")
        interpolation_mode = getattr(args, "interpolation_mode", "bilinear")
        
        pscan_use_decay = bool(getattr(args, "pscan_use_decay", True))
        pscan_use_residual = bool(getattr(args, "pscan_use_residual", True))
        pscan_chunk_size = int(getattr(args, "pscan_chunk_size", 32))

        prl_args = {
            "rank": rank,
            "dt_ref": dt_ref,
            "inj_k": inj_k,
            "use_noise": koopman_use_noise,
            "noise_scale": koopman_noise_scale,
            "dynamics_mode": dynamics_mode,
            "interpolation_mode": interpolation_mode,
            "pscan_use_decay": pscan_use_decay,
            "pscan_use_residual": pscan_use_residual,
            "pscan_chunk_size": pscan_chunk_size,
        }

        ffn_ratio = float(getattr(args, "ffn_ratio", 4.0))
        conv_type = str(getattr(args, "ConvType", "conv"))
        ffn_args = {"ffn_ratio": ffn_ratio, "conv_type": conv_type}

        spectral_modes_h = int(getattr(args, "spectral_modes_h", 12))
        spectral_modes_w = int(getattr(args, "spectral_modes_w", 12))

        self.uniphy_model = UniPhyBackbone(
            emb_ch,
            self.embedding.input_downsp_shape[1:],
            num_blocks,
            arch,
            down_mode,
            prl_args,
            ffn_args,
            spectral_modes_h=spectral_modes_h,
            spectral_modes_w=spectral_modes_w,
        )

        out_ch = int(getattr(args, "out_ch", 1))
        dist_mode = getattr(args, "dist_mode", "gaussian")
        self.decoder = ProbabilisticDecoder(out_ch, emb_ch, dist_mode, hidden_factor)
        self.revin = RevIN(input_ch, affine=True)

        skip_contains = ["norm", "params_log", "prior", "post_ifft", "forcing", "dispersion", "dct_matrix", "grid_embed", "sobel"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains):
                    continue
                if n.endswith(".bias"):
                    p.zero_()
                elif p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "p",
        out_gen_num: Optional[int] = None,
        listT: Optional[torch.Tensor] = None,
        listT_future: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        revin_stats: Optional[RevINStats] = None,
    ):
        if mode == "p":
            stats = revin_stats if revin_stats is not None else self.revin.stats(x)
            x_norm = self.revin(x, "norm", stats=stats)
            x_emb = self.embedding(x_norm)
            x_hid, last_hidden_outs = self.uniphy_model(x_emb, listT=listT)
            out = self.decoder(x_hid)
            out_tensor = out.permute(0, 2, 1, 3, 4).contiguous()
            if str(self.decoder.dist_mode).lower() == "gaussian":
                mu, sigma = torch.chunk(out_tensor, 2, dim=2)
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin(mu, "denorm", stats=stats)
                    sigma = sigma * stats.stdev
                return torch.cat([mu, sigma], dim=2), last_hidden_outs
            if out_tensor.size(2) == self.revin.num_features:
                return self.revin(out_tensor, "denorm", stats=stats), last_hidden_outs
            return out_tensor, last_hidden_outs

        if out_gen_num is None or int(out_gen_num) <= 0:
            raise ValueError("out_gen_num must be positive for inference mode")

        B = x.size(0)
        listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype) if listT is None else listT

        stats = revin_stats if revin_stats is not None else self.revin.stats(x)

        out_list: List[torch.Tensor] = []
        x_norm = self.revin(x, "norm", stats=stats)
        x_emb = self.embedding(x_norm)
        x_hidden, last_hidden_outs = self.uniphy_model(x_emb, listT=listT0)
        x_dec0 = self.decoder(x_hidden)
        x_step_dist = x_dec0.permute(0, 2, 1, 3, 4).contiguous()
        x_step_dist = x_step_dist[:, -1:, :, :, :]

        dist_mode = str(self.decoder.dist_mode).lower()
        if dist_mode in ["gaussian", "laplace"]:
            out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
            mu = x_step_dist[:, :, :out_ch, :, :]
            scale = x_step_dist[:, :, out_ch:, :, :]
            if mu.size(2) == self.revin.num_features:
                mu_denorm = self.revin(mu, "denorm", stats=stats)
                scale_denorm = scale * stats.stdev
            else:
                mu_denorm = mu
                scale_denorm = scale
            out_list.append(torch.cat([mu_denorm, scale_denorm], dim=2))
            x_step_mean = mu_denorm
        else:
            out_list.append(x_step_dist)
            x_step_mean = x_step_dist[:, :, : self.revin.num_features] if x_step_dist.size(2) >= self.revin.num_features else x_step_dist

        future = torch.ones(B, int(out_gen_num) - 1, device=x.device, dtype=x.dtype) if listT_future is None else listT_future
        for t in range(int(out_gen_num) - 1):
            dt = future[:, t : t + 1]
            curr_x = x_step_mean
            if curr_x.shape[1] != 1:
                curr_x = curr_x[:, -1:, :, :, :]
            
            if curr_x.size(2) == self.revin.num_features:
                x_step_norm = self.revin(curr_x, "norm", stats=stats)
            else:
                x_step_norm = curr_x

            x_in = self.embedding(x_step_norm)
            x_hidden, last_hidden_outs = self.uniphy_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt)
            x_dec = self.decoder(x_hidden)
            x_step_dist = x_dec.permute(0, 2, 1, 3, 4).contiguous()
            x_step_dist = x_step_dist[:, -1:, :, :, :]

            if dist_mode in ["gaussian", "laplace"]:
                out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
                mu = x_step_dist[:, :, :out_ch, :, :]
                scale = x_step_dist[:, :, out_ch:, :, :]
                if mu.size(2) == self.revin.num_features:
                    mu_denorm = self.revin(mu, "denorm", stats=stats)
                    scale_denorm = scale * stats.stdev
                else:
                    mu_denorm = mu
                    scale_denorm = scale
                out_list.append(torch.cat([mu_denorm, scale_denorm], dim=2))
                x_step_mean = mu_denorm
            else:
                out_list.append(x_step_dist)
                x_step_mean = x_step_dist[:, :, : self.revin.num_features] if x_step_dist.size(2) >= self.revin.num_features else x_step_dist

        return torch.cat(out_list, dim=1), last_hidden_outs

