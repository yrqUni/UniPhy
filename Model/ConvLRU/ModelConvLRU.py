import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import triton
import triton.language as tl

from pscan import pscan


def _kaiming_like_(tensor: torch.Tensor) -> torch.Tensor:
    nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="relu")
    return tensor


def icnr_conv3d_weight_(weight: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    out_ch, in_ch, kD, kH, kW = weight.shape
    base_out = out_ch // (rH * rW)
    base = weight.new_zeros((base_out, in_ch, 1, kH, kW))
    _kaiming_like_(base)
    base = base.repeat_interleave(rH * rW, dim=0)
    with torch.no_grad():
        weight.copy_(base)
    return weight


def _bilinear_kernel_2d(kH: int, kW: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    factor_h = (kH + 1) // 2
    factor_w = (kW + 1) // 2
    center_h = factor_h - 1 if kH % 2 == 1 else factor_h - 0.5
    center_w = factor_w - 1 if kW % 2 == 1 else factor_w - 0.5
    og_h = torch.arange(kH, device=device, dtype=dtype)
    og_w = torch.arange(kW, device=device, dtype=dtype)
    fh = (1 - torch.abs(og_h - center_h) / factor_h).unsqueeze(1)
    fw = (1 - torch.abs(og_w - center_w) / factor_w).unsqueeze(0)
    return fh @ fw


def deconv3d_bilinear_init_(weight: torch.Tensor) -> torch.Tensor:
    in_ch, out_ch, kD, kH, kW = weight.shape
    with torch.no_grad():
        weight.zero_()
        kernel = _bilinear_kernel_2d(kH, kW, weight.device, weight.dtype)
        c = min(in_ch, out_ch)
        for i in range(c):
            weight[i, i, 0, :, :] = kernel
    return weight


def pixel_shuffle_hw_3d(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    N, C_mul, D, H, W = x.shape
    C = C_mul // (rH * rW)
    x = x.view(N, C, rH, rW, D, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.view(N, C, D, H * rH, W * rW)
    return x


def pixel_unshuffle_hw_3d(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    N, C, D, H, W = x.shape
    x = x.view(N, C, D, H // rH, rH, W // rW, rW)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.view(N, C * rH * rW, D, H // rH, W // rW)
    return x


@triton.jit
def fused_moe_router_kernel(
    logits_ptr,
    weights_out_ptr,
    indices_out_ptr,
    n_experts,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    logits_row_start = logits_ptr + pid * n_experts
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_experts
    logits = tl.load(logits_row_start + offsets, mask=mask, other=-float("inf"))
    logits_max = tl.max(logits, 0)
    logits = logits - logits_max
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, 0)
    probs = numerator / denominator
    weights_row_start = weights_out_ptr + pid * k
    indices_row_start = indices_out_ptr + pid * k
    for i in range(k):
        current_max_val = tl.max(probs, 0)
        current_max_idx = tl.argmax(probs, 0)
        tl.store(weights_row_start + i, current_max_val)
        tl.store(indices_row_start + i, current_max_idx)
        mask_max = offsets == current_max_idx
        probs = tl.where(mask_max, 0.0, probs)


@triton.jit
def fused_gate_kernel(
    in_ptr,
    out_ptr,
    C,
    Spatial,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    batch_idx = offsets // (C * Spatial)
    rem = offsets % (C * Spatial)
    c_idx = rem // Spatial
    s_idx = rem % Spatial

    in_idx_x = batch_idx * (2 * C * Spatial) + c_idx * Spatial + s_idx
    in_idx_g = batch_idx * (2 * C * Spatial) + (c_idx + C) * Spatial + s_idx

    x = tl.load(in_ptr + in_idx_x, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(in_ptr + in_idx_g, mask=mask, other=0.0).to(tl.float32)

    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = (x * sigmoid_x) * g

    tl.store(out_ptr + offsets, out, mask=mask)


class FusedGatedSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        B, C2, D, H, W = x.shape
        C = C2 // 2
        out = torch.empty((B, C, D, H, W), device=x.device, dtype=x.dtype)
        n_elements = B * C * D * H * W
        Spatial = D * H * W
        grid = (triton.cdiv(n_elements, 256),)
        fused_gate_kernel[grid](x, out, C, Spatial, n_elements, BLOCK_SIZE=256)
        return out


@triton.jit
def rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    x_row_start = x_ptr + row_idx * stride_x_row
    x = tl.load(x_row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.rsqrt(mean_sq + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    out_row_start = out_ptr + row_idx * stride_x_row
    tl.store(out_row_start + col_offsets, y, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous()

        x_flat = x.view(-1, self.channels)
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        M, N = x_flat.shape
        out = torch.empty_like(x_flat)

        grid = (M,)
        BLOCK_SIZE = triton.next_power_of_2(N)
        BLOCK_SIZE = max(1, min(BLOCK_SIZE, 4096))

        if N > 4096:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            out_norm = x * torch.rsqrt(var + self.eps)
            return out_norm * self.weight

        rms_norm_kernel[grid](
            x_flat,
            self.weight,
            out,
            x_flat.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        out = out.view(*x.shape)
        if len(orig_shape) == 5:
            out = out.permute(0, 4, 1, 2, 3).contiguous()

        return out


class AdaRMSNorm(nn.Module):
    def __init__(self, channels: int, cond_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(channels, eps=eps)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_channels, channels * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm(x)
        if cond is not None:
            emb = self.linear(self.silu(cond))
            scale, shift = torch.chunk(emb, 2, dim=1)
            scale = scale.view(x.size(0), x.size(1), 1, 1, 1) + 1.0
            shift = shift.view(x.size(0), x.size(1), 1, 1, 1)
            x = x * scale + shift
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features, 1, 1))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = (1, 3, 4)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise NotImplementedError


class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.pad_k = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_k, self.pad_k, 0, 0, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad_k, self.pad_k, self.pad_k, self.pad_k), mode="replicate")
        return self.conv(x)


class FactorizedPeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        self.pad_sp = kernel_size // 2
        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=0,
            bias=False,
        )
        self.depth_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(self.pad_sp, 0, 0),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_sp = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        x_sp = F.pad(x_sp, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        x_sp = x_sp.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4)
        out = self.depth_conv(x_sp)
        return out


class DiscreteCosineTransform(nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.n = n
        self.dim = dim
        k = torch.arange(n).unsqueeze(1)
        i = torch.arange(n).unsqueeze(0)
        basis = torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
        basis[0] = basis[0] * math.sqrt(1 / n)
        basis[1:] = basis[1:] * math.sqrt(2 / n)
        self.register_buffer("dct_matrix", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim, -1)
        out = torch.matmul(x, self.dct_matrix.t())
        return out.transpose(self.dim, -1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim, -1)
        out = torch.matmul(x, self.dct_matrix)
        return out.transpose(self.dim, -1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            if not torch.is_complex(x_ft):
                x_ft = x_ft.to(torch.cfloat)
            B, L, C, H, W_freq = x_ft.shape
            out_ft = torch.zeros(B, L, self.out_channels, H, W_freq, device=x_ft.device, dtype=torch.cfloat)
            m1 = min(H, self.modes1)
            m2 = min(W_freq, self.modes2)

            if m1 > 0 and m2 > 0:
                out_ft[:, :, :, :m1, :m2] = torch.einsum(
                    "blcxy,coxy->bloxy",
                    x_ft[:, :, :, :m1, :m2],
                    self.weights1[:, :, :m1, :m2]
                )

            if m1 > 0 and m2 > 0 and H > 1:
                out_ft[:, :, :, -m1:, :m2] = torch.einsum(
                    "blcxy,coxy->bloxy",
                    x_ft[:, :, :, -m1:, :m2],
                    self.weights2[:, :, :m1, :m2]
                )

            return out_ft


class SphericalHarmonicsPrior(nn.Module):
    def __init__(self, channels: int, H: int, W: int, Lmax: int = 6, rank: int = 8, gain_init: float = 0.0):
        super().__init__()
        self.C = int(channels)
        self.H = int(H)
        self.W = int(W)
        self.Lmax = int(Lmax)
        self.R = int(rank)
        self.K = self.Lmax * self.Lmax
        self.W1 = nn.Parameter(torch.zeros(self.C, self.R))
        self.W2 = nn.Parameter(torch.zeros(self.R, self.K))
        self.gain = nn.Parameter(torch.full((self.C,), float(gain_init)))
        nn.init.normal_(self.W1, std=1e-3)
        nn.init.normal_(self.W2, std=1e-3)
        theta, phi = self._latlon_to_spherical(self.H, self.W, device=None)
        Y = self._real_sph_harm_basis(theta, phi, self.Lmax)
        self.register_buffer("Y_real", Y, persistent=True)

    @staticmethod
    def _latlon_to_spherical(H: int, W: int, device: Optional[torch.device]):
        lat = torch.linspace(math.pi / 2, -math.pi / 2, steps=H, device=device)
        lon = torch.linspace(-math.pi, math.pi, steps=W, device=device)
        theta = (math.pi / 2 - lat).unsqueeze(1).repeat(1, W)
        phi = lon.unsqueeze(0).repeat(H, 1)
        return theta, phi

    @staticmethod
    def _fact_ratio(l: int, m_abs: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        l_t = torch.tensor(l, dtype=dtype, device=device)
        m_t = torch.tensor(m_abs, dtype=dtype, device=device)
        return torch.exp(torch.lgamma(l_t - m_t + 1) - torch.lgamma(l_t + m_t + 1))

    @staticmethod
    def _double_factorial(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if n < 1:
            return torch.tensor(1.0, dtype=dtype, device=device)
        seq = torch.arange(n, 0, -2, dtype=dtype, device=device)
        return torch.prod(seq)

    @staticmethod
    def _real_sph_harm_basis(theta: torch.Tensor, phi: torch.Tensor, Lmax: int) -> torch.Tensor:
        H, W = theta.shape
        device = theta.device
        dtype = theta.dtype
        x = torch.cos(theta)
        one = torch.ones_like(x)
        pi = torch.tensor(math.pi, device=device, dtype=dtype)

        P: List[List[Optional[torch.Tensor]]] = [[None] * (l + 1) for l in range(Lmax)]
        P[0][0] = one
        if Lmax >= 2:
            P[1][0] = x

        for l in range(2, Lmax):
            l_f = torch.tensor(l, device=device, dtype=dtype)
            P[l][0] = ((2 * l_f - 1) * x * P[l - 1][0] - (l_f - 1) * P[l - 2][0]) / l_f

        for m in range(1, Lmax):
            m_f = torch.tensor(m, device=device, dtype=dtype)
            P_mm = ((-1) ** m) * SphericalHarmonicsPrior._double_factorial(2 * m - 1, dtype, device) * (1 - x * x).pow(m_f / 2)
            P[m][m] = P_mm
            if m + 1 < Lmax:
                P[m + 1][m] = (2 * m_f + 1) * x * P_mm
            for l in range(m + 2, Lmax):
                l_f = torch.tensor(l, device=device, dtype=dtype)
                P[l][m] = ((2 * l_f - 1) * x * P[l - 1][m] - (l_f + m_f - 1) * P[l - 2][m]) / (l_f - m_f)

        idx = torch.arange(0, Lmax, device=device, dtype=dtype).view(-1, 1, 1)
        cos_mphi = torch.cos(idx * phi)
        sin_mphi = torch.sin(idx * phi)

        Ys: List[torch.Tensor] = []
        for l in range(Lmax):
            l_f = torch.tensor(l, device=device, dtype=dtype)
            for m in range(-l, l + 1):
                m_abs = abs(m)
                N_lm = torch.sqrt((2 * l_f + 1) / (4 * pi) * SphericalHarmonicsPrior._fact_ratio(l, m_abs, dtype, device))
                if m == 0:
                    Y = N_lm * P[l][0]
                elif m > 0:
                    Y = math.sqrt(2.0) * N_lm * P[l][m_abs] * cos_mphi[m_abs]
                else:
                    Y = math.sqrt(2.0) * N_lm * P[l][m_abs] * sin_mphi[m_abs]
                Ys.append(Y)

        return torch.stack(Ys, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        Y = self.Y_real
        if Y.device != x.device or Y.dtype != x.dtype:
            Y = Y.to(device=x.device, dtype=x.dtype)
        coeff = torch.matmul(self.W1, self.W2)
        Yf = Y.view(self.K, H * W)
        bias = torch.matmul(coeff, Yf).view(C, H, W)
        bias = (self.gain.view(C, 1, 1) * bias).view(1, 1, C, H, W)
        return x + bias


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.mlp = nn.Sequential(
            nn.Linear(int(channels), hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(channels), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BL, C, H, W = x.shape
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)
        max_pool, _ = torch.max(x.reshape(BL, C, -1), dim=-1)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(BL, C, 1, 1)
        return x * attn


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class CBAM2DPerStep(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention2D(int(channels), int(reduction))
        self.sa = SpatialAttention2D(int(spatial_kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        return x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


class CrossScaleAttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_g = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.conv_l = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.norm = RMSNorm(channels)

    def forward(self, local_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        g = self.conv_g(global_x)
        l = self.conv_l(local_x)
        psi = self.relu(g + l)
        attn = self.sigmoid(self.psi(psi))
        return self.norm(local_x * attn)


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], use_cbam: bool = False, cond_channels: Optional[int] = None, use_ada_norm: bool = False, ada_norm_cond_dim: Optional[int] = None):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.use_ada_norm = use_ada_norm
        self.dw_conv = FactorizedPeriodicConv3d(int(channels), int(channels), kernel_size=7)

        if self.use_ada_norm and ada_norm_cond_dim is not None:
            self.norm = AdaRMSNorm(int(channels), int(ada_norm_cond_dim))
            self.cond_proj = None
        else:
            self.norm = RMSNorm(int(channels))
            self.cond_channels_spatial = int(cond_channels) if cond_channels is not None else 0
            self.cond_proj = nn.Conv3d(self.cond_channels_spatial, int(channels) * 2, kernel_size=1) if self.cond_channels_spatial > 0 else None

        self.pw_conv_in = nn.Conv3d(int(channels), int(channels) * 2, kernel_size=1)
        self.fused_gate = FusedGatedSiLU()
        self.pw_conv_out = nn.Conv3d(int(channels), int(channels), kernel_size=1)
        self.cbam = CBAM2DPerStep(int(channels), reduction=16) if self.use_cbam else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)

        if self.use_ada_norm:
            x = self.norm(x, time_emb)
        else:
            x = self.norm(x)

        if self.cond_proj is not None and cond is not None:
            if cond.dim() == 4:
                cond_in = cond.unsqueeze(2)
            else:
                cond_in = cond
            if cond_in.shape[-2:] != x.shape[-2:]:
                cond_rs = F.interpolate(cond_in.squeeze(2), size=x.shape[-2:], mode="bilinear", align_corners=False).unsqueeze(2)
            else:
                cond_rs = cond_in

            affine = self.cond_proj(cond_rs)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta

        x = self.pw_conv_in(x)
        x = self.fused_gate(x)

        if self.cbam is not None:
            x = self.cbam(x)

        x = self.pw_conv_out(x)
        return residual + x


class SpatialPatchMoE(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], num_experts: int, active_experts: int, use_cbam: bool, cond_channels: Optional[int]):
        super().__init__()
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.patch_size = 8
        self.expert_hidden_size = (self.patch_size, self.patch_size)
        self.channels = int(channels)
        self.cond_channels = int(cond_channels) if cond_channels is not None else 0
        self.experts = nn.ModuleList(
            [
                GatedConvBlock(self.channels, self.expert_hidden_size, use_cbam=bool(use_cbam), cond_channels=(self.cond_channels if self.cond_channels > 0 else None))
                for _ in range(self.num_experts)
            ]
        )
        router_in_dim = self.channels + (self.cond_channels if self.cond_channels > 0 else 0)
        self.router = nn.Linear(router_in_dim, self.num_experts)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L, H, W = x.shape
        P = self.patch_size
        pad_h = (P - (H % P)) % P
        pad_w = (P - (W % P)) % P

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            if cond is not None:
                cond = F.pad(cond, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[-2:]
        nH, nW = H_pad // P, W_pad // P

        x_patches = (
            x.view(B, C, L, nH, P, nW, P)
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(B * nH * nW, C, L, P, P)
        )

        cond_patches = None
        router_input = x_patches.mean(dim=(2, 3, 4))

        if self.cond_channels > 0 and cond is not None:
            if cond.dim() == 4:
                cond_l = cond.unsqueeze(2).expand(-1, -1, L, -1, -1)
            else:
                cond_l = cond
            cond_patches = (
                cond_l.view(B, self.cond_channels, L, nH, P, nW, P)
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(B * nH * nW, self.cond_channels, L, P, P)
            )
            router_cond = cond_patches.mean(dim=(2, 3, 4))
            router_input = torch.cat([router_input, router_cond], dim=1)

        router_logits = self.router(router_input)

        N_total = router_logits.size(0)
        topk_weights = torch.empty((N_total, self.active_experts), device=x.device, dtype=x.dtype)
        topk_indices = torch.empty((N_total, self.active_experts), device=x.device, dtype=torch.int32)

        BLOCK_SIZE = triton.next_power_of_2(self.num_experts)
        fused_moe_router_kernel[(N_total,)](
            router_logits,
            topk_weights,
            topk_indices,
            self.num_experts,
            self.active_experts,
            BLOCK_SIZE,
        )
        topk_indices = topk_indices.long()
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        flat_indices = topk_indices.view(-1)
        x_repeated = x_patches.repeat_interleave(self.active_experts, dim=0)

        cond_repeated = None
        if cond_patches is not None:
            cond_repeated = cond_patches.repeat_interleave(self.active_experts, dim=0)

        sorted_expert_ids, sorted_args = torch.sort(flat_indices)
        x_sorted = x_repeated[sorted_args]

        cond_sorted = None
        if cond_repeated is not None:
            cond_sorted = cond_repeated[sorted_args]

        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts).tolist()
        y_sorted = torch.empty_like(x_sorted)

        start_idx = 0
        for i, count in enumerate(expert_counts):
            if count == 0:
                continue
            end_idx = start_idx + count
            inp_slice = x_sorted[start_idx:end_idx]

            c_slice = None
            if cond_sorted is not None:
                c_slice = cond_sorted[start_idx:end_idx]

            out_slice = self.experts[i](inp_slice.view(count, C, L, P, P), cond=c_slice)
            y_sorted[start_idx:end_idx] = out_slice
            start_idx = end_idx

        flat_weights = topk_weights.view(-1)
        weights_sorted = flat_weights[sorted_args]
        y_sorted_weighted = y_sorted * weights_sorted.view(-1, 1, 1, 1, 1)

        out_patches = torch.zeros_like(x_patches)
        token_ids = torch.arange(N_total, device=x.device).repeat_interleave(self.active_experts)
        sorted_token_ids = token_ids[sorted_args]
        out_patches.index_add_(0, sorted_token_ids, y_sorted_weighted.to(out_patches.dtype))

        out = (
            out_patches.view(B, nH, nW, C, L, P, P)
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, C, L, H_pad, W_pad)
        )

        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]

        return out


class ConvLRULayer(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        self.S = int(self.hidden_size[0])
        self.W = int(self.hidden_size[1])
        self.rank = int(getattr(args, "lru_rank", 32))
        self.is_selective = bool(getattr(args, "use_selective", False))
        self.bidirectional = bool(getattr(args, "bidirectional", False))
        self.use_checkpointing = bool(getattr(args, "use_checkpointing", False))
        self.W_freq = self.W // 2 + 1

        dt_min, dt_max = 0.001, 0.1
        ts = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), self.rank))
        nu = 1.0 / ts
        nu = nu.unsqueeze(0).repeat(self.emb_ch, 1)
        nu_log = torch.log(nu)

        u2 = torch.rand(self.emb_ch, self.rank)
        theta_log = torch.log(u2 * (2 * torch.tensor(np.pi)))
        self.params_log_base = nn.Parameter(torch.stack([nu_log, theta_log], dim=0))
        self.dispersion_mod = nn.Parameter(torch.zeros(2, self.emb_ch, self.rank) * 0.01)

        self.mod_hidden = 32
        in_dim = self.emb_ch if self.is_selective else (self.emb_ch + 1)
        self.forcing_mlp = nn.Sequential(
            nn.Linear(in_dim, self.mod_hidden),
            nn.Tanh(),
            nn.Linear(self.mod_hidden, self.emb_ch * self.rank * 2),
        )
        self.forcing_scale = nn.Parameter(torch.tensor(0.1))

        self.local_conv = PeriodicConv3d(self.emb_ch, self.emb_ch, kernel_size=3)

        self.dct_h = DiscreteCosineTransform(self.S, dim=-2)

        self.selection_net = nn.Sequential(
            nn.Linear(self.emb_ch, self.emb_ch),
            nn.SiLU(),
            nn.Linear(self.emb_ch, self.rank * 2),
        )

        self.U_row = nn.Parameter(torch.randn(self.emb_ch, self.S, self.rank, dtype=torch.cfloat) / math.sqrt(max(1, self.S)))
        self.V_col = nn.Parameter(torch.randn(self.emb_ch, self.W_freq, self.rank, dtype=torch.cfloat) / math.sqrt(max(1, self.W_freq)))

        C = self.emb_ch
        self.proj_W = nn.Parameter(torch.randn(C, C, dtype=torch.cfloat) / math.sqrt(max(1, C)))
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat)) if self.use_bias else None

        out_dim_fusion = self.emb_ch if not self.bidirectional else self.emb_ch * 2
        self.post_ifft_proj = nn.Conv3d(out_dim_fusion, self.emb_ch, kernel_size=(1, 1, 1), padding="same")

        self.norm = RMSNorm(self.emb_ch)

        self.noise_level = nn.Parameter(torch.tensor(0.01))

        self.freq_prior = SpectralConv2d(self.emb_ch, self.emb_ch, 8, 8) if bool(getattr(args, "use_freq_prior", False)) else None
        self.sh_prior = SphericalHarmonicsPrior(self.emb_ch, self.S, self.W, Lmax=int(getattr(args, "sh_Lmax", 6)), rank=int(getattr(args, "sh_rank", 8)), gain_init=float(getattr(args, "sh_gain_init", 0.0))) if bool(getattr(args, "use_sh_prior", False)) else None

        if bool(getattr(args, "use_gate", False)):
            self.gate_conv = nn.Sequential(
                nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"),
                nn.Sigmoid(),
            )
        else:
            self.gate_conv = None

        self.pscan = pscan

    def _apply_forcing(self, x: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = x.mean(dim=(-2, -1))
        if self.is_selective:
            inp = ctx
        else:
            dt_feat = dt.view(x.size(0), x.size(1), 1)
            inp = torch.cat([ctx, dt_feat], dim=-1)
        mod = self.forcing_mlp(inp)
        mod = mod.view(x.size(0), x.size(1), self.emb_ch, self.rank, 2)
        dnu = self.forcing_scale * torch.tanh(mod[..., 0])
        dth = self.forcing_scale * torch.tanh(mod[..., 1])
        return dnu.unsqueeze(-2), dth.unsqueeze(-2)

    def _hybrid_forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        x_dct = self.dct_h(x.float())
        x_hybrid = torch.fft.rfft(x_dct, dim=-1, norm="ortho")
        return x_hybrid

    def _hybrid_inverse_transform(self, h: torch.Tensor) -> torch.Tensor:
        x_dct = torch.fft.irfft(h, dim=-1, n=self.W, norm="ortho")
        x_out = self.dct_h.inverse(x_dct)
        return x_out

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C, S, W = x.shape
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)

        with torch.amp.autocast("cuda", enabled=False):
            x_in_fp32 = x.float()
            dt_fp32 = dt.float()
            h = self._hybrid_forward_transform(x_in_fp32)
            h = h.contiguous()

            ctx = x_in_fp32.mean(dim=(-2, -1))
            selection = self.selection_net(ctx)
            sel_u, sel_v = torch.chunk(selection, 2, dim=-1)
            scale_u = torch.sigmoid(sel_u).view(B, L, 1, 1, self.rank)
            scale_v = torch.sigmoid(sel_v).view(B, L, 1, 1, self.rank)

            h_perm = h.permute(0, 1, 2, 4, 3)
            t0 = torch.einsum("blcwh,chr->blcwr", h_perm, self.U_row)
            t0 = t0 * scale_u
            zq = torch.einsum("blcwr,cwr->blcwr", t0, self.V_col)
            zq = zq * scale_v.unsqueeze(-2)

            if self.proj_b is not None:
                zq = zq + self.proj_b.view(1, 1, C, 1, 1)

            nu_log, theta_log = self.params_log_base.unbind(dim=0)
            disp_nu, disp_th = self.dispersion_mod.unbind(dim=0)
            nu_base = torch.exp(nu_log + disp_nu).view(1, 1, C, 1, self.rank)
            th_base = torch.exp(theta_log + disp_th).view(1, 1, C, 1, self.rank)

            dnu_force, dth_force = self._apply_forcing(x_in_fp32, dt_fp32)

            nu_t = torch.clamp(nu_base * dt_fp32 + dnu_force, min=1e-6)
            th_t = th_base * dt_fp32 + dth_force
            lamb = torch.exp(torch.complex(-nu_t, th_t))

            if self.training:
                noise_std = self.noise_level * torch.sqrt(dt_fp32 + 1e-6)
                noise = torch.randn_like(zq) * noise_std
                x_in_lru = zq + noise
            else:
                x_in_lru = zq

            gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
            x_in_lru = x_in_lru * gamma_t

            zero_prev = torch.zeros_like(x_in_lru[:, :1])
            if last_hidden_in is not None:
                x_in_fwd = torch.cat([last_hidden_in.to(x_in_lru.dtype), x_in_lru], dim=1)
            else:
                x_in_fwd = torch.cat([zero_prev, x_in_lru], dim=1)
            
            lamb_fwd = torch.cat([lamb[:, :1], lamb], dim=1)
            lamb_in_fwd = lamb_fwd.expand_as(x_in_fwd).contiguous()

            B_sz, L_sz, C_sz, W_sz, R_sz = x_in_fwd.shape
            x_flat = x_in_fwd.view(B_sz, L_sz, -1).transpose(1, 2)
            l_flat = lamb_in_fwd.view(B_sz, L_sz, -1).transpose(1, 2)
            
            z_flat = self.pscan(l_flat, x_flat)
            z_out = z_flat.transpose(1, 2).view(B_sz, L_sz, C_sz, W_sz, R_sz)[:, 1:]
            last_hidden_out = z_out[:, -1:]

            if self.bidirectional:
                x_in_bwd = x_in_lru.flip(1)
                lamb_bwd = lamb.flip(1)
                x_in_bwd = torch.cat([zero_prev, x_in_bwd], dim=1)
                lamb_bwd = torch.cat([lamb_bwd[:, :1], lamb_bwd], dim=1)
                lamb_in_bwd = lamb_bwd.expand_as(x_in_bwd).contiguous()
                
                x_flat_b = x_in_bwd.view(B_sz, L_sz, -1).transpose(1, 2)
                l_flat_b = lamb_in_bwd.view(B_sz, L_sz, -1).transpose(1, 2)
                z_flat_b = self.pscan(l_flat_b, x_flat_b)
                z_out_bwd = z_flat_b.transpose(1, 2).view(B_sz, L_sz, C_sz, W_sz, R_sz)[:, 1:].flip(1)
            else:
                z_out_bwd = None

            def project_back(z: torch.Tensor, sc_u: torch.Tensor, sc_v: torch.Tensor) -> torch.Tensor:
                z = z * sc_v.unsqueeze(-2)
                t1 = torch.einsum("blcwr,cwr->blcwr", z, self.V_col.conj())
                t1 = t1 * sc_u
                rec = torch.einsum("blcwr,chr->blcwh", t1, self.U_row.transpose(1, 2).conj())
                return rec

            h_rec_fwd = project_back(z_out, scale_u, scale_v)
            h_rec_fwd = h_rec_fwd.permute(0, 1, 2, 4, 3)
            feat_fwd = self._hybrid_inverse_transform(h_rec_fwd)

            if self.bidirectional and z_out_bwd is not None:
                h_rec_bwd = project_back(z_out_bwd, scale_u, scale_v)
                h_rec_bwd = h_rec_bwd.permute(0, 1, 2, 4, 3)
                feat_bwd = self._hybrid_inverse_transform(h_rec_bwd)
                feat_final = torch.cat([feat_fwd, feat_bwd], dim=2)
            else:
                feat_final = feat_fwd

        feat_final = feat_final.to(x.dtype)
        h_final = self.post_ifft_proj(feat_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()

        if self.sh_prior is not None:
            h_final = self.sh_prior(h_final)

        x_local = self.local_conv(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        h_final = h_final + x_local

        h_final = self.norm(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()

        if self.gate_conv is not None:
            gate = self.gate_conv(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x_out = (1 - gate) * x + gate * h_final
        else:
            x_out = x + h_final

        return x_out, last_hidden_out


class FeedForward(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.ffn_hidden_ch = int(getattr(args, "ffn_hidden_ch", 32))
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        self.layers_num = int(getattr(args, "ffn_hidden_layers_num", 1))
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.num_expert = int(getattr(args, "num_expert", -1))
        self.activate_expert = int(getattr(args, "activate_expert", 2))

        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 1, 1), padding="same")
        blocks: List[nn.Module] = []
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        for _ in range(self.layers_num):
            if self.num_expert > 1:
                blocks.append(SpatialPatchMoE(self.ffn_hidden_ch, self.hidden_size, self.num_expert, self.activate_expert, self.use_cbam, cond_ch))
            else:
                blocks.append(GatedConvBlock(self.ffn_hidden_ch, self.hidden_size, use_cbam=self.use_cbam, cond_channels=cond_ch))
        self.blocks = nn.ModuleList(blocks)
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.c_in(x.permute(0, 2, 1, 3, 4))
        x = self.act(x)
        for blk in self.blocks:
            x = blk(x, cond=cond)
        x = self.c_out(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return residual + x


class ConvLRUBlock(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)
        self.use_checkpointing = bool(getattr(args, "use_checkpointing", False))

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        def _inner_forward(x_in, last_h, t_val, c_val):
            x_mid, h_out = self.lru_layer(x_in, last_h, listT=t_val)
            x_out = self.feed_forward(x_mid, cond=c_val)
            return x_out, h_out

        if self.training and x.requires_grad and self.use_checkpointing:
            return checkpoint.checkpoint(_inner_forward, x, last_hidden_in, listT, cond, use_reentrant=False)
        else:
            return _inner_forward(x, last_hidden_in, listT, cond)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        with torch.no_grad():
            self.embedding.weight.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        emb = self.embedding.weight
        dist = torch.cdist(x.float(), emb.float(), p=2)
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, emb).view(B, L, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices


class DiffusionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError(f"DiffusionHead dim must be even, got {self.dim}")
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError("DiffusionHead invalid dim")
        scale = math.log(10000.0) / max(1, (half_dim - 1))
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * (-scale))
        tt = t.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([tt.sin(), tt.cos()], dim=-1)
        return self.mlp(emb.to(dtype=t.dtype))


class Decoder(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.head_mode = str(getattr(args, "head_mode", "gaussian")).lower()
        self.output_ch = int(getattr(args, "out_ch", 1))
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.dec_hidden_ch = int(getattr(args, "dec_hidden_ch", 0))
        self.dec_hidden_layers_num = int(getattr(args, "dec_hidden_layers_num", 0))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        self.dec_strategy = str(getattr(args, "dec_strategy", "pxsf")).lower()
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])

        if self.dec_hidden_layers_num != 0 and self.dec_hidden_ch <= 0:
            self.dec_hidden_ch = self.emb_ch

        out_ch_after_up = self.dec_hidden_ch if self.dec_hidden_layers_num != 0 else self.emb_ch

        if self.dec_strategy == "deconv":
            self.upsp = nn.ConvTranspose3d(
                in_channels=self.emb_ch,
                out_channels=out_ch_after_up,
                kernel_size=(1, self.rH, self.rW),
                stride=(1, self.rH, self.rW),
            )
            deconv3d_bilinear_init_(self.upsp.weight)
            with torch.no_grad():
                if self.upsp.bias is not None:
                    self.upsp.bias.zero_()
            self.pre_shuffle_conv = None
        else:
            self.pre_shuffle_conv = nn.Conv3d(
                in_channels=self.emb_ch,
                out_channels=out_ch_after_up * self.rH * self.rW,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            )
            icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.rH, self.rW)
            with torch.no_grad():
                if self.pre_shuffle_conv.bias is not None:
                    self.pre_shuffle_conv.bias.zero_()
            self.upsp = None

        if self.dec_hidden_layers_num != 0:
            cond_ch = self.emb_ch if self.static_ch > 0 else None
            use_ada = (self.head_mode == "diffusion")
            ada_cond_dim = out_ch_after_up if use_ada else None
            self.c_hidden = nn.ModuleList(
                [
                    GatedConvBlock(
                        out_ch_after_up,
                        (1, 1),
                        use_cbam=False,
                        cond_channels=cond_ch,
                        use_ada_norm=use_ada,
                        ada_norm_cond_dim=ada_cond_dim,
                    )
                    for _ in range(self.dec_hidden_layers_num)
                ]
            )
        else:
            self.c_hidden = None

        if self.head_mode == "gaussian":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
            self.vq = None
        elif self.head_mode == "diffusion":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = DiffusionHead(out_ch_after_up)
            self.vq = None
        elif self.head_mode == "token":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
            self.vq = VectorQuantizer(num_embeddings=1024, embedding_dim=self.output_ch)
        else:
            raise ValueError(f"Unknown head_mode: {self.head_mode}")

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None):
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.dec_strategy == "deconv":
            if self.upsp is None:
                raise RuntimeError("Decoder misconfigured")
            x = self.upsp(x)
        else:
            if self.pre_shuffle_conv is None:
                raise RuntimeError("Decoder misconfigured")
            x = self.pre_shuffle_conv(x)
            x = pixel_shuffle_hw_3d(x, self.rH, self.rW)

        x = self.activation(x)

        t_emb = None
        if self.head_mode == "diffusion" and timestep is not None:
            if self.time_embed is None:
                raise RuntimeError("Diffusion head missing time_embed")
            t_emb = self.time_embed(timestep)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=cond, time_emb=t_emb)

        x = self.c_out(x)

        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            sigma = F.softplus(log_sigma) + 1e-6
            return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4).contiguous()

        if self.head_mode == "token":
            if self.vq is None:
                raise RuntimeError("Token head missing VQ")
            quantized, loss, indices = self.vq(x)
            return quantized.permute(0, 2, 1, 3, 4).contiguous(), loss, indices

        return x.permute(0, 2, 1, 3, 4).contiguous()


class ConvLRUModel(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.use_unet = bool(getattr(args, "unet", False))
        layers = int(getattr(args, "convlru_num_blocks", 2))
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.csa_blocks = nn.ModuleList()

        C = int(getattr(args, "emb_ch", input_downsp_shape[0]))
        H, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])

        if not self.use_unet:
            self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, (C, H, W)) for _ in range(layers)])
            self.upsample = None
            self.fusion = None
        else:
            curr_H, curr_W = H, W
            encoder_res: List[Tuple[int, int]] = []
            for i in range(layers):
                self.down_blocks.append(ConvLRUBlock(self.args, (C, curr_H, curr_W)))
                encoder_res.append((curr_H, curr_W))
                if i < layers - 1:
                    curr_H = max(1, curr_H // 2)
                    curr_W = max(1, curr_W // 2)
            for i in range(layers - 2, -1, -1):
                h_up, w_up = encoder_res[i]
                self.up_blocks.append(ConvLRUBlock(self.args, (C, h_up, w_up)))
                self.csa_blocks.append(CrossScaleAttentionGate(C))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            self.fusion = nn.Conv3d(C * 2, C, 1)
            self.convlru_blocks = None

    def forward(self, x: torch.Tensor, last_hidden_ins: Optional[List[torch.Tensor]] = None, listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            if self.convlru_blocks is None:
                raise RuntimeError("Model misconfigured")
            last_hidden_outs: List[torch.Tensor] = []
            for idx, blk in enumerate(self.convlru_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None and idx < len(last_hidden_ins)) else None
                x, h_out = blk(x, h_in, listT=listT, cond=cond)
                last_hidden_outs.append(h_out)
            return x, last_hidden_outs

        skips: List[torch.Tensor] = []
        last_hidden_outs: List[torch.Tensor] = []

        num_down = len(self.down_blocks)
        hs_in_down = last_hidden_ins[:num_down] if last_hidden_ins is not None else [None] * num_down
        hs_in_up = last_hidden_ins[num_down:] if last_hidden_ins is not None else [None] * len(self.up_blocks)

        for i, blk in enumerate(self.down_blocks):
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x, h_out = blk(x, hs_in_down[i], listT=listT, cond=curr_cond)
            last_hidden_outs.append(h_out)
            if i < len(self.down_blocks) - 1:
                skips.append(x)
                x_s = x.permute(0, 2, 1, 3, 4).contiguous()
                x_s = F.avg_pool3d(x_s, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                x = x_s.permute(0, 2, 1, 3, 4).contiguous()

        if self.upsample is None or self.fusion is None:
            raise RuntimeError("UNet misconfigured")

        for i, blk in enumerate(self.up_blocks):
            x_s = x.permute(0, 2, 1, 3, 4).contiguous()
            x_s = self.upsample(x_s)
            x = x_s.permute(0, 2, 1, 3, 4).contiguous()
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(-2) - x.size(-2)
                diffX = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            skip = self.csa_blocks[i](skip, x)

            x = torch.cat([x, skip], dim=2)
            x = self.fusion(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()

            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x, h_out = blk(x, hs_in_up[i], listT=listT, cond=curr_cond)
            last_hidden_outs.append(h_out)

        return x, last_hidden_outs


class Embedding(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.input_ch = int(getattr(args, "input_ch", 1))
        self.input_size = tuple(getattr(args, "input_size", (64, 64)))
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.emb_hidden_ch = int(getattr(args, "emb_hidden_ch", 32))
        self.emb_hidden_layers_num = int(getattr(args, "emb_hidden_layers_num", 0))
        self.static_ch = int(getattr(args, "static_ch", 0))
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])

        self.input_ch_total = self.input_ch + 4
        self.patch_embed = nn.Conv3d(
            self.input_ch_total,
            self.emb_hidden_ch,
            kernel_size=(1, self.rH + 2, self.rW + 2),
            stride=(1, self.rH, self.rW),
            padding=(0, 1, 1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch_total, 1, int(self.input_size[0]), int(self.input_size[1]))
            out_dummy = self.patch_embed(dummy)
            _, _, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, int(H), int(W))

        self.register_buffer("grid_embed", self._make_grid(args), persistent=False)

        self.hidden_size = (int(self.input_downsp_shape[1]), int(self.input_downsp_shape[2]))

        if self.static_ch > 0:
            self.static_embed = nn.Sequential(
                nn.Conv2d(
                    self.static_ch,
                    self.emb_ch,
                    kernel_size=(self.rH + 2, self.rW + 2),
                    stride=(self.rH, self.rW),
                    padding=(1, 1),
                ),
                nn.SiLU(),
            )
        else:
            self.static_embed = None

        if self.emb_hidden_layers_num > 0:
            cond_ch = self.emb_ch if self.static_ch > 0 else None
            self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_size, use_cbam=False, cond_channels=cond_ch) for _ in range(self.emb_hidden_layers_num)])
        else:
            self.c_hidden = None

        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.norm = RMSNorm(self.emb_ch)

    def _make_grid(self, args):
        H, W = tuple(getattr(args, "input_size", (64, 64)))
        lat = torch.linspace(-np.pi / 2, np.pi / 2, H)
        lon = torch.linspace(0, 2 * np.pi, W)
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
        emb = torch.stack([torch.sin(grid_lat), torch.cos(grid_lat), torch.sin(grid_lon), torch.cos(grid_lon)], dim=0)
        return emb.unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, C, L, H, W = x.shape
        grid = self.grid_embed.expand(B, -1, L, -1, -1).to(x.device, x.dtype)
        x = torch.cat([x, grid], dim=1)
        x = self.patch_embed(x)
        x = self.activation(x)

        cond = None
        if self.static_ch > 0 and self.static_embed is not None and static_feats is not None:
            cond = self.static_embed(static_feats)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=cond)

        x = self.c_out(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x, cond


class ConvLRU(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        self.revin = RevIN(int(getattr(args, "input_ch", 1)), affine=True)

        skip_contains = ["norm", "params_log", "prior", "post_ifft", "forcing", "dispersion", "dct_matrix", "grid_embed"]
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
        static_feats: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        cond = None
        if self.embedding.static_ch > 0 and self.embedding.static_embed is not None and static_feats is not None:
            cond = self.embedding.static_embed(static_feats)

        if mode == "p":
            x = self.revin(x, "norm")
            x_emb, _ = self.embedding(x, static_feats=static_feats)
            x_emb = x_emb.permute(0, 2, 1, 3, 4).contiguous()
            x_hid, _ = self.convlru_model(x_emb, listT=listT, cond=cond)
            out = self.decoder(x_hid, cond=cond, timestep=timestep)
            
            if self.decoder.head_mode == "gaussian":
                mu, sigma = torch.chunk(out, 2, dim=1)
                mu = mu.permute(0, 2, 1, 3, 4).contiguous()
                sigma = sigma.permute(0, 2, 1, 3, 4).contiguous()
                mu = self.revin(mu, "denorm")
                sigma = sigma * self.revin.stdev
                return torch.cat([mu, sigma], dim=2) 
            elif self.decoder.head_mode == "token":
                return out
            else:
                 out = out.permute(0, 2, 1, 3, 4).contiguous()
                 return self.revin(out, "denorm")

        if out_gen_num is None or int(out_gen_num) <= 0:
            raise ValueError("out_gen_num must be positive for inference mode")

        B = x.size(0)
        if listT is None:
            listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT

        out_list: List[torch.Tensor] = []
        
        x_norm = self.revin(x, "norm")
        x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
        x_emb = x_emb.permute(0, 2, 1, 3, 4).contiguous()
        
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT0, cond=cond)
        x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
        
        if isinstance(x_dec, tuple):
             x_dec0 = x_dec[0]
        else:
             x_dec0 = x_dec

        x_step_dist = x_dec0[:, :, -1:] 
        
        if str(self.decoder.head_mode).lower() == "gaussian":
             x_step_mean = x_step_dist[:, : int(getattr(self.args, "out_ch", x_step_dist.size(1) // 2))]
        else:
             x_step_mean = x_step_dist
             
        x_step_dist_perm = x_step_dist.permute(0, 2, 1, 3, 4).contiguous()
        
        if str(self.decoder.head_mode).lower() == "gaussian":
            mu = x_step_dist_perm[..., : int(getattr(self.args, "out_ch", x_step_dist_perm.size(2) // 2)), :, :]
            sigma = x_step_dist_perm[..., int(getattr(self.args, "out_ch", x_step_dist_perm.size(2) // 2)) :, :, :]
            mu_denorm = self.revin(mu, "denorm")
            sigma_denorm = sigma * self.revin.stdev
            out_list.append(torch.cat([mu_denorm, sigma_denorm], dim=2))
        elif str(self.decoder.head_mode).lower() == "token":
             out_list.append(x_step_dist_perm)
        else:
             out_list.append(self.revin(x_step_dist_perm, "denorm"))

        future = listT_future
        if future is None:
            future = torch.ones(B, int(out_gen_num) - 1, device=x.device, dtype=x.dtype)

        for t in range(int(out_gen_num) - 1):
            dt = future[:, t : t + 1]
            x_in, _ = self.embedding(x_step_mean, static_feats=None)
            x_in = x_in.permute(0, 2, 1, 3, 4).contiguous()
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt, cond=cond)
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            x_dec0 = x_dec[0] if isinstance(x_dec, tuple) else x_dec
            x_step_dist = x_dec0[:, :, -1:]
            if str(self.decoder.head_mode).lower() == "gaussian":
                x_step_mean = x_step_dist[:, : int(getattr(self.args, "out_ch", x_step_dist.size(1) // 2))]
            else:
                x_step_mean = x_step_dist
            x_step_dist_perm = x_step_dist.permute(0, 2, 1, 3, 4).contiguous()
            if str(self.decoder.head_mode).lower() == "gaussian":
                mu = x_step_dist_perm[..., : int(getattr(self.args, "out_ch", x_step_dist_perm.size(2) // 2)), :, :]
                sigma = x_step_dist_perm[..., int(getattr(self.args, "out_ch", x_step_dist_perm.size(2) // 2)) :, :, :]
                mu_denorm = self.revin(mu, "denorm")
                sigma_denorm = sigma * self.revin.stdev
                out_list.append(torch.cat([mu_denorm, sigma_denorm], dim=2))
            elif str(self.decoder.head_mode).lower() == "token":
                 out_list.append(x_step_dist_perm)
            else:
                 out_list.append(self.revin(x_step_dist_perm, "denorm"))

        return torch.cat(out_list, dim=1)
