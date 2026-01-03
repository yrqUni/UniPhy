import math
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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

def pixel_shuffle_hw_3d(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    N, C_mul, D, H, W = x.shape
    C = C_mul // (rH * rW)
    x = x.reshape(N, C, rH, rW, D, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.reshape(N, C, D, H * rH, W * rW)
    return x

def pixel_unshuffle_hw_3d(x: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    N, C, D, H, W = x.shape
    x = x.reshape(N, C, D, H // rH, rH, W // rW, rW)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.reshape(N, C * rH * rW, D, H // rH, W // rW)
    return x

@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_row, stride_col,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    x_ptr_row = x_ptr + row_idx * stride_row
    x = tl.load(x_ptr_row + col_offsets * stride_col, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.rsqrt(mean_sq + eps)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    out_ptr_row = out_ptr + row_idx * stride_row
    tl.store(out_ptr_row + col_offsets * stride_col, y, mask=mask)

class RMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            x = x.real
        permuted = False
        if x.shape[-1] != self.channels:
            if x.dim() == 5 and x.shape[1] == self.channels:
                x = x.permute(0, 2, 3, 4, 1).contiguous()
                permuted = True
            elif x.dim() == 4 and x.shape[1] == self.channels:
                x = x.permute(0, 2, 3, 1).contiguous()
                permuted = True
            elif x.dim() == 6 and x.shape[2] == self.channels: 
                x = x.permute(0, 1, 3, 4, 5, 2).contiguous() 
                permuted = True
            else:
                 return F.rms_norm(x, (self.channels,), self.weight, self.eps)
        
        shape_permuted = x.shape
        x_flat = x.reshape(-1, self.channels)
        M, N = x_flat.shape
        out = torch.empty_like(x_flat)
        BLOCK_SIZE = triton.next_power_of_2(N)
        BLOCK_SIZE = max(1, min(BLOCK_SIZE, 4096))
        if N > 4096: 
             return F.rms_norm(x, (self.channels,), self.weight, self.eps)
        grid = (M,)
        rms_norm_kernel[grid](
            x_flat, self.weight, out,
            x_flat.stride(0), x_flat.stride(1),
            N, self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        out = out.reshape(*shape_permuted)
        if permuted:
            if out.dim() == 5: 
                out = out.permute(0, 4, 1, 2, 3).contiguous()
            elif out.dim() == 4:
                out = out.permute(0, 3, 1, 2).contiguous()
            elif out.dim() == 6:
                out = out.permute(0, 1, 5, 2, 3, 4).contiguous()
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

@triton.jit
def gated_silu_kernel(
    x_ptr, out_ptr,
    stride_row_in, stride_col_in,
    stride_row_out, stride_col_out,
    C_half,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C_half
    base_in = x_ptr + row_idx * stride_row_in
    val = tl.load(base_in + col_offsets * stride_col_in, mask=mask, other=0.0)
    gate = tl.load(base_in + (col_offsets + C_half) * stride_col_in, mask=mask, other=0.0)
    sig_gate = 1.0 / (1.0 + tl.exp(-gate))
    res = val * sig_gate
    base_out = out_ptr + row_idx * stride_row_out
    tl.store(base_out + col_offsets * stride_col_out, res, mask=mask)

class FusedGatedSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5: 
            B, C2, L, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C2)
        elif x.dim() == 4:
            B, C2, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, C2)
        else:
             chunk1, chunk2 = torch.chunk(x, 2, dim=1)
             return chunk1 * F.sigmoid(chunk2)
        N_rows, C2 = x_reshaped.shape
        C = C2 // 2
        out = torch.empty((N_rows, C), device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(C)
        grid = (N_rows,)
        gated_silu_kernel[grid](
            x_reshaped, out,
            x_reshaped.stride(0), x_reshaped.stride(1),
            out.stride(0), out.stride(1),
            C, BLOCK_SIZE=BLOCK_SIZE
        )
        if x.dim() == 5:
            B, _, L, H, W = x.shape
            return out.view(B, L, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        elif x.dim() == 4:
            B, _, H, W = x.shape
            return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

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
        dim2reduce = (3, 4)
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            self.mean = torch.mean(x_fp32, dim=dim2reduce, keepdim=True).detach().to(x.dtype)
            self.stdev = torch.sqrt(torch.var(x_fp32, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach().to(x.dtype)

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

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_offset = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size + kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.conv_dcn = torchvision.ops.DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=bias
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.conv_dcn(x, offset, mask)

class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.pad_k = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_k, self.pad_k, 0, 0, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad_k, self.pad_k, 0, 0), mode="replicate")
        return self.conv(x)

class FactorizedPeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, conv_type: str = "conv"):
        super().__init__()
        self.pad_sp = kernel_size // 2
        self.conv_type = conv_type
        if self.conv_type == "dcn":
            self.spatial_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False)
        else:
            self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=0, bias=False)
        self.depth_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_sp = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        x_sp = F.pad(x_sp, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        x_sp = x_sp.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        out = self.depth_conv(x_sp)
        return out

class DiscreteCosineTransform(nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.n = n
        self.dim = dim
        self.k = torch.arange(n).unsqueeze(1)
        self.i = torch.arange(n).unsqueeze(0)
        basis = torch.cos(math.pi * self.k * (2 * self.i + 1) / (2 * n))
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

class AdaptiveFreqGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x_ft).mean(dim=(-2, -1))
        scale = self.gate(mag).unsqueeze(-1).unsqueeze(-1)
        return x_ft * scale.to(x_ft.dtype)

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
        self.adaptive_gate = AdaptiveFreqGate(in_channels)

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            if not torch.is_complex(x_ft):
                x_ft = x_ft.to(torch.cfloat)
            x_ft = self.adaptive_gate(x_ft)
            B, L, C, H, W_freq = x_ft.shape
            out_ft = torch.zeros(B, L, self.out_channels, H, W_freq, device=x_ft.device, dtype=torch.cfloat)
            m1 = min(H, self.modes1)
            m2 = min(W_freq, self.modes2)
            if m1 > 0 and m2 > 0:
                out_ft[:, :, :, :m1, :m2] = torch.einsum("blcxy,coxy->bloxy", x_ft[:, :, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
            if m1 > 0 and m2 > 0 and H > 1:
                out_ft[:, :, :, -m1:, :m2] = torch.einsum("blcxy,coxy->bloxy", x_ft[:, :, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
            return out_ft

class DynamicSphericalHarmonicsPrior(nn.Module):
    def __init__(self, channels: int, H: int, W: int, Lmax: int = 6, rank: int = 8, gain_init: float = 0.0):
        super().__init__()
        self.C = int(channels)
        self.H = int(H)
        self.W = int(W)
        self.Lmax = int(Lmax)
        self.R = int(rank)
        self.K = self.Lmax * self.Lmax
        self.hypernet = nn.Sequential(
            nn.Linear(self.C, self.C // 2),
            nn.SiLU(),
            nn.Linear(self.C // 2, self.C * self.R + self.R * self.K)
        )
        self.gain = nn.Parameter(torch.full((self.C,), float(gain_init)))
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
            P_mm = ((-1) ** m) * DynamicSphericalHarmonicsPrior._double_factorial(2 * m - 1, dtype, device) * (1 - x * x).pow(m_f / 2)
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
                N_lm = torch.sqrt((2 * l_f + 1) / (4 * pi) * DynamicSphericalHarmonicsPrior._fact_ratio(l, m_abs, dtype, device))
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
        ctx = x.mean(dim=(-2, -1))
        weights_flat = self.hypernet(ctx)
        w1_size = self.C * self.R
        w1_flat = weights_flat[..., :w1_size]
        w2_flat = weights_flat[..., w1_size:]
        W1 = w1_flat.view(B, L, self.C, self.R)
        W2 = w2_flat.view(B, L, self.R, self.K)
        with torch.amp.autocast("cuda", enabled=False):
            coeff = torch.matmul(W1.float(), W2.float())
            Yf = Y.view(self.K, H * W).float()
            bias = torch.matmul(coeff, Yf).view(B, L, C, H, W)
            bias = (self.gain.view(1, 1, C, 1, 1).float() * bias)
            return x + bias.to(x.dtype)

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

class BottleneckAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B*L, H*W, C)
        B_N, N, C = x_flat.shape
        shortcut = x_flat
        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm).reshape(B_N, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B_N, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        out = shortcut + x_attn
        out = out.view(B, L, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return out

class CrossVariableAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, S, W = x.shape
        x_reshaped = x.permute(0, 1, 3, 4, 2).contiguous() 
        x_flat = x_reshaped.view(B * L * S * W, C)
        shortcut = x_flat
        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm).reshape(-1, C // self.num_heads, self.num_heads, 3).permute(3, 0, 2, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(-1, C)
        x_attn = self.proj(x_attn)
        out = shortcut + x_attn
        return out.view(B, L, S, W, C).permute(0, 1, 4, 2, 3).contiguous() 

class WaveletBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels * 4, hidden_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels * 4, kernel_size=1)
        
    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)

    def idwt_init(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, in_channel // 4, r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device, dtype=x.dtype)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

    def forward(self, x):
        B, L, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        B, L, C, H_pad, W_pad = x.shape
        x_flat = x.reshape(B * L, C, H_pad, W_pad)
        x_dwt = self.dwt_init(x_flat)
        x_feat = self.conv(x_dwt)
        x_feat = F.silu(x_feat)
        x_out_dwt = self.conv_out(x_feat)
        x_rec = self.idwt_init(x_out_dwt)
        out = x_rec.view(B, L, C, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]
        return out

class GraphInteraction(nn.Module):
    def __init__(self, channels, nodes=256):
        super().__init__()
        self.nodes = nodes
        self.proj_to_nodes = nn.Sequential(
            nn.AdaptiveAvgPool2d((int(math.sqrt(nodes)), int(math.sqrt(nodes)))),
            nn.Conv2d(channels, channels, 1)
        )
        self.gnn = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.proj_back = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        nodes = self.proj_to_nodes(x_flat)
        nodes = nodes.flatten(2).permute(0, 2, 1)
        nodes = self.norm(nodes)
        nodes = nodes + self.gnn(nodes)
        nodes = nodes.permute(0, 2, 1).reshape(B * L, C, int(math.sqrt(self.nodes)), int(math.sqrt(self.nodes)))
        out_grid = F.interpolate(nodes, size=(H, W), mode='bilinear', align_corners=False)
        out_grid = self.proj_back(out_grid)
        return out_grid.reshape(B, L, C, H, W)

class SpectralInteraction(nn.Module):
    def __init__(self, emb_ch, rank):
        super().__init__()
        self.mix_real = nn.Sequential(
            nn.Linear(rank, rank),
            nn.Tanh(),
            nn.Linear(rank, rank)
        )
        self.mix_imag = nn.Sequential(
            nn.Linear(rank, rank),
            nn.Tanh(),
            nn.Linear(rank, rank)
        )
    def forward(self, z):
        z_real = z.real
        z_imag = z.imag
        out_real = self.mix_real(z_real)
        out_imag = self.mix_imag(z_imag)
        return z + torch.complex(out_real, out_imag)

class ContextExtractor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x_fp = x.float()
            mean = x_fp.mean(dim=(-2, -1))
            var = x_fp.var(dim=(-2, -1), unbiased=False)
            std = torch.sqrt(var + 1e-6)
            max_val, _ = x_fp.view(x.size(0), x.size(1), x.size(2), -1).max(dim=-1)
        return torch.cat([mean, std, max_val], dim=-1).to(x.dtype)

class StochasticInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, 1, 1)
        )
        self.latest_kl = 0.0

    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            x = x.unsqueeze(1)
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        stats = self.conv(x_flat)
        mu, logvar = torch.chunk(stats, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.latest_kl = kl / (B * L)
        z = z.view(B, L, C, H, W)
        out = x + z
        if is_4d:
            return out.squeeze(1)
        return out

class HamiltonianGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1)
        )
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        H_field = self.conv(x_flat)
        
        lat = torch.linspace(-math.pi / 2, math.pi / 2, H, device=x.device).view(1, 1, H, 1)
        metric_correction = 1.0 / (torch.cos(lat) + 1e-6)
        
        H_padded = F.pad(H_field, (1, 1, 0, 0), mode='circular')
        H_padded = F.pad(H_padded, (0, 0, 1, 1), mode='replicate')
        
        grad_x = F.conv2d(H_padded, self.sobel_x.to(x.device))
        grad_y = F.conv2d(H_padded, self.sobel_y.to(x.device))
        
        u = grad_y 
        v = -grad_x * metric_correction
        
        flow = torch.cat([u, v], dim=1)
        flow = torch.tanh(flow) 
        
        return flow.view(B, L, 2, H, W)

class LieTransport(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h_prev, flow, dt):
        B, C, H, W, R = h_prev.shape
        chunk_size = 4
        h_warped_list = []
        
        for i in range(0, R, chunk_size):
            r_end = min(i + chunk_size, R)
            curr_R = r_end - i
            
            h_chunk = h_prev[..., i:r_end]
            h_flat = h_chunk.permute(0, 4, 1, 2, 3).reshape(B * curr_R, C, H, W)
            
            flow_chunk = flow.unsqueeze(1).repeat(1, curr_R, 1, 1, 1).reshape(B * curr_R, 2, H, W)
            dt_chunk = dt.view(B, 1, 1, 1).repeat_interleave(curr_R, dim=0)
            
            flow_dt = flow_chunk * dt_chunk
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=h_prev.device),
                torch.linspace(-1, 1, W, device=h_prev.device),
                indexing="ij"
            )
            base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).expand(B * curr_R, -1, -1, -1)
            
            flow_perm = flow_dt.permute(0, 2, 3, 1)
            sampling_grid = base_grid - flow_perm
            
            h_warped_flat = F.grid_sample(h_flat, sampling_grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            h_warped_list.append(h_warped_flat.reshape(B, curr_R, C, H, W).permute(0, 2, 3, 4, 1))
            
        return torch.cat(h_warped_list, dim=4)

class KoopmanParamEstimator(nn.Module):
    def __init__(self, in_ch, emb_ch, rank, w_freq):
        super().__init__()
        self.w_freq = w_freq
        self.emb_ch = emb_ch
        self.rank = rank
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, w_freq))
        self.head = nn.Linear(in_ch, emb_ch * rank * 3)

    def forward(self, x):
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        feat = self.conv(x_flat)
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 3, 1).contiguous().reshape(B, L, 1, self.w_freq, C)
        params = self.head(feat)
        params = params.view(B, L, self.w_freq, self.emb_ch, self.rank, 3)
        params = params.permute(0, 1, 3, 2, 4, 5).contiguous()
        nu = F.softplus(params[..., 0])
        theta = torch.tanh(params[..., 1]) * math.pi
        sigma = torch.sigmoid(params[..., 2])
        return nu, theta, sigma

class SpectralKoopmanSDE(nn.Module):
    def __init__(self, channels, rank, w_freq):
        super().__init__()
        self.estimator = KoopmanParamEstimator(channels, channels, rank, w_freq)
        self.channels = channels
        self.rank = rank
        
    def forward(self, h_trans, x_curr, dt):
        B, C, H, W, R = h_trans.shape 
        h_trans_perm = h_trans.permute(0, 4, 1, 2, 3) 
        h_freq = torch.fft.rfft2(h_trans_perm.float(), norm="ortho")
        
        is_x_4d = x_curr.dim() == 4
        if is_x_4d:
             x_curr = x_curr.unsqueeze(1)

        nu, theta, sigma = self.estimator(x_curr)
        
        dt_expanded = dt.view(dt.shape[0], dt.shape[1], 1, 1, 1, 1)
        nu = nu.unsqueeze(4) 
        theta = theta.unsqueeze(4)
        sigma = sigma.unsqueeze(4)
        
        nu = nu * dt_expanded
        theta = theta * dt_expanded
        
        decay = torch.exp(-nu)
        rotate = torch.exp(1j * theta)
        lambda_k = decay * rotate
        
        lambda_k = lambda_k.squeeze(1).permute(0, 4, 1, 3, 2)
        sigma_b = sigma.squeeze(1).permute(0, 4, 1, 3, 2)

        noise = torch.randn_like(h_freq) * sigma_b.mean(dim=-1, keepdim=True)
        
        h_evolved = h_freq * lambda_k + noise
        
        h_out = torch.fft.irfft2(h_evolved, s=(H, W), norm="ortho")
        return h_out.permute(0, 2, 3, 4, 1) 

class StaticInitState(nn.Module):
    def __init__(self, static_ch, emb_ch, rank, S, W_freq):
        super().__init__()
        self.static_ch = static_ch
        self.emb_ch = emb_ch
        self.rank = rank
        self.S = S
        self.W_freq = W_freq
        self.mapper = nn.Sequential(
            nn.Conv2d(static_ch, emb_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((S, W_freq)),
            nn.Conv2d(emb_ch, emb_ch * rank, kernel_size=1) 
        )

    def forward(self, static_feats):
        B = static_feats.size(0)
        out = self.mapper(static_feats) 
        out = out.view(B, self.emb_ch, self.rank, self.S, self.W_freq)
        return out.permute(0, 1, 3, 4, 2) 

class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], kernel_size: int = 7, use_cbam: bool = False, cond_channels: Optional[int] = None, use_ada_norm: bool = False, ada_norm_cond_dim: Optional[int] = None, conv_type: str = "conv"):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.use_ada_norm = use_ada_norm
        self.dw_conv = FactorizedPeriodicConv3d(int(channels), int(channels), kernel_size=kernel_size, conv_type=conv_type)
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
    def __init__(self, channels: int, hidden_size: Tuple[int, int], num_experts: int, active_experts: int, use_cbam: bool, cond_channels: Optional[int], conv_type: str = "conv"):
        super().__init__()
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.patch_size = 8
        self.expert_hidden_size = (self.patch_size, self.patch_size)
        self.channels = int(channels)
        self.cond_channels = int(cond_channels) if cond_channels is not None else 0
        expert_kernel_sizes = [3, 7, 11]
        self.experts = nn.ModuleList(
            [
                GatedConvBlock(self.channels, self.expert_hidden_size, kernel_size=expert_kernel_sizes[i % len(expert_kernel_sizes)], use_cbam=bool(use_cbam), cond_channels=(self.cond_channels if self.cond_channels > 0 else None), conv_type=conv_type)
                for i in range(self.num_experts)
            ]
        )
        self.shared_expert = nn.Sequential(
             GatedConvBlock(self.channels, self.expert_hidden_size, kernel_size=7, use_cbam=bool(use_cbam), cond_channels=(self.cond_channels if self.cond_channels > 0 else None), conv_type=conv_type),
             GatedConvBlock(self.channels, self.expert_hidden_size, kernel_size=3, use_cbam=False, cond_channels=(self.cond_channels if self.cond_channels > 0 else None), conv_type=conv_type)
        )
        router_in_dim = self.channels + (self.cond_channels if self.cond_channels > 0 else 0)
        self.router = nn.Linear(router_in_dim, self.num_experts)
        self.aux_loss = 0.0
        self.norm = RMSNorm(self.channels)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.aux_loss = 0.0
        B, C, L, H, W = x.shape
        out_frames = []
        total_aux_loss = 0.0
        
        for t in range(L):
            xt = x[:, :, t:t+1]
            ct = None
            if cond is not None:
                if cond.dim() == 5:
                     ct = cond[:, :, t:t+1]
                else:
                     ct = cond.unsqueeze(2) 
            
            P = self.patch_size
            pad_h = (P - (H % P)) % P
            pad_w = (P - (W % P)) % P
            if pad_h > 0 or pad_w > 0:
                xt = F.pad(xt, (0, pad_w, 0, pad_h))
                if ct is not None:
                    ct = F.pad(ct, (0, pad_w, 0, pad_h))
            
            H_pad, W_pad = xt.shape[-2:]
            nH, nW = H_pad // P, W_pad // P
            
            x_patches = (
                xt.view(B, C, 1, nH, P, nW, P)
                .permute(0, 3, 5, 2, 1, 4, 6)
                .reshape(B * nH * nW * 1, C, 1, P, P)
            )
            
            cond_patches = None
            router_input = x_patches.mean(dim=(2, 3, 4))
            
            if self.cond_channels > 0 and ct is not None:
                if ct.dim() == 4:
                    cond_l = ct.unsqueeze(2)
                else:
                    cond_l = ct
                cond_patches = (
                    cond_l.view(B, self.cond_channels, 1, nH, P, nW, P)
                    .permute(0, 3, 5, 2, 1, 4, 6)
                    .reshape(B * nH * nW * 1, self.cond_channels, 1, P, P)
                )
                router_cond = cond_patches.mean(dim=(2, 3, 4))
                router_input = torch.cat([router_input, router_cond], dim=1)

            s_out = x_patches
            for layer in self.shared_expert:
                 s_out = layer(s_out, cond=cond_patches)
            shared_out = s_out

            router_logits = self.router(router_input)
            
            with torch.amp.autocast("cuda", enabled=False):
                router_logits = router_logits.float()
                
                # Replace Triton Kernel with PyTorch Native TopK
                router_probs = F.softmax(router_logits, dim=-1)
                topk_weights, topk_indices = torch.topk(router_probs, self.active_experts, dim=-1)
                
                # Aux Loss calculation
                mean_probs = router_probs.mean(dim=0)
                flat_indices = topk_indices.view(-1)
                expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
                # Ensure it doesn't exceed num_experts size in rare cases
                if expert_counts.size(0) > self.num_experts:
                    expert_counts = expert_counts[:self.num_experts]
                
                fraction_selected = expert_counts / flat_indices.numel()
                step_aux_loss = (mean_probs * fraction_selected).sum() * self.num_experts
                
                # Normalize weights
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                
                total_aux_loss += step_aux_loss

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
                if count == 0: continue
                if i >= self.num_experts: break # Safety break
                end_idx = start_idx + count
                inp_slice = x_sorted[start_idx:end_idx]
                c_slice = cond_sorted[start_idx:end_idx] if cond_sorted is not None else None
                y_sorted[start_idx:end_idx] = self.experts[i](inp_slice, cond=c_slice)
                start_idx = end_idx
            
            flat_weights = topk_weights.view(-1)
            weights_sorted = flat_weights[sorted_args]
            y_sorted_weighted = y_sorted * weights_sorted.view(-1, 1, 1, 1, 1).to(y_sorted.dtype)
            
            out_patches = shared_out
            N_total = x_patches.shape[0]
            token_ids = torch.arange(N_total, device=x.device).repeat_interleave(self.active_experts)
            sorted_token_ids = token_ids[sorted_args]
            out_patches.index_add_(0, sorted_token_ids, y_sorted_weighted)
            
            out_patches_reshaped = out_patches.view(B * nH * nW * 1, C, 1, P, P).squeeze(2).permute(0, 2, 3, 1).contiguous()
            out_patches_norm = self.norm(out_patches_reshaped)
            out_patches = out_patches_norm.permute(0, 3, 1, 2).unsqueeze(2)
            
            out_t = (
                out_patches.view(B, nH, nW, 1, C, P, P)
                .permute(0, 4, 3, 1, 5, 2, 6)
                .reshape(B, C, 1, H_pad, W_pad)
            )
            if pad_h > 0 or pad_w > 0:
                out_t = out_t[..., :H, :W]
            
            out_frames.append(out_t)

        self.aux_loss = total_aux_loss / L
        return torch.cat(out_frames, dim=2)

class ContinuousHKLFLayer(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        H, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])
        self.rank = int(getattr(args, "lru_rank", 32))
        self.W_freq = W // 2 + 1
        
        self.hamiltonian = HamiltonianGenerator(self.emb_ch)
        self.lie_transport = LieTransport()
        self.koopman = SpectralKoopmanSDE(self.emb_ch, self.rank, self.W_freq)
        self.proj_out = nn.Linear(self.rank, 1)
        
        if bool(getattr(args, "learnable_init_state", False)) and int(getattr(args, "static_ch", 0)) > 0:
            self.init_state = StaticInitState(int(args.static_ch), self.emb_ch, self.rank, H, W)
        else:
            self.init_state = None
        
        self.use_graph_interaction = bool(getattr(args, "use_graph_interaction", False))
        if self.use_graph_interaction:
            self.graph_block = GraphInteraction(self.emb_ch)
        
        self.use_cross_var_attn = bool(getattr(args, "use_cross_var_attn", False))
        if self.use_cross_var_attn:
            self.cross_var_attn = CrossVariableAttention(self.emb_ch)
        
        self.use_stochastic = bool(getattr(args, "use_stochastic", False))
        if self.use_stochastic:
            self.stochastic_injector = StochasticInjector(self.emb_ch)
            
        self.use_wavelet_ssm = bool(getattr(args, "use_wavelet_ssm", False))
        if self.use_wavelet_ssm:
            self.wavelet_block = WaveletBlock(self.emb_ch, self.emb_ch)
            
        self.use_spectral_mixing = bool(getattr(args, "use_spectral_mixing", False))
        if self.use_spectral_mixing:
            self.spec_mixer = SpectralInteraction(self.emb_ch, self.rank)
            
        self.use_sh_prior = bool(getattr(args, "use_sh_prior", False))
        if self.use_sh_prior:
            self.sh_prior = DynamicSphericalHarmonicsPrior(self.emb_ch, H, W)
            
        self.use_freq_prior = bool(getattr(args, "use_freq_prior", False))
        if self.use_freq_prior:
            self.freq_prior = SpectralConv2d(self.emb_ch, self.emb_ch, 8, 8)
            
        self.post_ifft_proj = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.norm = RMSNorm(self.emb_ch)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_perm = x.permute(0, 2, 1, 3, 4)
        if self.use_cross_var_attn:
            x_perm = self.cross_var_attn(x_perm)
        if self.use_graph_interaction:
            x_perm = x_perm + self.graph_block(x_perm)
            
        B, L, C, H, W = x_perm.shape
        
        if last_hidden_in is None:
            if self.init_state is not None and static_feats is not None:
                h_prev = self.init_state(static_feats).unsqueeze(1).repeat(1, L, 1, 1, 1, 1) 
                h_prev = h_prev[:, 0] 
            else:
                h_prev = torch.zeros(B, C, H, W, self.rank, device=x.device, dtype=x.dtype)
        else:
            h_prev = last_hidden_in

        if listT is None:
            dt_seq = torch.ones(B, L, device=x.device, dtype=x.dtype)
        else:
            dt_seq = listT.to(x.device, x.dtype)

        h_states = []
        curr_h = h_prev
        self.latest_kl = 0.0
        
        for t in range(L):
            x_t = x_perm[:, t] 
            if self.use_stochastic and hasattr(self, 'stochastic_injector'):
                x_t = self.stochastic_injector(x_t)
                self.latest_kl += self.stochastic_injector.latest_kl

            dt_t = dt_seq[:, t].view(B, 1)
            
            flow = self.hamiltonian(x_t.unsqueeze(1)).squeeze(1)
            
            h_trans = self.lie_transport(curr_h, flow, dt_t)

            h_next = self.koopman(h_trans, x_t.unsqueeze(1), dt_t)

            x_inject = x_t.unsqueeze(-1).expand(-1, -1, -1, -1, self.rank)
            curr_h = h_next + x_inject
            h_states.append(curr_h)
            
        h_stack = torch.stack(h_states, dim=1)
        out = self.proj_out(h_stack).squeeze(-1)
        
        out = out.permute(0, 2, 1, 3, 4) 
        
        out = self.post_ifft_proj(out)

        if self.use_sh_prior:
            out = self.sh_prior(out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            
        if self.use_freq_prior:
            out_spatial = out.permute(0, 2, 1, 3, 4).contiguous()
            out_ft = torch.fft.rfft2(out_spatial.float(), norm="ortho")
            out_ft_res = self.freq_prior(out_ft)
            out_spatial_res = torch.fft.irfft2(out_ft_res, s=out_spatial.shape[-2:], norm="ortho")
            out = out + out_spatial_res.permute(0, 2, 1, 3, 4).to(out.dtype)

        if self.use_wavelet_ssm:
            out = out + self.wavelet_block(x_perm).permute(0, 2, 1, 3, 4)

        out = self.norm(out)
        gate = self.gate_conv(out)
        x_out = (1 - gate) * x + gate * out
        
        return x_out, curr_h

class FeedForward(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.ffn_ratio = float(getattr(args, "ffn_ratio", 4.0))
        self.hidden_dim = int(self.emb_ch * self.ffn_ratio)
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        self.layers_num = 1
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.num_expert = int(getattr(args, "num_expert", -1))
        self.activate_expert = int(getattr(args, "activate_expert", 2))
        self.conv_type = str(getattr(args, "ConvType", "conv"))
        self.c_in = nn.Conv3d(self.emb_ch, self.hidden_dim, kernel_size=(1, 1, 1), padding="same")
        blocks: List[nn.Module] = []
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        for _ in range(self.layers_num):
            if self.num_expert > 1:
                blocks.append(SpatialPatchMoE(self.hidden_dim, self.hidden_size, self.num_expert, self.activate_expert, self.use_cbam, cond_ch, conv_type=self.conv_type))
            else:
                blocks.append(GatedConvBlock(self.hidden_dim, self.hidden_size, kernel_size=7, use_cbam=self.use_cbam, cond_channels=cond_ch, conv_type=self.conv_type))
        self.blocks = nn.ModuleList(blocks)
        self.c_out = nn.Conv3d(self.hidden_dim, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.c_in(x)
        x = self.act(x)
        for blk in self.blocks:
            x = blk(x, cond=cond)
        x = self.c_out(x)
        return residual + x

class ConvLRUBlock(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.lru_layer = ContinuousHKLFLayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mid, h_out = self.lru_layer(x, last_hidden_in, listT=listT, static_feats=static_feats)
        x_out = self.feed_forward(x_mid, cond=cond)
        return x_out, h_out

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
        t_float = t.float()
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=t_float.dtype) * (-scale))
        tt = t_float.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([tt.sin(), tt.cos()], dim=-1)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))

class Decoder(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.head_mode = str(getattr(args, "head_mode", "gaussian")).lower()
        self.output_ch = int(getattr(args, "out_ch", 1))
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.dec_hidden_ch = self.emb_ch
        self.dec_hidden_layers_num = 0
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        out_ch_after_up = self.emb_ch
        self.pre_shuffle_conv = nn.Conv3d(
            in_channels=self.emb_ch, out_channels=out_ch_after_up * self.rH * self.rW, kernel_size=(1, 3, 3), padding=(0, 1, 1),
        )
        icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.rH, self.rW)
        with torch.no_grad():
            if self.pre_shuffle_conv.bias is not None:
                self.pre_shuffle_conv.bias.zero_()
        self.c_hidden = None
        if self.head_mode == "gaussian":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
            self.vq = None
        elif self.head_mode == "diffusion":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = DiffusionHead(out_ch_after_up)
            self.vq = None
        elif self.head_mode == "flow":
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
        if self.pre_shuffle_conv is None:
            raise RuntimeError("Decoder misconfigured")
        x = self.pre_shuffle_conv(x)
        x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        t_emb = None
        if self.head_mode in ["diffusion", "flow"] and timestep is not None:
            if self.time_embed is None:
                raise RuntimeError("Head missing time_embed")
            t_emb = self.time_embed(timestep)
        x = self.c_out(x)
        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
                sigma = F.softplus(log_sigma.float()).to(mu.dtype) + 1e-6
            return torch.cat([mu, sigma], dim=1)
        if self.head_mode == "token":
            if self.vq is None:
                raise RuntimeError("Token head missing VQ")
            quantized, loss, indices = self.vq(x)
            return quantized, loss, indices
        return x

class ShuffleDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_hw_3d(x, 2, 2)
        return self.proj(x)

class BiFPNFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.w2 = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.act = nn.SiLU()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x, skip):
        w1 = F.relu(self.w1)
        w2 = F.relu(self.w2)
        weight = w1 + w2 + 1e-4
        out = (w1 * x + w2 * skip) / weight
        return self.conv(self.act(out))

class ConvLRUModel(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.arch_mode = str(getattr(args, "Arch", "unet")).lower()
        self.use_unet = (self.arch_mode != "no_unet")
        layers = int(getattr(args, "convlru_num_blocks", 2))
        self.down_mode = str(getattr(args, "down_mode", "avg")).lower()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.csa_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
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
                    if self.down_mode == "conv":
                        self.downsamples.append(nn.Conv3d(C, C, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
                    elif self.down_mode == "shuffle":
                        self.downsamples.append(ShuffleDownsample(C))
                    else:
                        self.downsamples.append(nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
                    if curr_H % 2 != 0: curr_H += 1
                    if curr_W % 2 != 0: curr_W += 1
                    curr_H = max(1, curr_H // 2)
                    curr_W = max(1, curr_W // 2)
            
            heads = 8
            possible_heads = [8, 4, 2, 1]
            for h in possible_heads:
                if C % h == 0:
                    heads = h
                    break
            self.mid_attention = BottleneckAttention(C, num_heads=heads)
            for i in range(layers - 2, -1, -1):
                h_up, w_up = encoder_res[i]
                self.up_blocks.append(ConvLRUBlock(self.args, (C, h_up, w_up)))
                if self.arch_mode == "bifpn":
                    self.csa_blocks.append(BiFPNFusion(C))
                else:
                    self.csa_blocks.append(CrossScaleAttentionGate(C))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
            if self.arch_mode == "unet":
                self.fusion = nn.Conv3d(C * 2, C, 1)
            else:
                self.fusion = nn.Identity()
            self.convlru_blocks = None

    def forward(self, x: torch.Tensor, last_hidden_ins: Optional[List[torch.Tensor]] = None, listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            if self.convlru_blocks is None:
                raise RuntimeError("Model misconfigured")
            last_hidden_outs: List[torch.Tensor] = []
            for idx, blk in enumerate(self.convlru_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None and idx < len(last_hidden_ins)) else None
                x, h_out = blk(x, h_in, listT=listT, cond=cond, static_feats=static_feats)
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
            x, h_out = blk(x, hs_in_down[i], listT=listT, cond=curr_cond, static_feats=static_feats)
            last_hidden_outs.append(h_out)
            if i < len(self.down_blocks) - 1:
                skips.append(x)
                x_s = x
                if self.down_mode in ["shuffle", "avg", "conv"]:
                    pad_h = x_s.shape[-2] % 2
                    pad_w = x_s.shape[-1] % 2
                    if pad_h > 0 or pad_w > 0:
                        x_s = F.pad(x_s, (0, pad_w, 0, pad_h))
                if x_s.shape[-2] >= 2 and x_s.shape[-1] >= 2:
                    x_s = self.downsamples[i](x_s)
                x = x_s
        x = self.mid_attention(x)
        if self.upsample is None or self.fusion is None:
            raise RuntimeError("UNet misconfigured")
        for i, blk in enumerate(self.up_blocks):
            x_s = x
            x_s = self.upsample(x_s)
            x = x_s
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(-2) - x.size(-2)
                diffX = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            if self.arch_mode == "bifpn":
                x = self.csa_blocks[i](x, skip)
            else:
                skip = self.csa_blocks[i](skip, x)
                x = torch.cat([x, skip], dim=1)
                x = self.fusion(x)
            
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x_out, h_out = blk(x, hs_in_up[i], listT=listT, cond=curr_cond, static_feats=static_feats)
            x = x_out
            last_hidden_outs.append(h_out)
        return x, last_hidden_outs

class Embedding(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.input_ch = int(getattr(args, "input_ch", 1))
        self.input_size = tuple(getattr(args, "input_size", (64, 64)))
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.emb_hidden_ch = self.emb_ch
        self.static_ch = int(getattr(args, "static_ch", 0))
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        self.input_ch_total = self.input_ch + 4
        self.patch_embed = nn.Conv3d(
            self.input_ch_total, self.emb_hidden_ch,
            kernel_size=(1, self.rH + 2, self.rW + 2), stride=(1, self.rH, self.rW), padding=(0, 1, 1),
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
                nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.rH + 2, self.rW + 2), stride=(self.rH, self.rW), padding=(1, 1)),
                nn.SiLU(),
            )
        else:
            self.static_embed = None
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_size, kernel_size=7, use_cbam=False, cond_channels=cond_ch)])
        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.norm = RMSNorm(self.emb_ch)

    def _make_grid(self, args):
        H, W = tuple(getattr(args, "input_size", (64, 64)))
        lat = torch.linspace(-math.pi / 2, math.pi / 2, H)
        lon = torch.linspace(0, 2 * math.pi, W)
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
        for layer in self.c_hidden:
            x = layer(x, cond=cond)
        x = self.c_out(x)
        x = self.norm(x)
        return x, cond

class ConvLRU(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        self.revin = RevIN(int(getattr(args, "input_ch", 1)), affine=True)
        skip_contains = ["norm", "params_log", "prior", "post_ifft", "forcing", "dispersion", "dct_matrix", "grid_embed", "sobel"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains):
                    continue
                if n.endswith(".bias"):
                    p.zero_()
                elif p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def get_total_kl_loss(self):
        kl_sum = 0.0
        count = 0
        for module in self.modules():
            if hasattr(module, 'latest_kl'):
                kl_sum += module.latest_kl
                count += 1
        if count == 0:
            return torch.tensor(0.0)
        return kl_sum

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
            x_hid, _ = self.convlru_model(x_emb, listT=listT, cond=cond, static_feats=static_feats)
            out = self.decoder(x_hid, cond=cond, timestep=timestep)
            
            if isinstance(out, tuple):
                out_tensor = out[0]
                rest = out[1:]
            else:
                out_tensor = out
                rest = ()
            
            out_tensor = out_tensor.permute(0, 2, 1, 3, 4).contiguous()
            
            if self.decoder.head_mode == "gaussian":
                mu, sigma = torch.chunk(out_tensor, 2, dim=2)
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin(mu, "denorm")
                    stdev = torch.clamp(self.revin.stdev, min=1e-5)
                    sigma = sigma * stdev
                return torch.cat([mu, sigma], dim=2)
            elif self.decoder.head_mode == "token":
                return (out_tensor,) + rest
            else:
                if out_tensor.size(2) == self.revin.num_features:
                    return self.revin(out_tensor, "denorm")
                return out_tensor

        if out_gen_num is None or int(out_gen_num) <= 0:
            raise ValueError("out_gen_num must be positive for inference mode")
        B = x.size(0)
        if listT is None:
            listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT
            
        out_list: List[torch.Tensor] = []
        x_norm = self.revin(x, "norm")
        if self.revin.mean is not None and self.revin.mean.ndim == 5:
             self.revin.mean = self.revin.mean[:, -1:, :, :, :]
        if self.revin.stdev is not None and self.revin.stdev.ndim == 5:
             self.revin.stdev = self.revin.stdev[:, -1:, :, :, :]

        x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT0, cond=cond, static_feats=static_feats)
        x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
        
        if isinstance(x_dec, tuple):
            x_dec0 = x_dec[0]
        else:
            x_dec0 = x_dec
            
        x_step_dist = x_dec0.permute(0, 2, 1, 3, 4).contiguous()
        x_step_dist = x_step_dist[:, -1:, :, :, :]
        
        if str(self.decoder.head_mode).lower() == "gaussian":
            out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
            mu = x_step_dist[:, :, :out_ch, :, :]
            sigma = x_step_dist[:, :, out_ch:, :, :]
            
            if mu.size(2) == self.revin.num_features:
                mu_denorm = self.revin(mu, "denorm")
                sigma_denorm = sigma * self.revin.stdev
            else:
                mu_denorm = mu
                sigma_denorm = sigma
            out_list.append(torch.cat([mu_denorm, sigma_denorm], dim=2))
            x_step_mean = mu_denorm 
            
        elif str(self.decoder.head_mode).lower() == "token":
            out_list.append(x_step_dist)
            x_step_mean = x_step_dist 
        else:
            if x_step_dist.size(2) == self.revin.num_features:
                x_step_dist_denorm = self.revin(x_step_dist, "denorm")
            else:
                x_step_dist_denorm = x_step_dist
            out_list.append(x_step_dist_denorm)
            x_step_mean = x_step_dist_denorm

        future = listT_future
        if future is None:
            future = torch.ones(B, int(out_gen_num) - 1, device=x.device, dtype=x.dtype)
            
        for t in range(int(out_gen_num) - 1):
            dt = future[:, t : t + 1]
            
            curr_x = x_step_mean
            if curr_x.ndim == 5 and curr_x.shape[1] != 1:
                curr_x = curr_x[:, -1:, :, :, :]

            if curr_x.size(2) != self.embedding.input_ch:
                 B_in, L_in, C_out, H_in, W_in = curr_x.shape
                 C_target = self.embedding.input_ch
                 if C_out > C_target:
                      curr_x = curr_x[:, :, :C_target, :, :]
                 else:
                      diff = C_target - C_out
                      zeros = torch.zeros(B_in, L_in, diff, H_in, W_in, device=curr_x.device, dtype=curr_x.dtype)
                      curr_x = torch.cat([curr_x, zeros], dim=2)
            
            if curr_x.size(2) == self.revin.num_features:
                x_step_norm = self.revin(curr_x, "norm")
            else:
                x_step_norm = curr_x
                
            x_in, _ = self.embedding(x_step_norm, static_feats=static_feats)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt, cond=cond, static_feats=static_feats)
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            
            if isinstance(x_dec, tuple):
                x_dec0 = x_dec[0]
            else:
                x_dec0 = x_dec
            
            x_step_dist = x_dec0.permute(0, 2, 1, 3, 4).contiguous()
            x_step_dist = x_step_dist[:, -1:, :, :, :]
            
            if str(self.decoder.head_mode).lower() == "gaussian":
                out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
                mu = x_step_dist[:, :, :out_ch, :, :]
                sigma = x_step_dist[:, :, out_ch:, :, :]
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin(mu, "denorm")
                    stdev = torch.clamp(self.revin.stdev, min=1e-5)
                    sigma = sigma * stdev
                out_list.append(torch.cat([mu, sigma], dim=2))
                x_step_mean = mu
            elif str(self.decoder.head_mode).lower() == "token":
                out_list.append(x_step_dist)
                x_step_mean = x_step_dist
            else:
                if x_step_dist.size(2) == self.revin.num_features:
                    x_step_dist_denorm = self.revin(x_step_dist, "denorm")
                else:
                    x_step_dist_denorm = x_step_dist
                out_list.append(x_step_dist_denorm)
                x_step_mean = x_step_dist_denorm
                
        return torch.cat(out_list, dim=1)

