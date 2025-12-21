from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


def _is_fast_len(n: int) -> bool:
    if n <= 2:
        return True
    x = n
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def next_fast_len(n: int) -> int:
    if n <= 2:
        return n
    m = n
    while not _is_fast_len(m):
        m += 1
    return m


@dataclass(frozen=True)
class FFTPadSpec:
    H_in: int
    W_in: int
    H_fft: int
    W_fft: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


def make_fft_pad_spec(H: int, W: int) -> FFTPadSpec:
    H_fft = next_fast_len(H)
    W_fft = next_fast_len(W)
    pad_h = H_fft - H
    pad_w = W_fft - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return FFTPadSpec(
        H_in=H,
        W_in=W,
        H_fft=H_fft,
        W_fft=W_fft,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def pad_to_fft_friendly(x: torch.Tensor, spec: FFTPadSpec) -> torch.Tensor:
    return F.pad(x, (spec.pad_left, spec.pad_right, spec.pad_top, spec.pad_bottom), mode="reflect")


def crop_from_fft(x: torch.Tensor, spec: FFTPadSpec) -> torch.Tensor:
    pt, pl = spec.pad_top, spec.pad_left
    return x[..., pt : pt + spec.H_in, pl : pl + spec.W_in]


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


class FlashFFTConvInterface(nn.Module):
    def __init__(self, dim: int, size: Tuple[int, int]):
        super().__init__()
        self.dim = int(dim)
        self.H, self.W = int(size[0]), int(size[1])
        self.padded_H = next_fast_len(self.H)
        self.padded_W = next_fast_len(self.W)
        self.use_flash = False
        self.flash_conv = None
        try:
            from flash_fft_conv import FlashFFTConv  # type: ignore

            self.flash_conv = FlashFFTConv((self.H, self.W), dtype=torch.bfloat16)
            self.use_flash = True
        except Exception:
            self.use_flash = False
            self.flash_conv = None

    def forward(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.use_flash and self.flash_conv is not None:
            try:
                return self.flash_conv(u, k)
            except Exception:
                pass
        B, C, H, W = u.shape
        u_pad = F.pad(u, (0, self.padded_W - W, 0, self.padded_H - H))
        k_pad = F.pad(k, (0, self.padded_W - W, 0, self.padded_H - H))
        u_f = torch.fft.rfft2(u_pad.float())
        k_f = torch.fft.rfft2(k_pad.float())
        y_f = u_f * k_f
        y = torch.fft.irfft2(y_f, s=(self.padded_H, self.padded_W))
        return y[..., :H, :W].to(dtype=u.dtype)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("blcij,coij->bloij", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("SpectralConv2d expects complex input in frequency domain.")
        B, L, C, H, W = x.shape
        eff_m1 = min(H, self.modes1)
        eff_m2 = min(W, self.modes2)
        out = torch.zeros(B, L, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        w1 = self.weights1[:, :, :eff_m1, :eff_m2].to(device=x.device)
        out[:, :, :, :eff_m1, :eff_m2] = self._mul2d(x[:, :, :, :eff_m1, :eff_m2], w1)
        if H > 1 and (H > self.modes1 or W > self.modes2):
            eff_m1n = min(H, self.modes1)
            eff_m2p = min(W, self.modes2)
            x2 = x[:, :, :, -eff_m1n:, :eff_m2p]
            w2 = self.weights2[:, :, :eff_m1n, :eff_m2p].to(device=x.device)
            out[:, :, :, -eff_m1n:, :eff_m2p] = self._mul2d(x2, w2)
        return out


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
        theta, phi = self._latlon_to_spherical(self.H, self.W, device=torch.device("cpu"))
        Y = self._real_sph_harm_basis(theta, phi, self.Lmax)
        self.register_buffer("Y_real", Y, persistent=False)

    @staticmethod
    def _latlon_to_spherical(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = torch.linspace(math.pi / 2, -math.pi / 2, steps=H, device=device)
        lon = torch.linspace(-math.pi, math.pi, steps=W, device=device)
        theta = (math.pi / 2 - lat).unsqueeze(1).repeat(1, W)
        phi = lon.unsqueeze(0).repeat(H, 1)
        return theta, phi

    @staticmethod
    def _fact_ratio(l: int, m_abs: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        lt = torch.tensor(l, dtype=dtype, device=device)
        mt = torch.tensor(m_abs, dtype=dtype, device=device)
        return torch.exp(torch.lgamma(lt - mt + 1) - torch.lgamma(lt + mt + 1))

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
        P: list[list[Optional[torch.Tensor]]] = [[None] * (l + 1) for l in range(Lmax)]
        P[0][0] = one
        if Lmax >= 2:
            P[1][0] = x
        for l in range(2, Lmax):
            lf = torch.tensor(l, device=device, dtype=dtype)
            P[l][0] = ((2 * lf - 1) * x * P[l - 1][0] - (lf - 1) * P[l - 2][0]) / lf
        for m in range(1, Lmax):
            mf = torch.tensor(m, device=device, dtype=dtype)
            P_mm = ((-1) ** m) * SphericalHarmonicsPrior._double_factorial(2 * m - 1, dtype, device) * (
                1 - x * x
            ).pow(mf / 2)
            P[m][m] = P_mm
            if m + 1 < Lmax:
                P[m + 1][m] = (2 * mf + 1) * x * P_mm
            for l in range(m + 2, Lmax):
                lf = torch.tensor(l, device=device, dtype=dtype)
                P[l][m] = ((2 * lf - 1) * x * P[l - 1][m] - (lf + mf - 1) * P[l - 2][m]) / (lf - mf)
        idx = torch.arange(0, Lmax, device=device, dtype=dtype).view(-1, 1, 1)
        cos_mphi = torch.cos(idx * phi)
        sin_mphi = torch.sin(idx * phi)
        Ys: list[torch.Tensor] = []
        for l in range(Lmax):
            lf = torch.tensor(l, device=device, dtype=dtype)
            for m in range(-l, l + 1):
                m_abs = abs(m)
                N_lm = torch.sqrt(
                    (2 * lf + 1) / (4 * pi) * SphericalHarmonicsPrior._fact_ratio(l, m_abs, dtype, device)
                )
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
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
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
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class CBAM2DPerStep(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        x = x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return x


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], use_cbam: bool = False, cond_channels: int = 0):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.channels = int(channels)
        self.hidden_size = (int(hidden_size[0]), int(hidden_size[1]))
        self.dw_conv = nn.Conv3d(self.channels, self.channels, kernel_size=(1, 7, 7), padding="same", groups=self.channels)
        self.norm = nn.LayerNorm([self.hidden_size[0], self.hidden_size[1]])
        self.cond_channels = int(cond_channels)
        self.cond_proj = nn.Conv3d(self.cond_channels, self.channels * 2, kernel_size=1) if self.cond_channels > 0 else None
        self.pw_conv_in = nn.Conv3d(self.channels, self.channels * 2, kernel_size=1)
        self.act = nn.SiLU()
        self.pw_conv_out = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        self.cbam = CBAM2DPerStep(self.channels, reduction=16) if self.use_cbam else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4)
        if self.cond_proj is not None and cond is not None:
            if cond.dim() == 4:
                cond_in = cond.unsqueeze(2)
            else:
                cond_in = cond
            if cond_in.shape[-2:] != x.shape[-2:]:
                cond_in = F.interpolate(
                    cond_in.squeeze(2),
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(2)
            affine = self.cond_proj(cond_in)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta
        x = self.pw_conv_in(x)
        x_a, x_g = torch.chunk(x, 2, dim=1)
        x = self.act(x_a) * x_g
        if self.cbam is not None:
            x = self.cbam(x)
        x = self.pw_conv_out(x)
        return residual + x


class BatchedGatedConvExperts(nn.Module):
    def __init__(self, num_experts: int, channels: int, kernel_hw: int = 7, cond_channels: int = 0):
        super().__init__()
        self.E = int(num_experts)
        self.C = int(channels)
        self.cond_channels = int(cond_channels)
        k = int(kernel_hw)
        pad = k // 2
        self.pad = pad

        self.dw_weight = nn.Parameter(torch.empty(self.E * self.C, 1, 1, k, k))
        self.dw_bias = nn.Parameter(torch.zeros(self.E * self.C))
        nn.init.kaiming_normal_(self.dw_weight, a=0, mode="fan_in", nonlinearity="relu")

        self.gn = nn.GroupNorm(num_groups=self.E, num_channels=self.E * self.C, eps=1e-5, affine=True)

        self.pw_in_weight = nn.Parameter(torch.empty(self.E * (2 * self.C), self.C, 1, 1, 1))
        self.pw_in_bias = nn.Parameter(torch.zeros(self.E * (2 * self.C)))
        nn.init.kaiming_normal_(self.pw_in_weight, a=0, mode="fan_in", nonlinearity="relu")

        self.pw_out_weight = nn.Parameter(torch.empty(self.E * self.C, self.C, 1, 1, 1))
        self.pw_out_bias = nn.Parameter(torch.zeros(self.E * self.C))
        nn.init.kaiming_normal_(self.pw_out_weight, a=0, mode="fan_in", nonlinearity="relu")

        self.act = nn.SiLU()

        if self.cond_channels > 0:
            self.cond_proj_weight = nn.Parameter(torch.empty(self.E * (2 * self.C), self.cond_channels, 1, 1, 1))
            self.cond_proj_bias = nn.Parameter(torch.zeros(self.E * (2 * self.C)))
            nn.init.kaiming_normal_(self.cond_proj_weight, a=0, mode="fan_in", nonlinearity="relu")
        else:
            self.cond_proj_weight = None
            self.cond_proj_bias = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, C, L, P, P2 = x.shape
        if C != self.C or P != P2:
            raise ValueError("Invalid patch shape.")
        x_rep = x.unsqueeze(1).expand(N, self.E, C, L, P, P).reshape(N, self.E * C, L, P, P)
        y = F.conv3d(
            x_rep,
            self.dw_weight,
            self.dw_bias,
            stride=1,
            padding=(0, self.pad, self.pad),
            groups=self.E * self.C,
        )
        y = self.gn(y)
        if self.cond_channels > 0 and cond is not None:
            if cond.dim() != 5:
                raise ValueError("cond must be [N, condC, L, P, P]")
            if cond.shape[0] != N or cond.shape[1] != self.cond_channels:
                raise ValueError("cond shape mismatch")
            cond_rep = cond.unsqueeze(1).expand(N, self.E, self.cond_channels, L, P, P).reshape(
                N, self.E * self.cond_channels, L, P, P
            )
            affine = F.conv3d(
                cond_rep,
                self.cond_proj_weight,
                self.cond_proj_bias,
                stride=1,
                padding=0,
                groups=self.E,
            )
            gamma, beta = affine.chunk(2, dim=1)
            y = y * (1 + gamma) + beta
        y = F.conv3d(
            y,
            self.pw_in_weight,
            self.pw_in_bias,
            stride=1,
            padding=0,
            groups=self.E,
        )
        y_a, y_g = y.chunk(2, dim=1)
        y = self.act(y_a) * y_g
        y = F.conv3d(
            y,
            self.pw_out_weight,
            self.pw_out_bias,
            stride=1,
            padding=0,
            groups=self.E,
        )
        y = y.view(N, self.E, self.C, L, P, P)
        return x.unsqueeze(1) + y


class SpatialPatchMoE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int,
        active_experts: int,
        patch_size: int = 8,
        cond_channels: int = 0,
        router_use_cond: bool = True,
    ):
        super().__init__()
        self.C = int(channels)
        self.E = int(num_experts)
        self.K = int(active_experts)
        self.P = int(patch_size)
        self.cond_channels = int(cond_channels)
        self.router_use_cond = bool(router_use_cond)

        router_in = self.C + (self.cond_channels if (self.cond_channels > 0 and self.router_use_cond) else 0)
        self.router = nn.Linear(router_in, self.E, bias=True)
        self.experts = BatchedGatedConvExperts(num_experts=self.E, channels=self.C, kernel_hw=7, cond_channels=self.cond_channels)

    @staticmethod
    def _pad_hw(x: torch.Tensor, P: int) -> Tuple[torch.Tensor, int, int]:
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L, H, W = x.shape
        if C != self.C:
            raise ValueError("channels mismatch")
        x_pad, pad_h, pad_w = self._pad_hw(x, self.P)
        Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]
        nH, nW = Hp // self.P, Wp // self.P

        x_p = (
            x_pad.view(B, C, L, nH, self.P, nW, self.P)
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(B * nH * nW, C, L, self.P, self.P)
        )

        router_feat = x_p.mean(dim=(2, 3, 4))
        cond_p = None

        if self.cond_channels > 0 and cond is not None:
            if cond.dim() == 4:
                cond_2d = cond
                if cond_2d.shape[-2:] != (Hp, Wp):
                    cond_2d = F.interpolate(cond_2d, size=(Hp, Wp), mode="bilinear", align_corners=False)
                cond_ = cond_2d.unsqueeze(2).expand(B, self.cond_channels, L, Hp, Wp)
            elif cond.dim() == 5:
                cond_ = cond
                if cond_.shape[-2:] != (Hp, Wp):
                    cond_ = F.interpolate(cond_.reshape(B * L, self.cond_channels, cond_.shape[-2], cond_.shape[-1]),
                                         size=(Hp, Wp), mode="bilinear", align_corners=False).view(B, L, self.cond_channels, Hp, Wp).permute(0, 2, 1, 3, 4)
            else:
                raise ValueError("cond must be 4D or 5D")

            cond_p = (
                cond_.view(B, self.cond_channels, L, nH, self.P, nW, self.P)
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(B * nH * nW, self.cond_channels, L, self.P, self.P)
            )

            if self.router_use_cond:
                router_cond = cond_p.mean(dim=(2, 3, 4))
                router_in = torch.cat([router_feat, router_cond], dim=1)
            else:
                router_in = router_feat
        else:
            router_in = router_feat

        logits = self.router(router_in)

        if self.K >= self.E:
            weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        else:
            topk_val, topk_idx = torch.topk(logits, self.K, dim=-1)
            topk_w = torch.softmax(topk_val.float(), dim=-1).to(dtype=logits.dtype)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, topk_idx, topk_w)

        y_all = self.experts(x_p, cond=cond_p)
        y = torch.einsum("ne,neclpp->nclpp", weights, y_all)

        out = (
            y.view(B, nH, nW, C, L, self.P, self.P)
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, C, L, Hp, Wp)
        )
        if pad_h or pad_w:
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
        S, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])
        self.hidden_size = (S, W)
        self.rank = int(getattr(args, "lru_rank", min(S, W, 32)))
        self.is_selective = bool(getattr(args, "use_selective", False))
        self.bidirectional = bool(getattr(args, "bidirectional", False))

        self.flash_fft = FlashFFTConvInterface(self.emb_ch, (S, W))

        u1 = torch.rand(self.emb_ch, self.rank)
        u2 = torch.rand(self.emb_ch, self.rank)
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2))
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

        self.U_row = nn.Parameter(torch.randn(self.emb_ch, S, self.rank, dtype=torch.cfloat) / math.sqrt(S))
        self.V_col = nn.Parameter(torch.randn(self.emb_ch, W, self.rank, dtype=torch.cfloat) / math.sqrt(W))

        C = self.emb_ch
        self.proj_W = nn.Parameter(torch.randn(C, C, dtype=torch.cfloat) / math.sqrt(C))
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat)) if self.use_bias else None

        self.post_ifft_conv_real = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.post_ifft_conv_imag = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        out_dim_fusion = self.emb_ch * 2 if not self.bidirectional else self.emb_ch * 4
        self.post_ifft_proj = nn.Conv3d(out_dim_fusion, self.emb_ch, kernel_size=(1, 1, 1), padding="same")

        self.layer_norm = nn.LayerNorm([self.hidden_size[0], self.hidden_size[1]])
        self.noise_level = nn.Parameter(torch.tensor(0.01))

        self.freq_prior: Optional[SpectralConv2d] = (
            SpectralConv2d(self.emb_ch, self.emb_ch, 8, 8) if bool(getattr(args, "use_freq_prior", False)) else None
        )
        self.sh_prior: Optional[SphericalHarmonicsPrior] = (
            SphericalHarmonicsPrior(self.emb_ch, S, W, Lmax=int(getattr(args, "sh_Lmax", 6)))
            if bool(getattr(args, "use_sh_prior", False))
            else None
        )

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
        return dnu.unsqueeze(-1), dth.unsqueeze(-1)

    def _fft_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, FFTPadSpec]:
        B, L, C, S, W = x.shape
        spec = make_fft_pad_spec(S, W)
        x2 = x.reshape(B * L, C, S, W)
        x2 = pad_to_fft_friendly(x2, spec)
        x2 = x2.view(B, L, C, spec.H_fft, spec.W_fft)
        h = torch.fft.fft2(x2.to(torch.float32).to(torch.cfloat), dim=(-2, -1), norm="ortho")
        return h, spec

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor],
        listT: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=torch.float32)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=torch.float32)

        h, spec = self._fft_impl(x)
        h_perm = h.permute(0, 1, 3, 4, 2).contiguous().view(-1, C)
        h_proj = torch.matmul(h_perm, self.proj_W.to(device=h_perm.device))
        h = h_proj.view(B, L, spec.H_fft, spec.W_fft, C).permute(0, 1, 4, 2, 3).contiguous()
        if self.proj_b is not None:
            h = h + self.proj_b.view(1, 1, C, 1, 1)

        h_spatial = torch.fft.ifft2(h, dim=(-2, -1), norm="ortho")
        h_spatial = crop_from_fft(h_spatial, spec)
        h = torch.fft.fft2(h_spatial, dim=(-2, -1), norm="ortho")

        if self.freq_prior is not None:
            h = h + self.freq_prior(h)

        Uc = self.U_row.conj()
        t = torch.matmul(h.permute(0, 1, 2, 4, 3), Uc)
        t = t.permute(0, 1, 2, 4, 3)
        zq = torch.matmul(t, self.V_col)

        nu_log, theta_log = self.params_log_base.unbind(dim=0)
        disp_nu, disp_th = self.dispersion_mod.unbind(dim=0)
        nu_base = torch.exp((nu_log + disp_nu).float()).view(1, 1, C, self.rank, 1)
        th_base = torch.exp((theta_log + disp_th).float()).view(1, 1, C, self.rank, 1)

        dnu_force, dth_force = self._apply_forcing(x, dt)
        nu_t = torch.clamp(nu_base * dt + dnu_force.float(), min=1e-6)
        th_t = th_base * dt + dth_force.float()
        lamb = torch.exp(torch.complex(-nu_t, th_t)).to(dtype=torch.cfloat)

        if self.training:
            noise_std = (self.noise_level.float() * torch.sqrt(dt + 1e-6)).to(dtype=zq.real.dtype)
            noise = torch.randn_like(zq.real).to(dtype=torch.float32) * noise_std
            x_in = zq + noise.to(dtype=zq.dtype)
        else:
            x_in = zq

        gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t), min=1e-12)).to(dtype=torch.float32)
        x_in = x_in * gamma_t.to(dtype=x_in.dtype)

        zero_prev = torch.zeros_like(x_in[:, :1])
        if last_hidden_in is not None:
            x_in_fwd = torch.cat([last_hidden_in, x_in], dim=1)
        else:
            x_in_fwd = torch.cat([zero_prev, x_in], dim=1)

        lamb_fwd = torch.cat([lamb[:, :1], lamb], dim=1)
        lamb_in_fwd = lamb_fwd.expand_as(x_in_fwd).contiguous()

        z_out = self.pscan(lamb_in_fwd, x_in_fwd.contiguous())[:, 1:]
        last_hidden_out = z_out[:, -1:]

        if self.bidirectional:
            x_in_bwd = x_in.flip(1)
            lamb_bwd = lamb.flip(1)
            x_in_bwd = torch.cat([zero_prev, x_in_bwd], dim=1)
            lamb_bwd = torch.cat([lamb_bwd[:, :1], lamb_bwd], dim=1)
            lamb_in_bwd = lamb_bwd.expand_as(x_in_bwd).contiguous()
            z_out_bwd = self.pscan(lamb_in_bwd, x_in_bwd.contiguous())[:, 1:].flip(1)
        else:
            linking = None
            z_out_bwd = Linking  # type: ignore

        def project_back(z: torch.Tensor) -> torch.Tensor:
            t2 = torch.matmul(z, self.V_col.conj().transpose(1, 2))
            t2 = t2.permute(0, 1, 2, 4, 3)
            return torch.matmul(t2, self.U_row.transpose(1, 2)).permute(0, 1, 2, 4, 3)

        h_rec_fwd = project_back(z_out)

        def recover_spatial(h_rec: torch.Tensor) -> torch.Tensor:
            h_sp = torch.fft.ifft2(h_rec, dim=(-2, -1), norm="ortho")
            h_sp = crop_from_fft(h_sp, make_fft_pad_spec(h_sp.shape[-2], h_sp.shape[-1]))
            hr = self.post_ifft_conv_real(h_sp.real.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            hi = self.post_ifft_conv_imag(h_sp.imag.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            return torch.cat([hr, hi], dim=2)

        feat_fwd = recover_spatial(h_rec_fwd)

        if self.bidirectional:
            h_rec_bwd = project_back(z_out_bwd)
            feat_bwd = recover_spatial(h_rec_bwd)
            feat_final = torch.cat([feat_fwd, feat_bwd], dim=2)
        else:
            feat_final = feat_fwd

        h_final = self.post_ifft_proj(feat_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        if self.sh_prior is not None:
            h_final = self.sh_prior(h_final)

        h_final = self.layer_norm(h_final)

        if self.gate_conv is not None:
            gate = self.gate_conv(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x_out = (1 - gate) * x + gate * h_final
        else:
            x_out = x + h_final

        return x_out, last_hidden_out


class FeedForward(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.ffn_hidden_ch = int(getattr(args, "ffn_hidden_ch", 32))
        self.layers_num = int(getattr(args, "ffn_hidden_layers_num", 1))
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.num_expert = int(getattr(args, "num_expert", -1))
        self.activate_expert = int(getattr(args, "activate_expert", 2))

        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

        blocks: list[nn.Module] = []
        cond_ch = self.emb_ch if self.static_ch > 0 else 0
        for _ in range(self.layers_num):
            if self.num_expert > 1:
                blocks.append(
                    SpatialPatchMoE(
                        channels=self.ffn_hidden_ch,
                        num_experts=self.num_expert,
                        active_experts=self.activate_expert,
                        patch_size=8,
                        cond_channels=cond_ch,
                        router_use_cond=True,
                    )
                )
            else:
                blocks.append(GatedConvBlock(self.ffn_hidden_ch, (int(input_downsp_shape[1]), int(input_downsp_shape[2])), use_cbam=self.use_cbam, cond_channels=cond_ch))
        self.blocks = nn.ModuleList(blocks)
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        y = self.c_in(x.permute(0, 2, 1, 3, 4))
        y = self.act(y)
        for block in self.blocks:
            if isinstance(block, SpatialPatchMoE):
                y = block(y, cond=cond)
            else:
                y = block(y, cond=cond)
        y = self.c_out(y)
        y = y.permute(0, 2, 1, 3, 4)
        return residual + y


class ConvLRUBlock(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor],
        listT: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, last_hidden_out = self.lru_layer(x, last_hidden_in, listT=listT)
        x = self.feed_forward(x, cond=cond)
        return x, last_hidden_out


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L, H, W = inputs.shape
        if C != self.embedding_dim:
            raise ValueError(f"VQ dim mismatch: inputs.C={C} vs embedding_dim={self.embedding_dim}")
        flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        flat_f = flat.float()
        codebook_f = self.embedding.weight.float()
        dist = torch.cdist(flat_f, codebook_f, p=2)
        indices = torch.argmin(dist, dim=1)
        onehot = F.one_hot(indices, num_classes=self.num_embeddings).to(dtype=flat.dtype, device=flat.device)
        quant = onehot @ self.embedding.weight
        quant = quant.view(B, L, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        e_latent = F.mse_loss(quant.detach(), inputs)
        q_latent = F.mse_loss(quant, inputs.detach())
        loss = q_latent + self.commitment_cost * e_latent
        quant_st = inputs + (quant - inputs).detach()
        return quant_st, loss, indices.view(-1, 1)


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
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1))
        )
        emb = t[:, None] * freqs[None, :]
        out = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(out)


class Decoder(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.head_mode: Literal["gaussian", "diffusion", "token"] = getattr(args, "head_mode", "gaussian")
        self.output_ch = int(getattr(args, "out_ch", 1))
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.dec_hidden_ch = int(getattr(args, "dec_hidden_ch", 0))
        self.dec_hidden_layers_num = int(getattr(args, "dec_hidden_layers_num", 0))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.dec_strategy: Literal["pxsf", "deconv"] = getattr(args, "dec_strategy", "pxsf")
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])

        H0, W0 = int(input_downsp_shape[1]), int(input_downsp_shape[2])
        self.hidden_size = (H0, W0)

        if self.dec_hidden_layers_num != 0:
            self.dec_hidden_ch = int(getattr(args, "dec_hidden_ch", self.emb_ch))
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
            self.upsp = None
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

        if self.dec_hidden_layers_num != 0:
            H = self.hidden_size[0] * self.rH
            W = self.hidden_size[1] * self.rW
            cond_channels = self.emb_ch if self.static_ch > 0 else 0
            self.c_hidden = nn.ModuleList(
                [GatedConvBlock(out_ch_after_up, (H, W), use_cbam=False, cond_channels=cond_channels) for _ in range(self.dec_hidden_layers_num)]
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

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        x = x.permute(0, 2, 1, 3, 4)
        if self.dec_strategy == "deconv":
            x = self.upsp(x)
        else:
            x = self.pre_shuffle_conv(x)
            x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        if self.head_mode == "diffusion":
            if timestep is None:
                raise ValueError("Diffusion head requires timestep.")
            t_emb = self.time_embed(timestep)
            x = x + t_emb.view(x.size(0), x.size(1), 1, 1, 1)
        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=cond)
        x = self.c_out(x)
        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            sigma = F.softplus(log_sigma.float()).to(dtype=x.dtype) + 1e-6
            return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4)
        if self.head_mode == "token":
            quantized, loss, indices = self.vq(x)
            return quantized.permute(0, 2, 1, 3, 4), loss, indices
        return x.permute(0, 2, 1, 3, 4)


class ConvLRUModel(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.use_unet = bool(getattr(args, "unet", False))
        layers = int(getattr(args, "convlru_num_blocks", 2))
        C = int(getattr(args, "emb_ch", input_downsp_shape[0]))
        H, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])

        if not self.use_unet:
            self.convlru_blocks = nn.ModuleList([ConvLRUBlock(args, (C, H, W)) for _ in range(layers)])
            self.down_blocks = None
            self.up_blocks = None
            self.skip_convs = None
            self.upsample = None
            self.fusion = None
            return

        self.convlru_blocks = None
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        curr_H, curr_W = H, W
        encoder_res: list[Tuple[int, int]] = []
        for i in range(layers):
            self.down_blocks.append(ConvLRUBlock(args, (C, curr_H, curr_W)))
            encoder_res.append((curr_H, curr_W))
            if i < layers - 1:
                self.skip_convs.append(nn.Identity())
                curr_H //= 2
                curr_W //= 2
        for i in range(layers - 2, -1, -1):
            h_up, w_up = encoder_res[i]
            self.up_blocks.append(ConvLRUBlock(args, (C, h_up, w_up)))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.fusion = nn.Conv3d(C * 2, C, 1)

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_ins: Optional[Sequence[Optional[torch.Tensor]]] = None,
        listT: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        if not self.use_unet:
            assert self.convlru_blocks is not None
            last_hidden_outs: list[torch.Tensor] = []
            for idx, block in enumerate(self.convlru_blocks):
                h_in = None if last_hidden_ins is None else last_hidden_ins[idx]
                x, h_out = block(x, h_in, listT=listT, cond=cond)
                last_hidden_outs.append(h_out)
            return x, last_hidden_outs

        assert self.down_blocks is not None and self.up_blocks is not None and self.upsample is not None and self.fusion is not None
        skips: list[torch.Tensor] = []
        last_hidden_outs: list[torch.Tensor] = []

        if last_hidden_ins is not None:
            num_down = len(self.down_blocks)
            h_in_down = last_hidden_ins[:num_down]
            h_in_up = last_hidden_ins[num_down:]
        else:
            h_in_down = [None] * len(self.down_blocks)
            h_in_up = [None] * len(self.up_blocks)

        for idx, block in enumerate(self.down_blocks):
            h_in = h_in_down[idx]
            curr_cond = cond
            if cond is not None:
                target_size = x.shape[-2:]
                if cond.shape[-2:] != target_size:
                    curr_cond = F.interpolate(cond, size=target_size, mode="bilinear", align_corners=False)
            x, h_out = block(x, h_in, listT=listT, cond=curr_cond)
            last_hidden_outs.append(h_out)
            if idx < len(self.down_blocks) - 1:
                skips.append(x)
                x_s = x.permute(0, 2, 1, 3, 4)
                x_s = F.avg_pool3d(x_s, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                x = x_s.permute(0, 2, 1, 3, 4).contiguous()

        for idx, block in enumerate(self.up_blocks):
            x_s = x.permute(0, 2, 1, 3, 4)
            x_s = self.upsample(x_s)
            x = x_s.permute(0, 2, 1, 3, 4).contiguous()
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(-2) - x.size(-2)
                diffX = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=2)
            x = self.fusion(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()
            curr_cond = cond
            if cond is not None:
                target_size = x.shape[-2:]
                if cond.shape[-2:] != target_size:
                    curr_cond = F.interpolate(cond, size=target_size, mode="bilinear", align_corners=False)
            h_in = h_in_up[idx]
            x, h_out = block(x, h_in, listT=listT, cond=curr_cond)
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

        self.patch_embed = nn.Conv3d(
            self.input_ch,
            self.emb_hidden_ch,
            kernel_size=(1, self.rH + 2, self.rW + 2),
            stride=(1, self.rH, self.rW),
            padding=(0, 1, 1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch, 1, int(self.input_size[0]), int(self.input_size[1]))
            out_dummy = self.patch_embed(dummy)
            _, _, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, H, W)

        self.hidden_size = (int(self.input_downsp_shape[1]), int(self.input_downsp_shape[2]))

        self.static_embed = (
            nn.Sequential(
                nn.Conv2d(
                    self.static_ch,
                    self.emb_ch,
                    kernel_size=(self.rH + 2, self.rW + 2),
                    stride=(self.rH, self.rW),
                    padding=(1, 1),
                ),
                nn.SiLU(),
            )
            if self.static_ch > 0
            else None
        )

        if self.emb_hidden_layers_num > 0:
            cond_channels = self.emb_ch if self.static_ch > 0 else 0
            self.c_hidden = nn.ModuleList(
                [
                    GatedConvBlock(
                        self.emb_hidden_ch,
                        self.hidden_size,
                        use_cbam=False,
                        cond_channels=cond_channels,
                    )
                    for _ in range(self.emb_hidden_layers_num)
                ]
            )
        else:
            self.c_hidden = None

        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm([self.hidden_size[0], self.hidden_size[1]])

    def forward(self, x: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x)
        x = self.activation(x)

        cond = None
        if self.static_ch > 0 and self.static_embed is not None and static_feats is not None:
            cond = self.static_embed(static_feats)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=cond)

        x = self.c_out(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.layer_norm(x)
        return x, cond


class ConvLRU(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.embedding = Embedding(args)
        self.convlru_model = ConvLRUModel(args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(args, self.embedding.input_downsp_shape)

        skip_contains = ["layer_norm", "params_log", "prior", "post_ifft", "forcing", "dispersion"]
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
        mode: Literal["p", "i"] = "p",
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
            x_emb, _ = self.embedding(x, static_feats=static_feats)
            x_hid, _ = self.convlru_model(x_emb, listT=listT, cond=cond)
            return self.decoder(x_hid, cond=cond, timestep=timestep)

        if out_gen_num is None or out_gen_num <= 0:
            raise ValueError("mode='i' requires out_gen_num > 0")

        if listT is None:
            listT = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)

        out: list[torch.Tensor] = []

        x_emb, _ = self.embedding(x, static_feats=static_feats)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT, cond=cond)
        x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
        if isinstance(x_dec, tuple):
            x_dec_main = x_dec[0]
        else:
            x_dec_main = x_dec

        x_step_dist = x_dec_main[:, -1:]
        if self.decoder.head_mode == "gaussian":
            x_step_mean = x_step_dist[..., : int(getattr(self.args, "out_ch", x_step_dist.shape[2] // 2)), :, :]
        else:
            x_step_mean = x_step_dist
        out.append(x_step_dist)

        if listT_future is None:
            listT_future = torch.ones(x.size(0), out_gen_num - 1, device=x.device, dtype=x.dtype)

        for t in range(out_gen_num - 1):
            dt = listT_future[:, t : t + 1]
            x_in, _ = self.embedding(x_step_mean, static_feats=None)
            x_hidden, last_hidden_outs = self.convlru_model(
                x_in,
                last_hidden_ins=last_hidden_outs,
                listT=dt,
                cond=cond,
            )
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            if isinstance(x_dec, tuple):
                x_dec_main = x_dec[0]
            else:
                x_dec_main = x_dec
            x_step_dist = x_dec_main[:, -1:]
            if self.decoder.head_mode == "gaussian":
                x_step_mean = x_step_dist[..., : int(getattr(self.args, "out_ch", x_step_dist.shape[2] // 2)), :, :]
            else:
                x_step_mean = x_step_dist
            out.append(x_step_dist)

        return torch.cat(out, dim=1)
