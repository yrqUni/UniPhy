import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


def next_fast_len(n: int) -> int:
    if n <= 2:
        return 2
    def is_fast(m: int) -> bool:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        return x == 1
    m = n
    while not is_fast(m):
        m += 1
    return m


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
        self.use_flash = False
        try:
            from flash_fft_conv import FlashFFTConv  # type: ignore

            self.flash_conv = FlashFFTConv(size, dtype=torch.bfloat16)
            self.use_flash = True
        except Exception:
            self.flash_conv = None

    def forward(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.use_flash and self.flash_conv is not None:
            return self.flash_conv(u, k)
        B, C, H, W = u.shape
        Hp = next_fast_len(H)
        Wp = next_fast_len(W)
        u_pad = F.pad(u, (0, Wp - W, 0, Hp - H))
        k_pad = F.pad(k, (0, Wp - W, 0, Hp - H))
        u_f = torch.fft.rfft2(u_pad.float(), norm="ortho")
        k_f = torch.fft.rfft2(k_pad.float(), norm="ortho")
        y_f = u_f * k_f
        y = torch.fft.irfft2(y_f, s=(Hp, Wp), norm="ortho")
        return y[..., :H, :W].to(dtype=u.dtype)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("blcxy,coxy->bloxy", input_ft, weights)

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x_ft):
            raise TypeError("SpectralConv2d expects complex input")
        B, L, C, H, W = x_ft.shape
        out = torch.zeros(B, L, self.out_channels, H, W, device=x_ft.device, dtype=torch.cfloat)
        m1 = min(H, self.modes1)
        m2 = min(W, self.modes2)
        out[:, :, :, :m1, :m2] = self._mul(x_ft[:, :, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        if m1 > 0 and m2 > 0:
            out[:, :, :, -m1:, :m2] = self._mul(x_ft[:, :, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
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
        self.register_buffer("Y_real", Y)

    @staticmethod
    def _latlon_to_spherical(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = torch.linspace(math.pi / 2, -math.pi / 2, steps=H, device=device)
        lon = torch.linspace(-math.pi, math.pi, steps=W, device=device)
        theta = (math.pi / 2 - lat).unsqueeze(1).repeat(1, W)
        phi = lon.unsqueeze(0).repeat(H, 1)
        return theta, phi

    @staticmethod
    def _fact_ratio(l: int, m_abs: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        lt = torch.tensor(float(l), dtype=dtype, device=device)
        mt = torch.tensor(float(m_abs), dtype=dtype, device=device)
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

        P: List[List[Optional[torch.Tensor]]] = [[None] * (l + 1) for l in range(Lmax)]
        P[0][0] = one
        if Lmax >= 2:
            P[1][0] = x
        for l in range(2, Lmax):
            lf = torch.tensor(float(l), device=device, dtype=dtype)
            P[l][0] = ((2 * lf - 1) * x * P[l - 1][0] - (lf - 1) * P[l - 2][0]) / lf

        for m in range(1, Lmax):
            mf = torch.tensor(float(m), device=device, dtype=dtype)
            P_mm = ((-1.0) ** m) * SphericalHarmonicsPrior._double_factorial(2 * m - 1, dtype, device) * (1 - x * x).pow(mf / 2)
            P[m][m] = P_mm
            if m + 1 < Lmax:
                P[m + 1][m] = (2 * mf + 1) * x * P_mm
            for l in range(m + 2, Lmax):
                lf = torch.tensor(float(l), device=device, dtype=dtype)
                P[l][m] = ((2 * lf - 1) * x * P[l - 1][m] - (lf + mf - 1) * P[l - 2][m]) / (lf - mf)

        idx = torch.arange(0, Lmax, device=device, dtype=dtype).view(-1, 1, 1)
        cos_mphi = torch.cos(idx * phi)
        sin_mphi = torch.sin(idx * phi)

        Ys: List[torch.Tensor] = []
        for l in range(Lmax):
            lf = torch.tensor(float(l), device=device, dtype=dtype)
            for m in range(-l, l + 1):
                m_abs = abs(m)
                N_lm = torch.sqrt((2 * lf + 1) / (4 * pi) * SphericalHarmonicsPrior._fact_ratio(l, m_abs, dtype, device))
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
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BL, C, H, W = x.shape
        avg_pool = torch.mean(x, dim=(2, 3))
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
        return x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_hw: Tuple[int, int], use_cbam: bool = False, cond_channels: Optional[int] = None):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.dw_conv = nn.Conv3d(channels, channels, kernel_size=(1, 7, 7), padding="same", groups=channels)
        self.norm = nn.LayerNorm([int(hidden_hw[0]), int(hidden_hw[1])])
        self.cond_channels = int(cond_channels) if cond_channels is not None else 0
        self.cond_proj = nn.Conv3d(self.cond_channels, channels * 2, kernel_size=1) if self.cond_channels > 0 else None
        self.pw_conv_in = nn.Conv3d(channels, channels * 2, kernel_size=1)
        self.act = nn.SiLU()
        self.pw_conv_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.cbam = CBAM2DPerStep(channels, reduction=16) if self.use_cbam else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4)

        if self.cond_proj is not None and cond is not None:
            cond_in = cond.unsqueeze(2) if cond.dim() == 4 else cond
            if cond_in.shape[-2:] != x.shape[-2:]:
                cond_in = F.interpolate(cond_in.squeeze(2), size=x.shape[-2:], mode="bilinear", align_corners=False).unsqueeze(2)
            affine = self.cond_proj(cond_in)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta

        x = self.pw_conv_in(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.act(x) * gate

        if self.cbam is not None:
            x = self.cbam(x)

        x = self.pw_conv_out(x)
        return residual + x


class SpatialPatchMoE(nn.Module):
    def __init__(self, channels: int, hidden_hw: Tuple[int, int], num_experts: int, active_experts: int, use_cbam: bool, cond_channels: Optional[int]):
        super().__init__()
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.patch_size = 8
        P = self.patch_size
        self.expert_hidden_hw = (P, P)
        self.cond_channels = int(cond_channels) if cond_channels is not None else 0

        self.experts = nn.ModuleList(
            [GatedConvBlock(channels, self.expert_hidden_hw, use_cbam=use_cbam, cond_channels=self.cond_channels if self.cond_channels > 0 else None) for _ in range(self.num_experts)]
        )

        router_in = int(channels) + (self.cond_channels if self.cond_channels > 0 else 0)
        self.router = nn.Linear(router_in, self.num_experts)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L, H, W = x.shape
        P = self.patch_size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            if cond is not None:
                cond = F.pad(cond, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[-2:]
        nH, nW = H_pad // P, W_pad // P

        x_patches = x.view(B, C, L, nH, P, nW, P).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, C, L, P, P)
        router_x = x_patches.mean(dim=(2, 3, 4))

        cond_patches = None
        if cond is not None:
            if cond.dim() == 4:
                cond_in = cond.unsqueeze(2).expand(-1, -1, L, -1, -1)
            else:
                cond_in = cond
            cond_patches = cond_in.view(B, cond_in.size(1), L, nH, P, nW, P).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, cond_in.size(1), L, P, P)
            router_c = cond_patches.mean(dim=(2, 3, 4))
            router_in = torch.cat([router_x, router_c], dim=1)
        else:
            router_in = router_x

        logits = self.router(router_in)
        topk_logits, topk_idx = torch.topk(logits, self.active_experts, dim=1)
        topk_w = F.softmax(topk_logits, dim=1)

        E = self.num_experts
        Np = logits.size(0)
        out = x_patches.new_zeros(x_patches.shape)

        for j in range(self.active_experts):
            e_idx = topk_idx[:, j]
            wj = topk_w[:, j].view(Np, 1, 1, 1, 1)
            for e in range(E):
                mask = (e_idx == e)
                if not torch.any(mask):
                    continue
                sel = mask.nonzero(as_tuple=False).squeeze(1)
                xp = x_patches.index_select(0, sel)
                cp = cond_patches.index_select(0, sel) if cond_patches is not None else None
                yp = self.experts[e](xp, cond=cp)
                out.index_add_(0, sel, wj.index_select(0, sel) * yp)

        y = out.view(B, nH, nW, C, L, P, P).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, C, L, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            y = y[..., :H, :W]
        return y


class ConvLRULayer(nn.Module):
    def __init__(self, emb_ch: int, hidden_hw: Tuple[int, int], rank: int, use_selective: bool, bidirectional: bool, use_gate: bool, use_freq_prior: bool, use_sh_prior: bool, sh_Lmax: int):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.hidden_hw = (int(hidden_hw[0]), int(hidden_hw[1]))
        S, W = self.hidden_hw
        self.rank = int(rank)
        self.is_selective = bool(use_selective)
        self.bidirectional = bool(bidirectional)

        self.r_min = 0.8
        self.r_max = 0.99

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
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat))

        self.post_ifft_conv_real = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.post_ifft_conv_imag = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        out_dim_fusion = self.emb_ch * 2 if not self.bidirectional else self.emb_ch * 4
        self.post_ifft_proj = nn.Conv3d(out_dim_fusion, self.emb_ch, kernel_size=(1, 1, 1), padding="same")

        self.layer_norm = nn.LayerNorm([S, W])
        self.noise_level = nn.Parameter(torch.tensor(0.01))

        self.freq_prior = SpectralConv2d(self.emb_ch, self.emb_ch, 8, 8) if use_freq_prior else None
        self.sh_prior = SphericalHarmonicsPrior(self.emb_ch, S, W, Lmax=sh_Lmax) if use_sh_prior else None

        self.gate_conv = nn.Sequential(nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"), nn.Sigmoid()) if use_gate else None
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

    def _fft_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        B, L, C, S, W = x.shape
        Sp = next_fast_len(S)
        Wp = next_fast_len(W)
        pad_s = Sp - S
        pad_w = Wp - W
        x_r = x.contiguous().reshape(B * L, C, S, W)
        x_pad = F.pad(x_r, (0, pad_w, 0, pad_s))
        x_pad = x_pad.view(B, L, C, Sp, Wp)
        h = torch.fft.fft2(x_pad.to(torch.complex64), dim=(-2, -1), norm="ortho")
        return h, (Sp, Wp), (S, W)

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)

        h, (Sp, Wp), (S0, W0) = self._fft_impl(x)

        h_perm = h.permute(0, 1, 3, 4, 2).contiguous().view(B * L * Sp * Wp, C)
        h_proj = torch.matmul(h_perm, self.proj_W)
        h = h_proj.view(B, L, Sp, Wp, C).permute(0, 1, 4, 2, 3).contiguous()
        h = h + self.proj_b.view(1, 1, C, 1, 1)

        if self.freq_prior is not None:
            h = h + self.freq_prior(h)

        Uc = self.U_row.conj()
        t0 = torch.matmul(h.permute(0, 1, 2, 4, 3), Uc)
        t0 = t0.permute(0, 1, 2, 4, 3)
        zq = torch.matmul(t0, self.V_col)

        nu_log, theta_log = self.params_log_base.unbind(dim=0)
        disp_nu, disp_th = self.dispersion_mod.unbind(dim=0)
        nu_base = torch.exp(nu_log + disp_nu).view(1, 1, C, self.rank, 1)
        th_base = torch.exp(theta_log + disp_th).view(1, 1, C, self.rank, 1)

        dnu_force, dth_force = self._apply_forcing(x, dt)
        nu_t = torch.clamp(nu_base * dt + dnu_force, min=1e-6)
        th_t = th_base * dt + dth_force
        lamb = torch.exp(torch.complex(-nu_t, th_t))

        if self.training:
            noise_std = self.noise_level * torch.sqrt(dt + 1e-6)
            noise = torch.randn_like(zq) * noise_std
            x_in = zq + noise
        else:
            x_in = zq

        gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
        x_in = x_in * gamma_t

        zero_prev = torch.zeros_like(x_in[:, :1])
        x_in_fwd = torch.cat([last_hidden_in, x_in], dim=1) if last_hidden_in is not None else torch.cat([zero_prev, x_in], dim=1)
        lamb_fwd = torch.cat([lamb[:, :1], lamb], dim=1)
        z_out = self.pscan(lamb_fwd.expand_as(x_in_fwd).contiguous(), x_in_fwd.contiguous())[:, 1:]
        last_hidden_out = z_out[:, -1:]

        z_out_bwd = None
        if self.bidirectional:
            x_in_bwd = x_in.flip(1)
            lamb_bwd = lamb.flip(1)
            x_in_bwd = torch.cat([zero_prev, x_in_bwd], dim=1)
            lamb_bwd = torch.cat([lamb_bwd[:, :1], lamb_bwd], dim=1)
            z_out_bwd = self.pscan(lamb_bwd.expand_as(x_in_bwd).contiguous(), x_in_bwd.contiguous())[:, 1:]
            z_out_bwd = z_out_bwd.flip(1)

        def project_back(z: torch.Tensor) -> torch.Tensor:
            t1 = torch.matmul(z, self.V_col.conj().transpose(1, 2))
            t1 = t1.permute(0, 1, 2, 4, 3)
            return torch.matmul(t1, self.U_row.transpose(1, 2)).permute(0, 1, 2, 4, 3)

        def recover_spatial(h_rec: torch.Tensor) -> torch.Tensor:
            h_sp = torch.fft.ifft2(h_rec, dim=(-2, -1), norm="ortho")
            hr = self.post_ifft_conv_real(h_sp.real.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            hi = self.post_ifft_conv_imag(h_sp.imag.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            return torch.cat([hr, hi], dim=2)

        feat_fwd = recover_spatial(project_back(z_out))
        if z_out_bwd is not None:
            feat_bwd = recover_spatial(project_back(z_out_bwd))
            feat_final = torch.cat([feat_fwd, feat_bwd], dim=2)
        else:
            feat_final = feat_fwd

        h_final = self.post_ifft_proj(feat_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        if h_final.shape[-2:] != (S0, W0):
            h_final = h_final[..., :S0, :W0]

        if self.sh_prior is not None:
            h_final = self.sh_prior(h_final)

        h_final = self.layer_norm(h_final)

        if self.gate_conv is not None:
            gate = self.gate_conv(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x = (1 - gate) * x + gate * h_final
        else:
            x = x + h_final

        return x, last_hidden_out


class FeedForward(nn.Module):
    def __init__(self, emb_ch: int, ffn_hidden_ch: int, hidden_hw: Tuple[int, int], layers_num: int, use_cbam: bool, static_ch: int, num_expert: int, activate_expert: int):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.ffn_hidden_ch = int(ffn_hidden_ch)
        self.hidden_hw = (int(hidden_hw[0]), int(hidden_hw[1]))
        self.layers_num = int(layers_num)
        self.use_cbam = bool(use_cbam)
        self.static_ch = int(static_ch)
        self.num_expert = int(num_expert)
        self.activate_expert = int(activate_expert)

        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 1, 1), padding="same")
        cond_ch = self.emb_ch if self.static_ch > 0 else None

        blocks: List[nn.Module] = []
        for _ in range(self.layers_num):
            if self.num_expert > 1:
                blocks.append(SpatialPatchMoE(self.ffn_hidden_ch, self.hidden_hw, self.num_expert, self.activate_expert, self.use_cbam, cond_ch))
            else:
                blocks.append(GatedConvBlock(self.ffn_hidden_ch, self.hidden_hw, use_cbam=self.use_cbam, cond_channels=cond_ch))
        self.blocks = nn.ModuleList(blocks)

        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        y = self.c_in(x.permute(0, 2, 1, 3, 4))
        y = self.act(y)
        for block in self.blocks:
            y = block(y, cond=cond)
        y = self.c_out(y)
        y = y.permute(0, 2, 1, 3, 4)
        return residual + y


class ConvLRUBlock(nn.Module):
    def __init__(self, lru_layer: ConvLRULayer, ffn: FeedForward):
        super().__init__()
        self.lru_layer = lru_layer
        self.feed_forward = ffn

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor], listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        e = self.embedding.weight
        d = torch.cdist(x.float(), e.float(), p=2)
        idx = torch.argmin(d, dim=1).unsqueeze(1)
        enc = torch.zeros(idx.shape[0], self.num_embeddings, device=inputs.device, dtype=inputs.dtype)
        enc.scatter_(1, idx, 1)
        quant = torch.matmul(enc, e).view(B, L, H, W, C).permute(0, 4, 1, 2, 3)
        e_lat = F.mse_loss(quant.detach(), inputs)
        q_lat = F.mse_loss(quant, inputs.detach())
        loss = q_lat + self.commitment_cost * e_lat
        quant = inputs + (quant - inputs).detach()
        return quant, loss, idx


class DiffusionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if int(dim) % 2 != 0:
            raise ValueError(f"DiffusionHead dim must be even, got {dim}")
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * (-scale))
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)


class Decoder(nn.Module):
    def __init__(self, out_ch: int, emb_ch: int, hidden_hw: Tuple[int, int], hidden_factor: Tuple[int, int], head_mode: str, dec_strategy: str, dec_hidden_ch: int, dec_hidden_layers: int, static_ch: int):
        super().__init__()
        self.head_mode = str(head_mode)
        self.output_ch = int(out_ch)
        self.emb_ch = int(emb_ch)
        self.hidden_hw = (int(hidden_hw[0]), int(hidden_hw[1]))
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])
        self.dec_strategy = str(dec_strategy)
        self.dec_hidden_ch = int(dec_hidden_ch) if int(dec_hidden_layers) > 0 else int(emb_ch)
        self.dec_hidden_layers_num = int(dec_hidden_layers)
        self.static_ch = int(static_ch)

        out_ch_after_up = self.dec_hidden_ch

        if self.dec_strategy == "deconv":
            self.upsp = nn.ConvTranspose3d(self.emb_ch, out_ch_after_up, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            deconv3d_bilinear_init_(self.upsp.weight)
            with torch.no_grad():
                if self.upsp.bias is not None:
                    self.upsp.bias.zero_()
            self.pre_shuffle_conv = None
        else:
            self.upsp = None
            self.pre_shuffle_conv = nn.Conv3d(self.emb_ch, out_ch_after_up * self.rH * self.rW, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.rH, self.rW)
            with torch.no_grad():
                if self.pre_shuffle_conv.bias is not None:
                    self.pre_shuffle_conv.bias.zero_()

        H = self.hidden_hw[0] * self.rH
        W = self.hidden_hw[1] * self.rW

        if self.dec_hidden_layers_num > 0:
            cond_ch = self.emb_ch if self.static_ch > 0 else None
            self.c_hidden = nn.ModuleList([GatedConvBlock(out_ch_after_up, (H, W), use_cbam=False, cond_channels=cond_ch) for _ in range(self.dec_hidden_layers_num)])
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

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        y = x.permute(0, 2, 1, 3, 4)
        if self.upsp is not None:
            y = self.upsp(y)
        else:
            y = self.pre_shuffle_conv(y)
            y = pixel_shuffle_hw_3d(y, self.rH, self.rW)

        y = self.activation(y)

        if self.head_mode == "diffusion":
            if timestep is None:
                raise ValueError("timestep is required for diffusion head")
            t_emb = self.time_embed(timestep)
            y = y + t_emb.view(y.size(0), y.size(1), 1, 1, 1)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                y = layer(y, cond=cond)

        y = self.c_out(y)

        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(y, 2, dim=1)
            sigma = F.softplus(log_sigma) + 1e-6
            return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4)

        if self.head_mode == "token":
            quant, loss, idx = self.vq(y)
            return quant.permute(0, 2, 1, 3, 4), loss, idx

        return y.permute(0, 2, 1, 3, 4)


class Embedding(nn.Module):
    def __init__(self, input_ch: int, input_size: Tuple[int, int], emb_ch: int, emb_hidden_ch: int, emb_hidden_layers: int, static_ch: int, hidden_factor: Tuple[int, int]):
        super().__init__()
        self.input_ch = int(input_ch)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.emb_ch = int(emb_ch)
        self.emb_hidden_ch = int(emb_hidden_ch)
        self.emb_hidden_layers_num = int(emb_hidden_layers)
        self.static_ch = int(static_ch)
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])

        self.patch_embed = nn.Conv3d(
            self.input_ch,
            self.emb_hidden_ch,
            kernel_size=(1, self.rH + 2, self.rW + 2),
            stride=(1, self.rH, self.rW),
            padding=(0, 1, 1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch, 1, *self.input_size)
            out_dummy = self.patch_embed(dummy)
            _, C_hidden, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, int(H), int(W))

        self.hidden_hw = (self.input_downsp_shape[1], self.input_downsp_shape[2])

        self.static_embed = nn.Sequential(
            nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.rH + 2, self.rW + 2), stride=(self.rH, self.rW), padding=(1, 1)),
            nn.SiLU(),
        ) if self.static_ch > 0 else None

        if self.emb_hidden_layers_num > 0:
            cond_ch = self.emb_ch if self.static_ch > 0 else None
            self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_hw, use_cbam=False, cond_channels=cond_ch) for _ in range(self.emb_hidden_layers_num)])
        else:
            self.c_hidden = None

        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm([self.hidden_hw[0], self.hidden_hw[1]])

    def forward(self, x: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = x.permute(0, 2, 1, 3, 4)
        y = self.patch_embed(y)
        y = self.activation(y)

        cond = None
        if self.static_embed is not None and static_feats is not None:
            cond = self.static_embed(static_feats)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                y = layer(y, cond=cond)

        y = self.c_out(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.layer_norm(y)
        return y, cond


class ConvLRUModel(nn.Module):
    def __init__(
        self,
        emb_ch: int,
        hidden_hw: Tuple[int, int],
        convlru_num_blocks: int,
        use_unet: bool,
        lru_rank: int,
        use_selective: bool,
        bidirectional: bool,
        use_gate: bool,
        use_freq_prior: bool,
        use_sh_prior: bool,
        sh_Lmax: int,
        ffn_hidden_ch: int,
        ffn_hidden_layers: int,
        use_cbam: bool,
        static_ch: int,
        num_expert: int,
        activate_expert: int,
    ):
        super().__init__()
        C = int(emb_ch)
        H, W = int(hidden_hw[0]), int(hidden_hw[1])
        layers = int(convlru_num_blocks)
        self.use_unet = bool(use_unet)

        def make_block(hw: Tuple[int, int]) -> ConvLRUBlock:
            lru = ConvLRULayer(
                emb_ch=C,
                hidden_hw=hw,
                rank=int(lru_rank),
                use_selective=bool(use_selective),
                bidirectional=bool(bidirectional),
                use_gate=bool(use_gate),
                use_freq_prior=bool(use_freq_prior),
                use_sh_prior=bool(use_sh_prior),
                sh_Lmax=int(sh_Lmax),
            )
            ffn = FeedForward(
                emb_ch=C,
                ffn_hidden_ch=int(ffn_hidden_ch),
                hidden_hw=hw,
                layers_num=int(ffn_hidden_layers),
                use_cbam=bool(use_cbam),
                static_ch=int(static_ch),
                num_expert=int(num_expert),
                activate_expert=int(activate_expert),
            )
            return ConvLRUBlock(lru, ffn)

        if not self.use_unet:
            self.convlru_blocks = nn.ModuleList([make_block((H, W)) for _ in range(layers)])
            self.down_blocks = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.upsample = None
            self.fusion = None
        else:
            self.convlru_blocks = nn.ModuleList()
            self.down_blocks = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            curr_H, curr_W = H, W
            enc_hw: List[Tuple[int, int]] = []
            for i in range(layers):
                self.down_blocks.append(make_block((curr_H, curr_W)))
                enc_hw.append((curr_H, curr_W))
                if i < layers - 1:
                    self.skip_convs.append(nn.Identity())
                    curr_H //= 2
                    curr_W //= 2
            for i in range(layers - 2, -1, -1):
                self.up_blocks.append(make_block(enc_hw[i]))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            self.fusion = nn.Conv3d(C * 2, C, 1)

    def forward(self, x: torch.Tensor, last_hidden_ins: Optional[List[Optional[torch.Tensor]]] = None, listT: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            outs: List[torch.Tensor] = []
            hs_in = last_hidden_ins if last_hidden_ins is not None else [None] * len(self.convlru_blocks)
            for i, blk in enumerate(self.convlru_blocks):
                x, h = blk(x, hs_in[i], listT=listT, cond=cond)
                outs.append(h)
            return x, outs

        skips: List[torch.Tensor] = []
        outs: List[torch.Tensor] = []
        num_down = len(self.down_blocks)
        hs_in_down: List[Optional[torch.Tensor]]
        hs_in_up: List[Optional[torch.Tensor]]
        if last_hidden_ins is not None:
            hs_in_down = last_hidden_ins[:num_down]
            hs_in_up = last_hidden_ins[num_down:]
        else:
            hs_in_down = [None] * num_down
            hs_in_up = [None] * len(self.up_blocks)

        for i, blk in enumerate(self.down_blocks):
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x, h = blk(x, hs_in_down[i], listT=listT, cond=curr_cond)
            outs.append(h)
            if i < len(self.down_blocks) - 1:
                skips.append(x)
                xs = x.permute(0, 2, 1, 3, 4)
                xs = F.avg_pool3d(xs, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                x = xs.permute(0, 2, 1, 3, 4).contiguous()

        for i, blk in enumerate(self.up_blocks):
            xs = x.permute(0, 2, 1, 3, 4)
            xs = self.upsample(xs)
            x = xs.permute(0, 2, 1, 3, 4).contiguous()
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                dy = skip.size(-2) - x.size(-2)
                dx = skip.size(-1) - x.size(-1)
                x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = torch.cat([x, skip], dim=2)
            x = self.fusion(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()

            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x, h = blk(x, hs_in_up[i], listT=listT, cond=curr_cond)
            outs.append(h)

        return x, outs


class ConvLRU(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args

        self.input_ch = int(getattr(args, "input_ch", 1))
        self.input_size = tuple(getattr(args, "input_size", (64, 64)))
        self.out_ch = int(getattr(args, "out_ch", 1))
        self.static_ch = int(getattr(args, "static_ch", 0))

        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.emb_hidden_ch = int(getattr(args, "emb_hidden_ch", 64))
        self.emb_hidden_layers_num = int(getattr(args, "emb_hidden_layers_num", 0))
        self.hidden_factor = tuple(getattr(args, "hidden_factor", (2, 2)))

        self.convlru_num_blocks = int(getattr(args, "convlru_num_blocks", 2))
        self.lru_rank = int(getattr(args, "lru_rank", 16))
        self.use_selective = bool(getattr(args, "use_selective", False))
        self.bidirectional = bool(getattr(args, "bidirectional", False))
        self.use_gate = bool(getattr(args, "use_gate", True))

        self.use_freq_prior = bool(getattr(args, "use_freq_prior", False))
        self.use_sh_prior = bool(getattr(args, "use_sh_prior", False))
        self.sh_Lmax = int(getattr(args, "sh_Lmax", 6))

        self.ffn_hidden_ch = int(getattr(args, "ffn_hidden_ch", 64))
        self.ffn_hidden_layers_num = int(getattr(args, "ffn_hidden_layers_num", 1))
        self.use_cbam = bool(getattr(args, "use_cbam", False))

        self.num_expert = int(getattr(args, "num_expert", -1))
        self.activate_expert = int(getattr(args, "activate_expert", 2))

        self.unet = bool(getattr(args, "unet", False))

        self.dec_strategy = str(getattr(args, "dec_strategy", "pxsf"))
        self.dec_hidden_ch = int(getattr(args, "dec_hidden_ch", self.emb_ch))
        self.dec_hidden_layers_num = int(getattr(args, "dec_hidden_layers_num", 0))
        self.head_mode = str(getattr(args, "head_mode", "gaussian"))

        self.embedding = Embedding(
            input_ch=self.input_ch,
            input_size=(int(self.input_size[0]), int(self.input_size[1])),
            emb_ch=self.emb_ch,
            emb_hidden_ch=self.emb_hidden_ch,
            emb_hidden_layers=self.emb_hidden_layers_num,
            static_ch=self.static_ch,
            hidden_factor=(int(self.hidden_factor[0]), int(self.hidden_factor[1])),
        )

        C, Hs, Ws = self.embedding.input_downsp_shape
        self.convlru_model = ConvLRUModel(
            emb_ch=C,
            hidden_hw=(Hs, Ws),
            convlru_num_blocks=self.convlru_num_blocks,
            use_unet=self.unet,
            lru_rank=self.lru_rank,
            use_selective=self.use_selective,
            bidirectional=self.bidirectional,
            use_gate=self.use_gate,
            use_freq_prior=self.use_freq_prior,
            use_sh_prior=self.use_sh_prior,
            sh_Lmax=self.sh_Lmax,
            ffn_hidden_ch=self.ffn_hidden_ch,
            ffn_hidden_layers=self.ffn_hidden_layers_num,
            use_cbam=self.use_cbam,
            static_ch=self.static_ch,
            num_expert=self.num_expert,
            activate_expert=self.activate_expert,
        )

        self.decoder = Decoder(
            out_ch=self.out_ch,
            emb_ch=C,
            hidden_hw=(Hs, Ws),
            hidden_factor=(int(self.hidden_factor[0]), int(self.hidden_factor[1])),
            head_mode=self.head_mode,
            dec_strategy=self.dec_strategy,
            dec_hidden_ch=self.dec_hidden_ch,
            dec_hidden_layers=self.dec_hidden_layers_num,
            static_ch=self.static_ch,
        )

        skip_contains = ["layer_norm", "params_log", "prior", "post_ifft", "forcing", "dispersion"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains):
                    continue
                if n.endswith(".bias"):
                    p.zero_()
                    continue
                if p.dim() > 1 and not torch.is_complex(p):
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cond = None
        if self.embedding.static_embed is not None and static_feats is not None:
            cond = self.embedding.static_embed(static_feats)

        if mode == "p":
            x_emb, _ = self.embedding(x, static_feats=static_feats)
            x_hid, _ = self.convlru_model(x_emb, listT=listT, cond=cond)
            return self.decoder(x_hid, cond=cond, timestep=timestep)

        if out_gen_num is None or out_gen_num <= 0:
            raise ValueError("out_gen_num must be set for inference mode")

        if listT is None:
            listT0 = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT

        out: List[torch.Tensor] = []
        x_emb, _ = self.embedding(x, static_feats=static_feats)
        x_hid, last_hidden = self.convlru_model(x_emb, listT=listT0, cond=cond)
        x_dec = self.decoder(x_hid, cond=cond, timestep=timestep)
        if isinstance(x_dec, tuple):
            x_dec0 = x_dec[0]
        else:
            x_dec0 = x_dec

        x_step_dist = x_dec0[:, -1:]
        if self.decoder.head_mode == "gaussian":
            x_step_mean = x_step_dist[..., : self.out_ch, :, :]
        else:
            x_step_mean = x_step_dist

        out.append(x_step_dist)

        if listT_future is None:
            listT_future = torch.ones(x.size(0), out_gen_num - 1, device=x.device, dtype=x.dtype)

        for t in range(out_gen_num - 1):
            dt = listT_future[:, t : t + 1]
            x_in, _ = self.embedding(x_step_mean, static_feats=None)
            x_hid, last_hidden = self.convlru_model(x_in, last_hidden_ins=last_hidden, listT=dt, cond=cond)
            x_dec = self.decoder(x_hid, cond=cond, timestep=timestep)
            if isinstance(x_dec, tuple):
                x_dec = x_dec[0]
            x_step_dist = x_dec[:, -1:]
            if self.decoder.head_mode == "gaussian":
                x_step_mean = x_step_dist[..., : self.out_ch, :, :]
            else:
                x_step_mean = x_step_dist
            out.append(x_step_dist)

        return torch.cat(out, dim=1)
