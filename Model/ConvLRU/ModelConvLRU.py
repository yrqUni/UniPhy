import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


Tensor = torch.Tensor


def _kaiming_like_(tensor: Tensor) -> Tensor:
    nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="relu")
    return tensor


def icnr_conv3d_weight_(weight: Tensor, rH: int, rW: int) -> Tensor:
    out_ch, in_ch, kD, kH, kW = weight.shape
    base_out = out_ch // (rH * rW)
    base = weight.new_zeros((base_out, in_ch, 1, kH, kW))
    _kaiming_like_(base)
    base = base.repeat_interleave(rH * rW, dim=0)
    with torch.no_grad():
        weight.copy_(base)
    return weight


def _bilinear_kernel_2d(kH: int, kW: int, device, dtype) -> Tensor:
    factor_h = (kH + 1) // 2
    factor_w = (kW + 1) // 2
    center_h = factor_h - 1 if kH % 2 == 1 else factor_h - 0.5
    center_w = factor_w - 1 if kW % 2 == 1 else factor_w - 0.5
    og_h = torch.arange(kH, device=device, dtype=dtype)
    og_w = torch.arange(kW, device=device, dtype=dtype)
    fh = (1 - torch.abs(og_h - center_h) / factor_h).unsqueeze(1)
    fw = (1 - torch.abs(og_w - center_w) / factor_w).unsqueeze(0)
    return fh @ fw


def deconv3d_bilinear_init_(weight: Tensor) -> Tensor:
    in_ch, out_ch, kD, kH, kW = weight.shape
    with torch.no_grad():
        weight.zero_()
        kernel = _bilinear_kernel_2d(kH, kW, weight.device, weight.dtype)
        c = min(in_ch, out_ch)
        for i in range(c):
            weight[i, i, 0, :, :] = kernel
    return weight


def pixel_shuffle_hw_3d(x: Tensor, rH: int, rW: int) -> Tensor:
    N, C_mul, D, H, W = x.shape
    C = C_mul // (rH * rW)
    x = x.view(N, C, rH, rW, D, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.view(N, C, D, H * rH, W * rW)
    return x


def pixel_unshuffle_hw_3d(x: Tensor, rH: int, rW: int) -> Tensor:
    N, C, D, H, W = x.shape
    x = x.view(N, C, D, H // rH, rH, W // rW, rW)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.view(N, C * rH * rW, D, H // rH, W // rW)
    return x


def _resize_complex_1d(x: Tensor, new_len: int) -> Tensor:
    if x.size(1) == new_len:
        return x
    if not x.is_complex():
        raise TypeError("x must be complex")
    C, S, R = x.shape
    xr = torch.view_as_real(x).permute(0, 2, 3, 1).contiguous().view(C * R * 2, 1, S)
    xr = F.interpolate(xr, size=new_len, mode="linear", align_corners=False)
    xr = xr.view(C, R, 2, new_len).permute(0, 3, 1, 2).contiguous()
    return torch.view_as_complex(xr)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        x = x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return x


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], use_cbam: bool = False, cond_channels: Optional[int] = None):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.dw_conv = nn.Conv3d(channels, channels, kernel_size=(1, 7, 7), padding="same", groups=channels)
        self.norm = nn.LayerNorm([hidden_size[0], hidden_size[1]])
        self.cond_channels = int(cond_channels) if cond_channels is not None else 0
        if self.cond_channels > 0:
            self.cond_proj = nn.Conv3d(self.cond_channels, channels * 2, kernel_size=1)
        else:
            self.cond_proj = None
        self.pw_conv_in = nn.Conv3d(channels, channels * 2, kernel_size=1)
        self.act = nn.SiLU()
        self.pw_conv_out = nn.Conv3d(channels, channels, kernel_size=1)
        if self.use_cbam:
            self.cbam = CBAM2DPerStep(channels, reduction=16)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
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
                cond_in = F.interpolate(cond_in.squeeze(2), size=x.shape[-2:], mode="bilinear", align_corners=False).unsqueeze(2)
            affine = self.cond_proj(cond_in)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta
        x = self.pw_conv_in(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.act(x) * gate
        if self.use_cbam:
            x = self.cbam(x)
        x = self.pw_conv_out(x)
        return residual + x


class SpatialPatchMoE(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_size: Tuple[int, int],
        num_experts: int,
        active_experts: int,
        use_cbam: bool,
        cond_channels: Optional[int],
        patch_size: int = 8,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.patch_size = int(patch_size)
        self.expert_hidden_size = (self.patch_size, self.patch_size)
        self.experts = nn.ModuleList(
            [
                GatedConvBlock(
                    channels,
                    self.expert_hidden_size,
                    use_cbam=use_cbam,
                    cond_channels=cond_channels,
                )
                for _ in range(self.num_experts)
            ]
        )
        router_in_dim = int(channels) + (int(cond_channels) if (cond_channels is not None and int(cond_channels) > 0) else 0)
        self.router = nn.Linear(router_in_dim, self.num_experts)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
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

        router_in = x_patches.mean(dim=(2, 3, 4))
        if cond is not None:
            if cond.dim() == 4:
                cond_e = cond.unsqueeze(2).expand(-1, -1, L, -1, -1)
            else:
                cond_e = cond
            cond_patches = cond_e.view(B, -1, L, nH, P, nW, P).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, cond_e.size(1), L, P, P)
            router_cond = cond_patches.mean(dim=(2, 3, 4))
            router_input = torch.cat([router_in, router_cond], dim=1)
        else:
            cond_patches = None
            router_input = router_in

        logits = self.router(router_input)
        topk_logits, topk_indices = torch.topk(logits, self.active_experts, dim=1)
        topk_weights = F.softmax(topk_logits, dim=1)

        out = torch.zeros_like(x_patches)
        for k in range(self.active_experts):
            e_idx = topk_indices[:, k]
            w = topk_weights[:, k].view(-1, 1, 1, 1, 1)
            for e in range(self.num_experts):
                mask = (e_idx == e)
                if not bool(mask.any()):
                    continue
                sel = mask.nonzero(as_tuple=False).squeeze(1)
                x_sel = x_patches.index_select(0, sel)
                c_sel = cond_patches.index_select(0, sel) if cond_patches is not None else None
                y_sel = self.experts[e](x_sel, cond=c_sel)
                out.index_add_(0, sel, y_sel * w.index_select(0, sel))

        output = out.view(B, nH, nW, C, L, P, P).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, C, L, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            output = output[..., :H, :W]
        return output


class ConvLRULayer(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.hidden_size = [int(input_downsp_shape[1]), int(input_downsp_shape[2])]
        S, W = self.hidden_size
        self.rank = int(getattr(args, "lru_rank", min(S, W, 32)))
        self.is_selective = bool(getattr(args, "use_selective", False))
        self.bidirectional = bool(getattr(args, "bidirectional", False))

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

        if bool(getattr(args, "use_gate", False)):
            self.gate_conv = nn.Sequential(
                nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"),
                nn.Sigmoid(),
            )
        else:
            self.gate_conv = None

        self.pscan = pscan

    def _apply_forcing(self, x: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
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

    def _fft_impl(self, x: Tensor) -> Tuple[Tensor, int]:
        B, L, C, S, W = x.shape
        pad_size = S // 4
        x_reshaped = x.contiguous().reshape(B * L, C, S, W)
        x_pad = F.pad(x_reshaped, (0, 0, pad_size, pad_size), mode="reflect")
        x_pad = x_pad.view(B, L, C, S + 2 * pad_size, W)
        h = torch.fft.fft2(x_pad.to(torch.cfloat), dim=(-2, -1), norm="ortho")
        return h, pad_size

    def forward(self, x: Tensor, last_hidden_in: Optional[Tensor], listT: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)

        h, pad_size = self._fft_impl(x)
        S_pad, W_pad = h.shape[-2], h.shape[-1]

        U_eff = _resize_complex_1d(self.U_row, S_pad)
        V_eff = _resize_complex_1d(self.V_col, W_pad)
        Uc = U_eff.conj()

        h_perm = h.permute(0, 1, 3, 4, 2).contiguous().view(B * L * S_pad * W_pad, C)
        h_proj = torch.matmul(h_perm, self.proj_W)
        h = h_proj.view(B, L, S_pad, W_pad, C).permute(0, 1, 4, 2, 3).contiguous()
        if self.proj_b is not None:
            h = h + self.proj_b.view(1, 1, C, 1, 1)

        h_spatial = torch.fft.ifft2(h, dim=(-2, -1), norm="ortho")
        if pad_size > 0:
            h_spatial = h_spatial[..., pad_size:-pad_size, :]
        h = torch.fft.fft2(h_spatial, dim=(-2, -1), norm="ortho")

        t0 = torch.matmul(h.permute(0, 1, 2, 4, 3), Uc)
        t0 = t0.permute(0, 1, 2, 4, 3)
        zq = torch.matmul(t0, V_eff)

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
            zq = zq + torch.randn_like(zq) * noise_std

        gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
        x_in = zq * gamma_t

        zero_prev = torch.zeros_like(x_in[:, :1])
        x_in_fwd = torch.cat([last_hidden_in, x_in], dim=1) if last_hidden_in is not None else torch.cat([zero_prev, x_in], dim=1)
        lamb_fwd = torch.cat([lamb[:, :1], lamb], dim=1)
        lamb_in_fwd = lamb_fwd.expand_as(x_in_fwd).contiguous()
        z_out = self.pscan(lamb_in_fwd, x_in_fwd.contiguous())[:, 1:]
        last_hidden_out = z_out[:, -1:]

        z_out_bwd = None
        if self.bidirectional:
            x_in_bwd = x_in.flip(1)
            lamb_bwd = lamb.flip(1)
            x_in_bwd = torch.cat([zero_prev, x_in_bwd], dim=1)
            lamb_bwd = torch.cat([lamb_bwd[:, :1], lamb_bwd], dim=1)
            lamb_in_bwd = lamb_bwd.expand_as(x_in_bwd).contiguous()
            z_out_bwd = self.pscan(lamb_in_bwd, x_in_bwd.contiguous())[:, 1:].flip(1)

        def project_back(z: Tensor) -> Tensor:
            t1 = torch.matmul(z, V_eff.conj().transpose(1, 2))
            t1 = t1.permute(0, 1, 2, 4, 3)
            return torch.matmul(t1, U_eff.transpose(1, 2)).permute(0, 1, 2, 4, 3)

        h_rec_fwd = project_back(z_out)

        def recover_spatial(h_rec: Tensor) -> Tensor:
            h_sp = torch.fft.ifft2(h_rec, dim=(-2, -1), norm="ortho")
            hr = self.post_ifft_conv_real(h_sp.real.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            hi = self.post_ifft_conv_imag(h_sp.imag.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            return torch.cat([hr, hi], dim=2)

        feat_fwd = recover_spatial(h_rec_fwd)
        if self.bidirectional and z_out_bwd is not None:
            h_rec_bwd = project_back(z_out_bwd)
            feat_bwd = recover_spatial(h_rec_bwd)
            feat_final = torch.cat([feat_fwd, feat_bwd], dim=2)
        else:
            feat_final = feat_fwd

        h_final = self.post_ifft_proj(feat_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
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
        cond_ch = self.emb_ch if self.static_ch > 0 else None

        blocks: List[nn.Module] = []
        for _ in range(self.layers_num):
            if self.num_expert > 1:
                blocks.append(
                    SpatialPatchMoE(
                        self.ffn_hidden_ch,
                        self.hidden_size,
                        self.num_expert,
                        self.activate_expert,
                        self.use_cbam,
                        cond_ch,
                    )
                )
            else:
                blocks.append(GatedConvBlock(self.ffn_hidden_ch, self.hidden_size, use_cbam=self.use_cbam, cond_channels=cond_ch))
        self.blocks = nn.ModuleList(blocks)

        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        residual = x
        x3 = self.c_in(x.permute(0, 2, 1, 3, 4))
        x3 = self.act(x3)
        for blk in self.blocks:
            x3 = blk(x3, cond=cond)
        x3 = self.c_out(x3)
        x = x3.permute(0, 2, 1, 3, 4)
        return residual + x


class ConvLRUBlock(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(self, x: Tensor, last_hidden_in: Optional[Tensor], listT: Optional[Tensor] = None, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
        with torch.no_grad():
            self.embedding.weight.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, L, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        dist = torch.cdist(x.float(), self.embedding.weight.float(), p=2.0)
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings.float(), self.embedding.weight.float()).to(inputs.dtype)
        quantized = quantized.view(B, L, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
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

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(1, (half_dim - 1))
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * (-emb_scale))
        emb = t[:, None].to(dtype=t.dtype) * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


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
            H = self.hidden_size[0] * self.rH
            W = self.hidden_size[1] * self.rW
            cond_ch = self.emb_ch if self.static_ch > 0 else None
            self.c_hidden = nn.ModuleList(
                [GatedConvBlock(out_ch_after_up, (H, W), use_cbam=False, cond_channels=cond_ch) for _ in range(self.dec_hidden_layers_num)]
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

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, timestep: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x3 = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.upsp is not None:
            x3 = self.upsp(x3)
        else:
            x3 = self.pre_shuffle_conv(x3)
            x3 = pixel_shuffle_hw_3d(x3, self.rH, self.rW)
        x3 = self.activation(x3)
        if self.head_mode == "diffusion":
            if timestep is None:
                raise ValueError("timestep is required for diffusion head_mode")
            t_emb = self.time_embed(timestep)
            x3 = x3 + t_emb.view(x3.size(0), x3.size(1), 1, 1, 1)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x3 = layer(x3, cond=cond)

        x3 = self.c_out(x3)

        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x3, 2, dim=1)
            sigma = F.softplus(log_sigma) + 1e-6
            return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4).contiguous()

        if self.head_mode == "token":
            quantized, loss, indices = self.vq(x3)
            return quantized.permute(0, 2, 1, 3, 4).contiguous(), loss, indices

        return x3.permute(0, 2, 1, 3, 4).contiguous()


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
            dummy = torch.zeros(1, self.input_ch, 1, *self.input_size)
            out_dummy = self.patch_embed(dummy)
            _, _, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, H, W)

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
        self.layer_norm = nn.LayerNorm([self.hidden_size[0], self.hidden_size[1]])

    def forward(self, x: Tensor, static_feats: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        x3 = x.permute(0, 2, 1, 3, 4).contiguous()
        x3 = self.patch_embed(x3)
        x3 = self.activation(x3)

        cond = None
        if self.static_ch > 0 and static_feats is not None and self.static_embed is not None:
            cond = self.static_embed(static_feats)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x3 = layer(x3, cond=cond)

        x3 = self.c_out(x3)
        x = x3.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer_norm(x)
        return x, cond


class ConvLRUModel(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.use_unet = bool(getattr(args, "unet", False))
        layers = int(getattr(args, "convlru_num_blocks", 2))
        C = int(getattr(args, "emb_ch", input_downsp_shape[0]))
        H, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])

        if not self.use_unet:
            self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, (C, H, W)) for _ in range(layers)])
            self.down_blocks = None
            self.up_blocks = None
            self.upsample = None
            self.fusion = None
        else:
            self.down_blocks = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.encoder_res: List[Tuple[int, int]] = []
            curr_H, curr_W = H, W
            for i in range(layers):
                self.down_blocks.append(ConvLRUBlock(self.args, (C, curr_H, curr_W)))
                self.encoder_res.append((curr_H, curr_W))
                if i < layers - 1:
                    curr_H //= 2
                    curr_W //= 2
            for i in range(layers - 2, -1, -1):
                h_up, w_up = self.encoder_res[i]
                self.up_blocks.append(ConvLRUBlock(self.args, (C, h_up, w_up)))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            self.fusion = nn.Conv3d(C * 2, C, 1)
            self.convlru_blocks = None

    def forward(
        self,
        x: Tensor,
        last_hidden_ins: Optional[List[Tensor]] = None,
        listT: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        if not self.use_unet:
            outs: List[Tensor] = []
            hs_out: List[Tensor] = []
            h_in_list = last_hidden_ins if last_hidden_ins is not None else [None] * len(self.convlru_blocks)
            for i, blk in enumerate(self.convlru_blocks):
                x, h = blk(x, h_in_list[i], listT=listT, cond=cond)
                hs_out.append(h)
            return x, hs_out

        assert self.down_blocks is not None and self.up_blocks is not None and self.upsample is not None and self.fusion is not None

        num_down = len(self.down_blocks)
        num_up = len(self.up_blocks)

        if last_hidden_ins is None:
            hs_in_down = [None] * num_down
            hs_in_up = [None] * num_up
        else:
            hs_in_down = last_hidden_ins[:num_down]
            hs_in_up = last_hidden_ins[num_down:]

        skips: List[Tensor] = []
        hs_out: List[Tensor] = []

        for i, blk in enumerate(self.down_blocks):
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x, h = blk(x, hs_in_down[i], listT=listT, cond=curr_cond)
            hs_out.append(h)
            if i < num_down - 1:
                skips.append(x)
                x3 = x.permute(0, 2, 1, 3, 4).contiguous()
                x3 = F.avg_pool3d(x3, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                x = x3.permute(0, 2, 1, 3, 4).contiguous()

        for i, blk in enumerate(self.up_blocks):
            x3 = x.permute(0, 2, 1, 3, 4).contiguous()
            x3 = self.upsample(x3)
            x = x3.permute(0, 2, 1, 3, 4).contiguous()

            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(-2) - x.size(-2)
                diffX = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x, skip], dim=2)
            x = self.fusion(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()

            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x, h = blk(x, hs_in_up[i], listT=listT, cond=curr_cond)
            hs_out.append(h)

        return x, hs_out


class ConvLRU(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)

        skip_contains = ["layer_norm", "params_log", "prior", "post_ifft", "spatial_mod", "forcing", "dispersion"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains):
                    continue
                if n.endswith(".bias"):
                    p.zero_()
                    continue
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: Tensor,
        mode: str = "p",
        out_gen_num: Optional[int] = None,
        listT: Optional[Tensor] = None,
        listT_future: Optional[Tensor] = None,
        static_feats: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        mode = str(mode).lower()
        cond = None
        if self.embedding.static_ch > 0 and static_feats is not None and self.embedding.static_embed is not None:
            cond = self.embedding.static_embed(static_feats)

        if mode == "p":
            x_emb, _ = self.embedding(x, static_feats=static_feats)
            x_hid, _ = self.convlru_model(x_emb, listT=listT, cond=cond)
            return self.decoder(x_hid, cond=cond, timestep=timestep)

        if out_gen_num is None or out_gen_num <= 0:
            raise ValueError("out_gen_num must be provided for inference mode")

        if listT is None:
            B = x.size(0)
            listT = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype)

        x_emb, _ = self.embedding(x, static_feats=static_feats)
        x_hid, hs = self.convlru_model(x_emb, listT=listT, cond=cond)
        x_dec = self.decoder(x_hid, cond=cond, timestep=timestep)
        if isinstance(x_dec, tuple):
            x_dec = x_dec[0]
        out: List[Tensor] = []
        x_step_dist = x_dec[:, -1:]
        if str(getattr(self.args, "head_mode", "gaussian")).lower() == "gaussian":
            x_step_mean = x_step_dist[..., : int(getattr(self.args, "out_ch", 1)), :, :]
        else:
            x_step_mean = x_step_dist
        out.append(x_step_dist)

        for t in range(int(out_gen_num) - 1):
            if listT_future is None:
                dt = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            else:
                dt = listT_future[:, t : t + 1]
            x_in, _ = self.embedding(x_step_mean, static_feats=None)
            x_hid, hs = self.convlru_model(x_in, last_hidden_ins=hs, listT=dt, cond=cond)
            x_dec = self.decoder(x_hid, cond=cond, timestep=timestep)
            if isinstance(x_dec, tuple):
                x_dec = x_dec[0]
            x_step_dist = x_dec[:, -1:]
            if str(getattr(self.args, "head_mode", "gaussian")).lower() == "gaussian":
                x_step_mean = x_step_dist[..., : int(getattr(self.args, "out_ch", 1)), :, :]
            else:
                x_step_mean = x_step_dist
            out.append(x_step_dist)

        return torch.cat(out, dim=1)
