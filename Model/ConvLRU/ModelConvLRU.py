import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def icnr_conv3d_weight_(weight: torch.Tensor, rH: int, rW: int) -> torch.Tensor:
    out_ch, in_ch, kD, kH, kW = weight.shape
    base_out = out_ch // (rH * rW)
    base = weight.new_zeros((base_out, in_ch, 1, kH, kW))
    nn.init.kaiming_normal_(base, a=0, mode="fan_in", nonlinearity="relu")
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


class DeformConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.conv_offset = nn.Conv2d(
            int(in_channels),
            int(2 * kernel_size * kernel_size + kernel_size * kernel_size),
            kernel_size=int(kernel_size),
            padding=int(padding),
        )
        self.conv_dcn = torchvision.ops.DeformConv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(padding),
            bias=bool(bias),
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.conv_dcn(x, offset, mask)


class FactorizedPeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, conv_type: str = "conv"):
        super().__init__()
        self.pad_sp = int(kernel_size) // 2
        self.conv_type = str(conv_type)
        if self.conv_type == "dcn":
            self.spatial_conv = DeformConv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=0, bias=False)
        else:
            self.spatial_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=0, bias=False)
        self.depth_conv = nn.Conv3d(int(out_channels), int(out_channels), kernel_size=(1, 1, 1), padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_sp = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        x_sp = F.pad(x_sp, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        x_sp = x_sp.reshape(B, D, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return self.depth_conv(x_sp)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(int(int(channels) // int(reduction)), 4)
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


class HamiltonianGenerator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        mid = max(ch // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, mid, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid, 1, 3, padding=1),
        )
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        H_field = self.conv(x_flat)
        lat = torch.linspace(-math.pi / 2, math.pi / 2, H, device=x.device).view(1, 1, H, 1)
        metric_correction = 1.0 / (torch.cos(lat) + 1e-6)
        H_padded = F.pad(H_field, (1, 1, 0, 0), mode="circular")
        H_padded = F.pad(H_padded, (0, 0, 1, 1), mode="replicate")
        grad_x = F.conv2d(H_padded, self.sobel_x.to(x.device))
        grad_y = F.conv2d(H_padded, self.sobel_y.to(x.device))
        u = grad_y
        v = -grad_x * metric_correction
        flow = torch.cat([u, v], dim=1)
        flow = torch.tanh(flow)
        return flow.view(B, L, 2, H, W)


_BASE_GRID_CACHE: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_base_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(H), int(W), device, dtype)
    g = _BASE_GRID_CACHE.get(key, None)
    if g is not None:
        return g
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
    _BASE_GRID_CACHE[key] = base_grid
    return base_grid


class LieTransport(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h_prev: torch.Tensor, flow: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B, C, H, W, R = h_prev.shape
        if flow.dim() == 5:
            flow = flow[:, 0]
        dt = dt.view(B, 1)
        base_grid = _get_base_grid(H, W, h_prev.device, h_prev.dtype)
        chunk_size = 4
        h_warped_list: List[torch.Tensor] = []
        for i in range(0, R, chunk_size):
            r_end = min(i + chunk_size, R)
            curr_R = r_end - i
            h_chunk = h_prev[..., i:r_end]
            h_flat = h_chunk.permute(0, 4, 1, 2, 3).reshape(B * curr_R, C, H, W)
            flow_chunk = flow.unsqueeze(1).expand(-1, curr_R, -1, -1, -1).reshape(B * curr_R, 2, H, W)
            dt_chunk = dt.expand(-1, curr_R).reshape(B * curr_R, 1, 1, 1)
            flow_dt = flow_chunk * dt_chunk
            sampling_grid = base_grid.expand(B * curr_R, -1, -1, -1) - flow_dt.permute(0, 2, 3, 1)
            h_warped_flat = F.grid_sample(h_flat, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)
            h_warped_list.append(h_warped_flat.reshape(B, curr_R, C, H, W).permute(0, 2, 3, 4, 1))
        return torch.cat(h_warped_list, dim=4)


class KoopmanParamEstimator(nn.Module):
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
        self.pool = nn.AdaptiveAvgPool2d((1, self.w_freq))
        self.head = nn.Linear(ch, self.emb_ch * self.rank * 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        feat = self.conv(x)
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 3, 1).reshape(B, 1, self.w_freq, C)
        params = self.head(feat)
        params = params.view(B, 1, self.w_freq, self.emb_ch, self.rank, 3)
        params = params.permute(0, 1, 3, 2, 4, 5).contiguous()
        nu_rate = F.softplus(params[..., 0])
        theta_rate = torch.tanh(params[..., 1]) * math.pi
        sigma = torch.sigmoid(params[..., 2])
        return nu_rate, theta_rate, sigma


class SpectralKoopmanSDE(nn.Module):
    def __init__(self, channels: int, rank: int, w_freq: int, use_noise: bool = False, noise_scale: float = 1.0):
        super().__init__()
        self.estimator = KoopmanParamEstimator(int(channels), int(channels), int(rank), int(w_freq))
        self.channels = int(channels)
        self.rank = int(rank)
        self.use_noise = bool(use_noise)
        self.noise_scale = float(noise_scale)

    def forward(self, h_trans: torch.Tensor, x_curr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B, C, H, W, R = h_trans.shape
        h_freq = torch.fft.rfft2(h_trans.permute(0, 4, 1, 2, 3).float(), norm="ortho")
        x_flat = x_curr.squeeze(2)
        nu_rate, theta_rate, sigma = self.estimator(x_flat)
        dt_ = dt.view(B, -1)
        dt_expanded = dt_.view(B, dt_.shape[1], 1, 1, 1, 1)
        nu = nu_rate.unsqueeze(4) * dt_expanded
        theta = theta_rate.unsqueeze(4) * dt_expanded
        sigma = sigma.unsqueeze(4)
        decay = torch.exp(-nu)
        rotate = torch.exp(1j * theta)
        lambda_k = decay * rotate
        lambda_k = lambda_k.squeeze(1).permute(0, 4, 1, 3, 2)
        sigma_b = sigma.squeeze(1).permute(0, 4, 1, 3, 2)
        if self.use_noise:
            dt_sqrt = torch.sqrt(dt_.clamp_min(0.0)).view(B, 1, 1, 1, 1)
            amp = sigma_b.mean(dim=-1, keepdim=True) * (self.noise_scale * dt_sqrt)
            noise = torch.randn_like(h_freq) * amp
        else:
            noise = torch.zeros_like(h_freq)
        h_evolved = h_freq * lambda_k + noise
        h_out = torch.fft.irfft2(h_evolved, s=(H, W), norm="ortho")
        return h_out.permute(0, 2, 3, 4, 1)


class StaticInitState(nn.Module):
    def __init__(self, static_ch: int, emb_ch: int, rank: int, S: int, W_freq: int):
        super().__init__()
        self.static_ch = int(static_ch)
        self.emb_ch = int(emb_ch)
        self.rank = int(rank)
        self.S = int(S)
        self.W_freq = int(W_freq)
        self.mapper = nn.Sequential(
            nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.S, self.W_freq)),
            nn.Conv2d(self.emb_ch, self.emb_ch * self.rank, kernel_size=1),
        )

    def forward(self, static_feats: torch.Tensor) -> torch.Tensor:
        B = static_feats.size(0)
        out = self.mapper(static_feats)
        out = out.view(B, self.emb_ch, self.rank, self.S, self.W_freq)
        return out.permute(0, 1, 3, 4, 2)


class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, L, H, W = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
            y = super().forward(y)
            return y.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return super().forward(x)


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


class SimplifiedHKLFLayer(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        H, W = int(input_downsp_shape[1]), int(input_downsp_shape[2])
        self.rank = int(getattr(args, "lru_rank", 32))
        self.W_freq = W // 2 + 1
        use_noise = bool(getattr(args, "koopman_use_noise", False))
        noise_scale = float(getattr(args, "koopman_noise_scale", 1.0))
        self.dt_ref = float(getattr(args, "dt_ref", getattr(args, "T", 1.0)))
        if self.dt_ref <= 0:
            self.dt_ref = 1.0
        self.inj_k = float(getattr(args, "inj_k", 2.0))
        self.inj_k = max(self.inj_k, 0.0)
        self.hamiltonian = HamiltonianGenerator(self.emb_ch)
        self.lie_transport = LieTransport()
        self.koopman = SpectralKoopmanSDE(self.emb_ch, self.rank, self.W_freq, use_noise=use_noise, noise_scale=noise_scale)
        self.proj_out = nn.Linear(self.rank, 1)
        if bool(getattr(args, "learnable_init_state", False)) and int(getattr(args, "static_ch", 0)) > 0:
            self.init_state = StaticInitState(int(args.static_ch), self.emb_ch, self.rank, H, W)
        else:
            self.init_state = None
        self.post_ifft_proj = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.norm = SpatialGroupNorm(4, self.emb_ch)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor] = None,
        listT: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L, H, W = x.shape
        if last_hidden_in is None:
            if self.init_state is not None and static_feats is not None:
                h0 = self.init_state(static_feats).unsqueeze(1).repeat(1, L, 1, 1, 1, 1)
                curr_h = h0[:, 0]
            else:
                curr_h = torch.zeros(B, C, H, W, self.rank, device=x.device, dtype=x.dtype)
        else:
            curr_h = last_hidden_in
        if listT is None:
            dt_seq = torch.ones(B, L, device=x.device, dtype=x.dtype)
        else:
            dt_seq = _match_dt_seq(listT.to(x.device, x.dtype), L)
        h_states: List[torch.Tensor] = []
        x_perm = x.permute(0, 2, 1, 3, 4)
        dt_ref = torch.tensor(self.dt_ref, device=x.device, dtype=x.dtype).clamp_min(1e-6)
        for t in range(L):
            x_t = x[:, :, t : t + 1]
            dt_t = dt_seq[:, t : t + 1]
            flow = self.hamiltonian(x_perm[:, t : t + 1])
            h_trans = self.lie_transport(curr_h, flow.squeeze(1), dt_t)
            h_next = self.koopman(h_trans, x_t, dt_t)
            x_inject = x_t.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, self.rank)
            dt_scaled = (dt_t / dt_ref).clamp_min(0.0)
            if self.inj_k > 0:
                g = 1.0 - torch.exp(-dt_scaled * torch.tensor(self.inj_k, device=x.device, dtype=x.dtype))
            else:
                g = dt_scaled
            curr_h = h_next + g.view(B, 1, 1, 1, 1) * x_inject
            h_states.append(curr_h)
        h_stack = torch.stack(h_states, dim=2)
        out = self.proj_out(h_stack.permute(0, 2, 3, 4, 1, 5)).squeeze(-1)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.post_ifft_proj(out)
        out = self.norm(out)
        gate = self.gate_conv(out)
        x_out = (1 - gate) * x + gate * out
        return x_out, curr_h


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], kernel_size: int = 7, use_cbam: bool = False, cond_channels: Optional[int] = None, conv_type: str = "conv"):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        self.dw_conv = FactorizedPeriodicConv3d(int(channels), int(channels), kernel_size=int(kernel_size), conv_type=str(conv_type))
        self.norm = SpatialGroupNorm(4, int(channels))
        self.cond_channels_spatial = int(cond_channels) if cond_channels is not None else 0
        self.cond_proj = nn.Conv3d(self.cond_channels_spatial, int(channels) * 2, kernel_size=1) if self.cond_channels_spatial > 0 else None
        self.pw_conv_in = nn.Conv3d(int(channels), int(channels) * 2, kernel_size=1)
        self.pw_conv_out = nn.Conv3d(int(channels), int(channels), kernel_size=1)
        self.cbam = CBAM2DPerStep(int(channels), reduction=16) if self.use_cbam else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        if self.cond_proj is not None and cond is not None:
            cond_in = cond.unsqueeze(2) if cond.dim() == 4 else cond
            if cond_in.shape[-2:] != x.shape[-2:]:
                cond_rs = F.interpolate(cond_in.squeeze(2), size=x.shape[-2:], mode="bilinear", align_corners=False).unsqueeze(2)
            else:
                cond_rs = cond_in
            affine = self.cond_proj(cond_rs)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta
        x = self.pw_conv_in(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * torch.sigmoid(x2)
        if self.cbam is not None:
            x = self.cbam(x)
        x = self.pw_conv_out(x)
        return residual + x


class FeedForward(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        self.ffn_ratio = float(getattr(args, "ffn_ratio", 4.0))
        self.hidden_dim = int(self.emb_ch * self.ffn_ratio)
        self.hidden_size = (int(input_downsp_shape[1]), int(input_downsp_shape[2]))
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.conv_type = str(getattr(args, "ConvType", "conv"))
        self.c_in = nn.Conv3d(self.emb_ch, self.hidden_dim, kernel_size=(1, 1, 1), padding="same")
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        self.block = GatedConvBlock(self.hidden_dim, self.hidden_size, kernel_size=7, use_cbam=self.use_cbam, cond_channels=cond_ch, conv_type=self.conv_type)
        self.c_out = nn.Conv3d(self.hidden_dim, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.c_in(x)
        x = self.act(x)
        x = self.block(x, cond=cond)
        x = self.c_out(x)
        return residual + x


class ConvLRUBlock(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.lru_layer = SimplifiedHKLFLayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_in: Optional[torch.Tensor] = None,
        listT: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mid, h_out = self.lru_layer(x, last_hidden_in, listT=listT, static_feats=static_feats)
        x_out = self.feed_forward(x_mid, cond=cond)
        return x_out, h_out


class BottleneckAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(dim) // int(num_heads)
        self.scale = head_dim ** -0.5
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(BN, N, Cc)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        out = shortcut + x_attn
        out = out.view(B, L, H, W, Cc).permute(0, 4, 1, 2, 3).contiguous()
        return out


class CrossScaleAttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.conv_g = nn.Conv3d(ch, ch, kernel_size=1, bias=False)
        self.conv_l = nn.Conv3d(ch, ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv3d(ch, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.norm = SpatialGroupNorm(4, ch)

    def forward(self, local_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        g = self.conv_g(global_x)
        l = self.conv_l(local_x)
        psi = self.relu(g + l)
        attn = self.sigmoid(self.psi(psi))
        return self.norm(local_x * attn)


class BiFPNFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.w1 = nn.Parameter(torch.ones(1, ch, 1, 1, 1))
        self.w2 = nn.Parameter(torch.ones(1, ch, 1, 1, 1))
        self.act = nn.SiLU()
        self.conv = nn.Conv3d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        w1 = F.relu(self.w1)
        w2 = F.relu(self.w2)
        weight = w1 + w2 + 1e-4
        out = (w1 * x + w2 * skip) / weight
        return self.conv(self.act(out))


class ShuffleDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(int(channels) * 4, int(channels), kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_hw_3d(x, 2, 2)
        return self.proj(x)


class ConvLRUModel(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.args = args
        self.arch_mode = str(getattr(args, "Arch", "unet")).lower()
        self.use_unet = self.arch_mode != "no_unet"
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
            self.mid_attention = None
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
                    if curr_H % 2 != 0:
                        curr_H += 1
                    if curr_W % 2 != 0:
                        curr_W += 1
                    curr_H = max(1, curr_H // 2)
                    curr_W = max(1, curr_W // 2)
            heads = 8
            for h in [8, 4, 2, 1]:
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
            self.fusion = nn.Conv3d(C * 2, C, 1) if self.arch_mode == "unet" else nn.Identity()
            self.convlru_blocks = None

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_ins: Optional[List[torch.Tensor]] = None,
        listT: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            last_hidden_outs: List[torch.Tensor] = []
            assert self.convlru_blocks is not None
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
                pad_h = x_s.shape[-2] % 2
                pad_w = x_s.shape[-1] % 2
                if pad_h > 0 or pad_w > 0:
                    x_s = F.pad(x_s, (0, pad_w, 0, pad_h))
                if x_s.shape[-2] >= 2 and x_s.shape[-1] >= 2:
                    x_s = self.downsamples[i](x_s)
                x = x_s
        assert self.mid_attention is not None
        x = self.mid_attention(x)
        for i, blk in enumerate(self.up_blocks):
            x = self.upsample(x)
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
            x, h_out = blk(x, hs_in_up[i], listT=listT, cond=curr_cond, static_feats=static_feats)
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
                nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.rH + 2, self.rW + 2), stride=(self.rH, self.rW), padding=(1, 1)),
                nn.SiLU(),
            )
        else:
            self.static_embed = None
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_size, kernel_size=7, use_cbam=False, cond_channels=cond_ch)])
        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.norm = SpatialGroupNorm(4, self.emb_ch)

    def _make_grid(self, args: Any) -> torch.Tensor:
        H, W = tuple(getattr(args, "input_size", (64, 64)))
        lat = torch.linspace(-math.pi / 2, math.pi / 2, int(H))
        lon = torch.linspace(0, 2 * math.pi, int(W))
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
        emb = torch.stack([torch.sin(grid_lat), torch.cos(grid_lat), torch.sin(grid_lon), torch.cos(grid_lon)], dim=0)
        return emb.unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 5:
            raise ValueError(f"Embedding expects [B,L,C,H,W], got {tuple(x.shape)}")
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
        self.static_ch = int(getattr(args, "static_ch", 0))
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        out_ch_after_up = self.emb_ch
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
        if self.head_mode == "gaussian":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
        elif self.head_mode in ["diffusion", "flow"]:
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = DiffusionHead(out_ch_after_up)
        else:
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_shuffle_conv(x)
        x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        if self.head_mode in ["diffusion", "flow"] and timestep is not None and self.time_embed is not None:
            t_emb = self.time_embed(timestep)
            x = x + t_emb.view(x.size(0), x.size(1), 1, 1, 1)
        x = self.c_out(x)
        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
                sigma = F.softplus(log_sigma.float()).to(mu.dtype) + 1e-6
            return torch.cat([mu, sigma], dim=1)
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

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "p",
        out_gen_num: Optional[int] = None,
        listT: Optional[torch.Tensor] = None,
        listT_future: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        revin_stats: Optional[RevINStats] = None,
    ):
        cond = None
        if self.embedding.static_ch > 0 and self.embedding.static_embed is not None and static_feats is not None:
            cond = self.embedding.static_embed(static_feats)
        if mode == "p":
            stats = revin_stats if revin_stats is not None else self.revin.stats(x)
            x_norm = self.revin(x, "norm", stats=stats)
            x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
            x_hid, last_hidden_outs = self.convlru_model(x_emb, listT=listT, cond=cond, static_feats=static_feats)
            out = self.decoder(x_hid, cond=cond, timestep=timestep)
            out_tensor = out.permute(0, 2, 1, 3, 4).contiguous()
            if str(self.decoder.head_mode).lower() == "gaussian":
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
        if listT is None:
            listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT
        stats = revin_stats if revin_stats is not None else self.revin.stats(x)
        out_list: List[torch.Tensor] = []
        x_norm = self.revin(x, "norm", stats=stats)
        x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT0, cond=cond, static_feats=static_feats)
        x_dec0 = self.decoder(x_hidden, cond=cond, timestep=timestep)
        x_step_dist = x_dec0.permute(0, 2, 1, 3, 4).contiguous()
        x_step_dist = x_step_dist[:, -1:, :, :, :]
        head_mode = str(self.decoder.head_mode).lower()
        if head_mode == "gaussian":
            out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
            mu = x_step_dist[:, :, :out_ch, :, :]
            sigma = x_step_dist[:, :, out_ch:, :, :]
            if mu.size(2) == self.revin.num_features:
                mu_denorm = self.revin(mu, "denorm", stats=stats)
                sigma_denorm = sigma * stats.stdev
            else:
                mu_denorm = mu
                sigma_denorm = sigma
            out_list.append(torch.cat([mu_denorm, sigma_denorm], dim=2))
            x_step_mean = mu_denorm
        else:
            if x_step_dist.size(2) == self.revin.num_features:
                x_step_dist_denorm = self.revin(x_step_dist, "denorm", stats=stats)
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
            if curr_x.dim() != 5:
                raise RuntimeError(f"i-mode expects curr_x [B,1,C,H,W], got {tuple(curr_x.shape)}")
            if curr_x.shape[1] != 1:
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
            x_step_norm = self.revin(curr_x, "norm", stats=stats) if curr_x.size(2) == self.revin.num_features else curr_x
            x_in, _ = self.embedding(x_step_norm, static_feats=static_feats)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt, cond=cond, static_feats=static_feats)
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            x_step_dist = x_dec.permute(0, 2, 1, 3, 4).contiguous()
            x_step_dist = x_step_dist[:, -1:, :, :, :]
            if head_mode == "gaussian":
                out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
                mu = x_step_dist[:, :, :out_ch, :, :]
                sigma = x_step_dist[:, :, out_ch:, :, :]
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin(mu, "denorm", stats=stats)
                    sigma = sigma * stats.stdev
                out_list.append(torch.cat([mu, sigma], dim=2))
                x_step_mean = mu
            else:
                if x_step_dist.size(2) == self.revin.num_features:
                    x_step_dist_denorm = self.revin(x_step_dist, "denorm", stats=stats)
                else:
                    x_step_dist_denorm = x_step_dist
                out_list.append(x_step_dist_denorm)
                x_step_mean = x_step_dist_denorm
        return torch.cat(out_list, dim=1)

