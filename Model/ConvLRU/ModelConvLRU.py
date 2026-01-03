import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr_conv3d_weight_(weight: torch.Tensor, r_h: int, r_w: int) -> torch.Tensor:
    out_ch, in_ch, k_d, k_h, k_w = weight.shape
    base_out = out_ch // (r_h * r_w)
    base = weight.new_zeros((base_out, in_ch, 1, k_h, k_w))
    nn.init.kaiming_normal_(base, a=0, mode="fan_in", nonlinearity="relu")
    base = base.repeat_interleave(r_h * r_w, dim=0)
    with torch.no_grad():
        weight.copy_(base)
    return weight


def pixel_shuffle_hw_3d(x: torch.Tensor, r_h: int, r_w: int) -> torch.Tensor:
    n, c_mul, d, h, w = x.shape
    c = c_mul // (r_h * r_w)
    x = x.reshape(n, c, r_h, r_w, d, h, w)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.reshape(n, c, d, h * r_h, w * r_w)
    return x


def pixel_unshuffle_hw_3d(x: torch.Tensor, r_h: int, r_w: int) -> torch.Tensor:
    n, c, d, h, w = x.shape
    x = x.reshape(n, c, d, h // r_h, r_h, w // r_w, r_w)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.reshape(n, c * r_h * r_w, d, h // r_h, w // r_w)
    return x


class DeformConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.conv_offset = nn.Conv2d(
            in_channels,
            3 * self.kernel_size * self.kernel_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self._dcn: Optional[nn.Module] = None
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)
        self.bias = bool(bias)
        self._init_weights()

    def _ensure_dcn(self, device: torch.device, dtype: torch.dtype) -> nn.Module:
        if self._dcn is None:
            import torchvision

            self._dcn = torchvision.ops.DeformConv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            ).to(device=device, dtype=dtype)
        return self._dcn

    def _init_weights(self) -> None:
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        dcn = self._ensure_dcn(x.device, x.dtype)
        return dcn(x, offset, mask)


class FactorizedPeriodicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, conv_type: str = "conv"):
        super().__init__()
        self.pad_sp = int(kernel_size) // 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv_type = str(conv_type)

        if self.conv_type == "dcn":
            self.spatial_conv = DeformConv2d(self.in_channels, self.out_channels, kernel_size=int(kernel_size), padding=0, bias=False)
        else:
            self.spatial_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=int(kernel_size), padding=0, bias=False)

        self.depth_conv = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=(1, 1, 1), padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_sp = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        x_sp = F.pad(x_sp, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        c_out = x_sp.shape[1]
        x_sp = x_sp.reshape(b, d, c_out, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return self.depth_conv(x_sp)


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        ch = int(channels)
        red = int(reduction)
        hidden = max(ch // max(red, 1), 4)
        self.mlp = nn.Sequential(
            nn.Linear(ch, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bl, c, h, w = x.shape
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)
        max_pool, _ = torch.max(x.reshape(bl, c, -1), dim=-1)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(bl, c, 1, 1)
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
        b, c, l, h, w = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * l, c, h, w)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        return x_flat.view(b, l, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class HamiltonianGenerator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch // 2, 1, 3, padding=1),
        )
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
        self._metric_cache: dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

    def _metric(self, device: torch.device, dtype: torch.dtype, h: int) -> torch.Tensor:
        key = (device, dtype, int(h))
        if key not in self._metric_cache:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device, dtype=dtype).view(1, 1, h, 1)
            self._metric_cache[key] = 1.0 / (torch.cos(lat) + 1e-6)
        return self._metric_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c, h, w = x.shape
        x_flat = x.reshape(b * l, c, h, w)
        h_field = self.conv(x_flat)
        metric_correction = self._metric(x.device, h_field.dtype, h)
        h_padded = F.pad(h_field, (1, 1, 0, 0), mode="circular")
        h_padded = F.pad(h_padded, (0, 0, 1, 1), mode="replicate")
        grad_x = F.conv2d(h_padded, self.sobel_x.to(device=x.device, dtype=h_field.dtype))
        grad_y = F.conv2d(h_padded, self.sobel_y.to(device=x.device, dtype=h_field.dtype))
        u = grad_y
        v = -grad_x * metric_correction
        flow = torch.cat([u, v], dim=1)
        flow = torch.tanh(flow)
        return flow.view(b, l, 2, h, w)


class LieTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid_cache: dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

    def _base_grid(self, device: torch.device, dtype: torch.dtype, h: int, w: int) -> torch.Tensor:
        key = (device, dtype, int(h), int(w))
        if key not in self._grid_cache:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device, dtype=dtype),
                torch.linspace(-1, 1, w, device=device, dtype=dtype),
                indexing="ij",
            )
            self._grid_cache[key] = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
        return self._grid_cache[key]

    def forward(self, h_prev: torch.Tensor, flow: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        b, c, h, w, r = h_prev.shape
        dt = dt.view(b, 1, 1, 1)
        base_grid = self._base_grid(h_prev.device, h_prev.dtype, h, w)
        chunk_size = 4
        out_chunks: List[torch.Tensor] = []
        for i in range(0, r, chunk_size):
            r_end = min(i + chunk_size, r)
            curr_r = r_end - i
            h_chunk = h_prev[..., i:r_end]
            h_flat = h_chunk.permute(0, 4, 1, 2, 3).reshape(b * curr_r, c, h, w)
            flow_chunk = flow.unsqueeze(1).expand(b, curr_r, 2, h, w).reshape(b * curr_r, 2, h, w)
            dt_chunk = dt.repeat_interleave(curr_r, dim=0)
            flow_dt = flow_chunk * dt_chunk
            sampling_grid = base_grid.expand(b * curr_r, -1, -1, -1) - flow_dt.permute(0, 2, 3, 1)
            h_warped = F.grid_sample(h_flat, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)
            out_chunks.append(h_warped.reshape(b, curr_r, c, h, w).permute(0, 2, 3, 4, 1))
        return torch.cat(out_chunks, dim=4)


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
        b, c, h, w = x.shape
        feat = self.conv(x)
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 3, 1).reshape(b, 1, self.w_freq, c)
        params = self.head(feat)
        params = params.view(b, 1, self.w_freq, self.emb_ch, self.rank, 3)
        params = params.permute(0, 1, 3, 2, 4, 5).contiguous()
        nu = F.softplus(params[..., 0])
        theta = torch.tanh(params[..., 1]) * math.pi
        sigma = torch.sigmoid(params[..., 2])
        return nu, theta, sigma


class SpectralKoopmanSDE(nn.Module):
    def __init__(self, channels: int, rank: int, w_freq: int, noise_fn: Optional[Callable[[torch.Size, torch.device, torch.dtype], torch.Tensor]] = None):
        super().__init__()
        ch = int(channels)
        self.rank = int(rank)
        self.channels = ch
        self.w_freq = int(w_freq)
        self.estimator = KoopmanParamEstimator(ch, ch, self.rank, self.w_freq)
        self.noise_fn = noise_fn

    def _noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.noise_fn is not None:
            return self.noise_fn(shape, device, dtype)
        return torch.randn(shape, device=device, dtype=dtype)

    def forward(self, h_trans: torch.Tensor, x_curr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        b, c, h, w, r = h_trans.shape
        h_freq = torch.fft.rfft2(h_trans.permute(0, 4, 1, 2, 3).float(), norm="ortho")
        x_flat = x_curr.squeeze(2)
        nu, theta, sigma = self.estimator(x_flat)
        dt_expanded = dt.view(b, dt.shape[1], 1, 1, 1, 1)
        nu = nu.unsqueeze(4) * dt_expanded
        theta = theta.unsqueeze(4) * dt_expanded
        sigma = sigma.unsqueeze(4)
        decay = torch.exp(-nu)
        rotate = torch.exp(1j * theta)
        lambda_k = decay * rotate
        lambda_k = lambda_k.squeeze(1).permute(0, 4, 1, 3, 2)
        sigma_b = sigma.squeeze(1).permute(0, 4, 1, 3, 2)
        noise_scale = sigma_b.mean(dim=-1, keepdim=True)
        noise = self._noise(h_freq.shape, h_freq.device, torch.float32).to(h_freq.dtype) * noise_scale
        h_evolved = h_freq * lambda_k + noise
        h_out = torch.fft.irfft2(h_evolved, s=(h, w), norm="ortho")
        return h_out.permute(0, 2, 3, 4, 1)


class StaticInitState(nn.Module):
    def __init__(self, static_ch: int, emb_ch: int, rank: int, s: int, w_freq: int):
        super().__init__()
        self.static_ch = int(static_ch)
        self.emb_ch = int(emb_ch)
        self.rank = int(rank)
        self.s = int(s)
        self.w_freq = int(w_freq)
        self.mapper = nn.Sequential(
            nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.s, self.w_freq)),
            nn.Conv2d(self.emb_ch, self.emb_ch * self.rank, kernel_size=1),
        )

    def forward(self, static_feats: torch.Tensor) -> torch.Tensor:
        b = static_feats.size(0)
        out = self.mapper(static_feats)
        out = out.view(b, self.emb_ch, self.rank, self.s, self.w_freq)
        return out.permute(0, 1, 3, 4, 2)


class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, c, l, h, w = x.shape
            x2 = x.permute(0, 2, 1, 3, 4).reshape(b * l, c, h, w)
            x2 = super().forward(x2)
            return x2.view(b, l, c, h, w).permute(0, 2, 1, 3, 4)
        return super().forward(x)


class SimplifiedHKLFLayer(nn.Module):
    def __init__(self, args: Any, input_downsp_shape: Tuple[int, int, int]):
        super().__init__()
        self.emb_ch = int(getattr(args, "emb_ch", 32))
        h, w = int(input_downsp_shape[1]), int(input_downsp_shape[2])
        self.rank = int(getattr(args, "lru_rank", 32))
        self.w_freq = w // 2 + 1
        self.hamiltonian = HamiltonianGenerator(self.emb_ch)
        self.lie_transport = LieTransport()

        noise_mode = str(getattr(args, "sde_noise_mode", "default")).lower()
        if noise_mode in {"none", "off", "zero", "0"}:
            noise_fn = lambda shape, device, dtype: torch.zeros(shape, device=device, dtype=dtype)
        else:
            noise_fn = None

        self.koopman = SpectralKoopmanSDE(self.emb_ch, self.rank, self.w_freq, noise_fn=noise_fn)
        self.proj_out = nn.Linear(self.rank, 1)

        if bool(getattr(args, "learnable_init_state", False)) and int(getattr(args, "static_ch", 0)) > 0:
            self.init_state = StaticInitState(int(args.static_ch), self.emb_ch, self.rank, h, w)
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
        list_t: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, l, h, w = x.shape
        if last_hidden_in is None:
            if self.init_state is not None and static_feats is not None:
                h_prev = self.init_state(static_feats).unsqueeze(1).expand(-1, l, -1, -1, -1, -1)[:, 0]
            else:
                h_prev = torch.zeros(b, c, h, w, self.rank, device=x.device, dtype=x.dtype)
        else:
            h_prev = last_hidden_in

        if list_t is None:
            dt_seq = torch.ones(b, l, device=x.device, dtype=x.dtype)
        else:
            dt_seq = list_t.to(device=x.device, dtype=x.dtype)

        h_states: List[torch.Tensor] = []
        curr_h = h_prev
        x_perm = x.permute(0, 2, 1, 3, 4)

        for t in range(l):
            x_t = x[:, :, t : t + 1]
            dt_t = dt_seq[:, t].view(b, 1)
            flow = self.hamiltonian(x_perm[:, t : t + 1])
            h_trans = self.lie_transport(curr_h, flow.squeeze(1), dt_t)
            h_next = self.koopman(h_trans, x_t, dt_t)
            x_inject = x_t.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, self.rank)
            curr_h = h_next + x_inject
            h_states.append(curr_h)

        h_stack = torch.stack(h_states, dim=2)
        out = self.proj_out(h_stack.permute(0, 2, 3, 4, 1, 5)).squeeze(-1)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.post_ifft_proj(out)
        out = self.norm(out)
        gate = self.gate_conv(out)
        x_out = (1.0 - gate) * x + gate * out
        return x_out, curr_h


class GatedConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_size: Tuple[int, int],
        kernel_size: int = 7,
        use_cbam: bool = False,
        cond_channels: Optional[int] = None,
        conv_type: str = "conv",
    ):
        super().__init__()
        self.use_cbam = bool(use_cbam)
        ch = int(channels)
        self.dw_conv = FactorizedPeriodicConv3d(ch, ch, kernel_size=int(kernel_size), conv_type=str(conv_type))
        self.norm = SpatialGroupNorm(4, ch)
        self.cond_channels_spatial = int(cond_channels) if cond_channels is not None else 0
        self.cond_proj = nn.Conv3d(self.cond_channels_spatial, ch * 2, kernel_size=1) if self.cond_channels_spatial > 0 else None
        self.pw_conv_in = nn.Conv3d(ch, ch * 2, kernel_size=1)
        self.pw_conv_out = nn.Conv3d(ch, ch, kernel_size=1)
        self.cbam = CBAM2DPerStep(ch, reduction=16) if self.use_cbam else None

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
            x = x * (1.0 + gamma) + beta

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
        list_t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mid, h_out = self.lru_layer(x, last_hidden_in, list_t=list_t, static_feats=static_feats)
        x_out = self.feed_forward(x_mid, cond=cond)
        return x_out, h_out


class BottleneckAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(dim) // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(int(dim), int(dim) * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(int(dim), int(dim))
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(b * l, h * w, c)
        b_n, n, c2 = x_flat.shape
        shortcut = x_flat
        x_norm = self.norm(x_flat)
        qkv = self.qkv(x_norm).reshape(b_n, n, 3, self.num_heads, c2 // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(b_n, n, c2)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        out = shortcut + x_attn
        return out.view(b, l, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


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
        ch = int(channels)
        self.proj = nn.Conv3d(ch * 4, ch, kernel_size=1, bias=False)

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

        c = int(getattr(args, "emb_ch", input_downsp_shape[0]))
        h, w = int(input_downsp_shape[1]), int(input_downsp_shape[2])

        if not self.use_unet:
            self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, (c, h, w)) for _ in range(layers)])
            self.upsample = None
            self.fusion = None
            self.mid_attention = None
        else:
            curr_h, curr_w = h, w
            encoder_res: List[Tuple[int, int]] = []
            for i in range(layers):
                self.down_blocks.append(ConvLRUBlock(self.args, (c, curr_h, curr_w)))
                encoder_res.append((curr_h, curr_w))
                if i < layers - 1:
                    if self.down_mode == "conv":
                        self.downsamples.append(nn.Conv3d(c, c, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
                    elif self.down_mode == "shuffle":
                        self.downsamples.append(ShuffleDownsample(c))
                    else:
                        self.downsamples.append(nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

                    if curr_h % 2 != 0:
                        curr_h += 1
                    if curr_w % 2 != 0:
                        curr_w += 1
                    curr_h = max(1, curr_h // 2)
                    curr_w = max(1, curr_w // 2)

            heads = 8
            for cand in [8, 4, 2, 1]:
                if c % cand == 0:
                    heads = cand
                    break
            self.mid_attention = BottleneckAttention(c, num_heads=heads)

            for i in range(layers - 2, -1, -1):
                h_up, w_up = encoder_res[i]
                self.up_blocks.append(ConvLRUBlock(self.args, (c, h_up, w_up)))
                if self.arch_mode == "bifpn":
                    self.csa_blocks.append(BiFPNFusion(c))
                else:
                    self.csa_blocks.append(CrossScaleAttentionGate(c))

            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
            if self.arch_mode == "unet":
                self.fusion = nn.Conv3d(c * 2, c, 1)
            else:
                self.fusion = nn.Identity()
            self.convlru_blocks = None

    def forward(
        self,
        x: torch.Tensor,
        last_hidden_ins: Optional[List[torch.Tensor]] = None,
        list_t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.use_unet:
            outs: List[torch.Tensor] = []
            assert self.convlru_blocks is not None
            for idx, blk in enumerate(self.convlru_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None and idx < len(last_hidden_ins)) else None
                x, h_out = blk(x, h_in, list_t=list_t, cond=cond, static_feats=static_feats)
                outs.append(h_out)
            return x, outs

        skips: List[torch.Tensor] = []
        last_hidden_outs: List[torch.Tensor] = []
        num_down = len(self.down_blocks)
        hs_in_down = last_hidden_ins[:num_down] if last_hidden_ins is not None else [None] * num_down
        hs_in_up = last_hidden_ins[num_down:] if last_hidden_ins is not None else [None] * len(self.up_blocks)

        for i, blk in enumerate(self.down_blocks):
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x, h_out = blk(x, hs_in_down[i], list_t=list_t, cond=curr_cond, static_feats=static_feats)
            last_hidden_outs.append(h_out)
            if i < len(self.down_blocks) - 1:
                skips.append(x)
                x_s = x
                if self.down_mode in {"shuffle", "avg", "conv"}:
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
            assert self.upsample is not None
            x = self.upsample(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.size(-2) - x.size(-2)
                diff_x = skip.size(-1) - x.size(-1)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

            if self.arch_mode == "bifpn":
                x = self.csa_blocks[i](x, skip)
            else:
                skip2 = self.csa_blocks[i](skip, x)
                x = torch.cat([x, skip2], dim=1)
                assert self.fusion is not None
                x = self.fusion(x)

            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x, h_out = blk(x, hs_in_up[i], list_t=list_t, cond=curr_cond, static_feats=static_feats)
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
        self.r_h, self.r_w = int(hf[0]), int(hf[1])
        self.input_ch_total = self.input_ch + 4
        self.patch_embed = nn.Conv3d(
            self.input_ch_total,
            self.emb_hidden_ch,
            kernel_size=(1, self.r_h + 2, self.r_w + 2),
            stride=(1, self.r_h, self.r_w),
            padding=(0, 1, 1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch_total, 1, int(self.input_size[0]), int(self.input_size[1]))
            out_dummy = self.patch_embed(dummy)
            _, _, _, h, w = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, int(h), int(w))

        self.register_buffer("grid_embed", self._make_grid(args), persistent=False)
        self.hidden_size = (int(self.input_downsp_shape[1]), int(self.input_downsp_shape[2]))

        if self.static_ch > 0:
            self.static_embed = nn.Sequential(
                nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.r_h + 2, self.r_w + 2), stride=(self.r_h, self.r_w), padding=(1, 1)),
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
        h, w = tuple(getattr(args, "input_size", (64, 64)))
        lat = torch.linspace(-math.pi / 2, math.pi / 2, h)
        lon = torch.linspace(0, 2 * math.pi, w)
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
        emb = torch.stack([torch.sin(grid_lat), torch.cos(grid_lat), torch.sin(grid_lon), torch.cos(grid_lon)], dim=0)
        return emb.unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        b, c, l, h, w = x.shape
        grid = self.grid_embed.expand(b, -1, l, -1, -1).to(device=x.device, dtype=x.dtype)
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
        hf = getattr(args, "hidden_factor", (2, 2))
        self.r_h, self.r_w = int(hf[0]), int(hf[1])

        out_ch_after_up = self.emb_ch
        self.pre_shuffle_conv = nn.Conv3d(
            in_channels=self.emb_ch,
            out_channels=out_ch_after_up * self.r_h * self.r_w,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.r_h, self.r_w)
        with torch.no_grad():
            if self.pre_shuffle_conv.bias is not None:
                self.pre_shuffle_conv.bias.zero_()

        if self.head_mode == "gaussian":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None
        elif self.head_mode in {"diffusion", "flow"}:
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = DiffusionHead(out_ch_after_up)
        else:
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = None

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_shuffle_conv(x)
        x = pixel_shuffle_hw_3d(x, self.r_h, self.r_w)
        x = self.activation(x)

        if self.head_mode in {"diffusion", "flow"} and timestep is not None and self.time_embed is not None:
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

    def stats(self, x: torch.Tensor) -> RevINStats:
        dim2reduce = (3, 4)
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            mean = torch.mean(x_fp32, dim=dim2reduce, keepdim=True).detach().to(x.dtype)
            stdev = torch.sqrt(torch.var(x_fp32, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach().to(x.dtype)
        return RevINStats(mean=mean, stdev=stdev)

    def normalize(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        x = (x - stats.mean) / stats.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * stats.stdev + stats.mean
        return x

    def forward(self, x: torch.Tensor, mode: str, stats: Optional[RevINStats] = None) -> Tuple[torch.Tensor, RevINStats]:
        if stats is None:
            stats = self.stats(x)
        if mode == "norm":
            return self.normalize(x, stats), stats
        if mode == "denorm":
            return self.denormalize(x, stats), stats
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
            x_norm, stats = self.revin(x, "norm", stats=revin_stats)
            x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
            x_hid, last_hidden_outs = self.convlru_model(x_emb, list_t=listT, cond=cond, static_feats=static_feats)
            out = self.decoder(x_hid, cond=cond, timestep=timestep)
            out_tensor = out.permute(0, 2, 1, 3, 4).contiguous()

            if str(self.decoder.head_mode).lower() == "gaussian":
                mu, sigma = torch.chunk(out_tensor, 2, dim=2)
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin.denormalize(mu, stats)
                    sigma = sigma * stats.stdev
                return torch.cat([mu, sigma], dim=2), last_hidden_outs
            if out_tensor.size(2) == self.revin.num_features:
                out_tensor = self.revin.denormalize(out_tensor, stats)
            return out_tensor, last_hidden_outs

        if out_gen_num is None or int(out_gen_num) <= 0:
            raise ValueError("out_gen_num must be positive for inference mode")

        b = x.size(0)
        if listT is None:
            listT0 = torch.ones(b, x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT

        out_list: List[torch.Tensor] = []

        x_norm, stats = self.revin(x, "norm", stats=revin_stats)
        if stats.mean.ndim == 5:
            stats = RevINStats(mean=stats.mean[:, -1:, :, :, :], stdev=stats.stdev[:, -1:, :, :, :])

        x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, list_t=listT0, cond=cond, static_feats=static_feats)
        x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
        x_step_dist = x_dec.permute(0, 2, 1, 3, 4).contiguous()
        x_step_dist = x_step_dist[:, -1:, :, :, :]

        if str(self.decoder.head_mode).lower() == "gaussian":
            out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
            mu = x_step_dist[:, :, :out_ch, :, :]
            sigma = x_step_dist[:, :, out_ch:, :, :]
            if mu.size(2) == self.revin.num_features:
                mu_den = self.revin.denormalize(mu, stats)
                sigma_den = sigma * stats.stdev
            else:
                mu_den = mu
                sigma_den = sigma
            out_list.append(torch.cat([mu_den, sigma_den], dim=2))
            x_step_mean = mu_den
        else:
            if x_step_dist.size(2) == self.revin.num_features:
                x_step_dist_den = self.revin.denormalize(x_step_dist, stats)
            else:
                x_step_dist_den = x_step_dist
            out_list.append(x_step_dist_den)
            x_step_mean = x_step_dist_den

        future = listT_future
        if future is None:
            future = torch.ones(b, int(out_gen_num) - 1, device=x.device, dtype=x.dtype)

        for t in range(int(out_gen_num) - 1):
            dt = future[:, t : t + 1]
            curr_x = x_step_mean
            if curr_x.ndim == 5 and curr_x.shape[1] != 1:
                curr_x = curr_x[:, -1:, :, :, :]

            if curr_x.size(2) != self.embedding.input_ch:
                b_in, l_in, c_out, h_in, w_in = curr_x.shape
                c_target = self.embedding.input_ch
                if c_out > c_target:
                    curr_x = curr_x[:, :, :c_target, :, :]
                else:
                    diff = c_target - c_out
                    zeros = torch.zeros(b_in, l_in, diff, h_in, w_in, device=curr_x.device, dtype=curr_x.dtype)
                    curr_x = torch.cat([curr_x, zeros], dim=2)

            if curr_x.size(2) == self.revin.num_features:
                x_step_norm = self.revin.normalize(curr_x, stats)
            else:
                x_step_norm = curr_x

            x_in, _ = self.embedding(x_step_norm, static_feats=static_feats)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, list_t=dt, cond=cond, static_feats=static_feats)
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            x_step_dist = x_dec.permute(0, 2, 1, 3, 4).contiguous()
            x_step_dist = x_step_dist[:, -1:, :, :, :]

            if str(self.decoder.head_mode).lower() == "gaussian":
                out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
                mu = x_step_dist[:, :, :out_ch, :, :]
                sigma = x_step_dist[:, :, out_ch:, :, :]
                if mu.size(2) == self.revin.num_features:
                    mu_den = self.revin.denormalize(mu, stats)
                    sigma_den = sigma * stats.stdev
                else:
                    mu_den = mu
                    sigma_den = sigma
                out_list.append(torch.cat([mu_den, sigma_den], dim=2))
                x_step_mean = mu_den
            else:
                if x_step_dist.size(2) == self.revin.num_features:
                    x_step_dist_den = self.revin.denormalize(x_step_dist, stats)
                else:
                    x_step_dist_den = x_step_dist
                out_list.append(x_step_dist_den)
                x_step_mean = x_step_dist_den

        return torch.cat(out_list, dim=1)

