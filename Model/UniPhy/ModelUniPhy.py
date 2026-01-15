import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from UniPhyOps import UniPhyLayer

def get_safe_groups(channels: int, target: int = 4) -> int:
    return target if channels % target == 0 else 1

class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, L, H, W = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
            y = super().forward(y)
            return y.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return super().forward(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeAwareResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        scale_shift = self.mlp(time_emb)
        scale_shift = scale_shift.unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.block1(x)
        h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res_conv(x)

class DiffusionHead(nn.Module):
    def __init__(self, out_ch, emb_ch, hidden_factor=(2, 2)):
        super().__init__()
        self.emb_ch = emb_ch
        self.out_ch = out_ch
        self.rH, self.rW = hidden_factor
        
        time_dim = emb_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_ch),
            nn.Linear(emb_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.up_scale = nn.Upsample(scale_factor=(self.rH, self.rW), mode='nearest')
        
        self.pre_conv = nn.Conv2d(out_ch, emb_ch, 3, padding=1)
        
        self.block1 = TimeAwareResBlock(emb_ch * 2, emb_ch, time_dim)
        self.block2 = TimeAwareResBlock(emb_ch, emb_ch, time_dim)
        self.block3 = TimeAwareResBlock(emb_ch, emb_ch, time_dim)
        
        self.final = nn.Conv2d(emb_ch, out_ch, 1)

    def forward(self, x_cond, x_noisy, t):
        x_cond = x_cond.permute(0, 2, 1, 3, 4).contiguous()
        B, L, C, H, W = x_cond.shape
        x_cond_flat = x_cond.view(B * L, C, H, W)
        x_noisy_flat = x_noisy.reshape(B * L, -1, H * self.rH, W * self.rW)
        t_flat = t.reshape(B * L)
        time_emb = self.time_mlp(t_flat)
        x_cond_up = self.up_scale(x_cond_flat)
        x_noisy_emb = self.pre_conv(x_noisy_flat)
        h = torch.cat([x_noisy_emb, x_cond_up], dim=1)
        h = self.block1(h, time_emb)
        h = self.block2(h, time_emb)
        h = self.block3(h, time_emb)
        out = self.final(h)
        out = out.reshape(B, L, -1, H * self.rH, W * self.rW)
        return out

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
            self.spatial_conv = DeformConv2d(int(in_channels), int(in_channels), kernel_size=int(kernel_size), padding=0, bias=False, groups=int(in_channels))
        else:
            self.spatial_conv = nn.Conv2d(int(in_channels), int(in_channels), kernel_size=int(kernel_size), padding=0, bias=False, groups=int(in_channels))
        self.depth_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_sp = F.pad(x_reshaped, (self.pad_sp, self.pad_sp, 0, 0), mode="circular")
        x_sp = F.pad(x_sp, (0, 0, self.pad_sp, self.pad_sp), mode="replicate")
        x_sp = self.spatial_conv(x_sp)
        return self.depth_conv(x_sp).view(B, L, -1, H, W).permute(0, 2, 1, 3, 4)

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

class PhysicalRecurrentLayer(nn.Module):
    def __init__(
        self,
        emb_ch: int,
        input_shape: Tuple[int, int],
        rank: int = 32,
        **kwargs
    ):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.H, self.W = int(input_shape[0]), int(input_shape[1])
        self.core = UniPhyLayer(self.emb_ch, (self.H, self.W), rank=rank)

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor] = None, listT: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L, H, W = x.shape
        dt_seq_in = torch.ones(B, L, device=x.device, dtype=x.dtype) if listT is None else listT.to(x.device, x.dtype)
        dt_seq = _match_dt_seq(dt_seq_in, L)

        h = last_hidden_in
        outputs = []
        
        for t in range(L):
            x_t = x[:, :, t, :, :]
            dt_t = dt_seq[:, t:t+1]
            out, h = self.core(x_t, h, dt_t)
            outputs.append(out)

        x_out = torch.stack(outputs, dim=2)
        return x + x_out, h

class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], kernel_size: int = 7, conv_type: str = "conv"):
        super().__init__()
        self.dw_conv = PeriodicConv2d(int(channels), int(channels), kernel_size=int(kernel_size), conv_type=str(conv_type))
        self.norm = SpatialGroupNorm(get_safe_groups(int(channels)), int(channels))
        self.pw_conv_in = nn.Linear(int(channels), int(channels) * 2)
        self.pw_conv_out = nn.Linear(int(channels), int(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.pw_conv_in(x)
        x_val, x_gate = torch.chunk(x, 2, dim=-1)
        x = x_val * self.sigmoid(x_gate)
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
        self.prl_layer = PhysicalRecurrentLayer(emb_ch, input_shape, **prl_args)
        self.feed_forward = FeedForwardNetwork(emb_ch, input_shape, **ffn_args)

    def forward(self, x: torch.Tensor, last_hidden_in: Optional[torch.Tensor] = None, listT: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

            for i in range(layers - 2, -1, -1):
                h_up, w_up = encoder_res[i]
                self.up_blocks.append(UniPhyBlock(C, (h_up, w_up), prl_args, ffn_args))
                self.csa_blocks.append(CrossScaleGate(C))

            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.fusion = nn.Conv2d(C * 2, C, 1)
            self.uniphy_blocks = None

    def forward(self, x: torch.Tensor, last_hidden_ins: Optional[List[torch.Tensor]] = None, listT: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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

        assert self.mid_attention is not None
        x = self.mid_attention(x)

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
        return emb.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Embedding expects [B,L,C,H,W], got {tuple(x.shape)}")
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, L, C, H, W = x.shape
        
        grid = self.grid_embed.expand(B, L, -1, -1, -1).to(x.device, x.dtype)
        x = torch.cat([x, grid], dim=2)
        
        x_flat = x.reshape(B * L, -1, H, W)
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
        if self.dist_mode in ["gaussian", "laplace"]:
            self.final_out_ch = self.output_ch * 2
        elif self.dist_mode == "mdn":
            self.num_mixtures = 3
            self.final_out_ch = self.output_ch * 3 * self.num_mixtures
        else:
            self.final_out_ch = self.output_ch
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
        x = x_flat.view(B, L, C_out, H_out, W_out)
        if self.dist_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=2)
            with torch.amp.autocast("cuda", enabled=False):
                log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
                sigma = F.softplus(log_sigma.float()).to(mu.dtype) + 1e-3
            x_res = torch.cat([mu, sigma], dim=2)
            return x_res.permute(0, 2, 1, 3, 4)
        if self.dist_mode == "laplace":
            mu, log_b = torch.chunk(x, 2, dim=2)
            with torch.amp.autocast("cuda", enabled=False):
                log_b = torch.clamp(log_b, min=-5.0, max=5.0)
                b = F.softplus(log_b.float()).to(mu.dtype) + 1e-3
            x_res = torch.cat([mu, b], dim=2)
            return x_res.permute(0, 2, 1, 3, 4)
        return x.permute(0, 2, 1, 3, 4)

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
        
        prl_args = {
            "rank": rank,
        }

        ffn_ratio = float(getattr(args, "ffn_ratio", 4.0))
        conv_type = str(getattr(args, "ConvType", "conv"))
        ffn_args = {"ffn_ratio": ffn_ratio, "conv_type": conv_type}

        self.uniphy_model = UniPhyBackbone(
            emb_ch,
            self.embedding.input_downsp_shape[1:],
            num_blocks,
            arch,
            down_mode,
            prl_args,
            ffn_args,
        )

        out_ch = int(getattr(args, "out_ch", 1))
        dist_mode = getattr(args, "dist_mode", "gaussian")
        
        if dist_mode == "diffusion":
            self.decoder = DiffusionHead(out_ch, emb_ch, hidden_factor)
        else:
            self.decoder = ProbabilisticDecoder(out_ch, emb_ch, dist_mode, hidden_factor)

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
        sample: bool = False,
        **kwargs
    ):
        x_noisy = kwargs.get("x_noisy", None)
        t = kwargs.get("t", None)

        if mode == "p":
            x_emb = self.embedding(x)
            x_hid, last_hidden_outs = self.uniphy_model(x_emb, listT=listT)
            
            if isinstance(self.decoder, DiffusionHead):
                if x_noisy is not None and t is not None:
                    out = self.decoder(x_hid, x_noisy, t)
                else:
                    B, C, L, H, W = x.shape
                    x_noisy = torch.randn(B, L, self.decoder.out_ch, H, W, device=x.device, dtype=x.dtype)
                    steps = 10
                    for i in reversed(range(steps)):
                        t_tensor = torch.full((B * L,), i, device=x.device, dtype=torch.float)
                        pred_noise = self.decoder(x_hid, x_noisy, t_tensor)
                        x_noisy = x_noisy - 0.1 * pred_noise 
                    out = x_noisy
                out_tensor = out.permute(0, 2, 1, 3, 4).contiguous()
            else:
                out = self.decoder(x_hid)
                out_tensor = out
            
            return out_tensor, last_hidden_outs

        if out_gen_num is None or int(out_gen_num) <= 0:
            raise ValueError("out_gen_num must be positive for inference mode")

        B = x.size(0)
        listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype) if listT is None else listT
        out_list: List[torch.Tensor] = []

        x_emb = self.embedding(x)
        x_hidden, last_hidden_outs = self.uniphy_model(x_emb, listT=listT0)
        
        if isinstance(self.decoder, DiffusionHead):
            B, C, L, H, W = x.shape
            x_noisy = torch.randn(B, L, self.decoder.out_ch, H, W, device=x.device, dtype=x.dtype)
            t_tensor = torch.zeros(B * L, device=x.device)
            out = self.decoder(x_hidden, x_noisy, t_tensor)
            x_dec0 = out.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x_dec0 = self.decoder(x_hidden)

        x_step_dist = x_dec0
        x_step_dist_slice = x_step_dist[:, :, -1:, :, :]
        
        out_list.append(x_step_dist_slice.cpu())
        curr_x_phys = x_step_dist_slice

        future = torch.ones(B, int(out_gen_num) - 1, device=x.device, dtype=x.dtype) if listT_future is None else listT_future

        for t in range(int(out_gen_num) - 1):
            dt = future[:, t : t + 1]
            if curr_x_phys.shape[2] != 1:
                curr_x_phys = curr_x_phys[:, :, -1:, :, :]
            
            x_next_input = curr_x_phys[:, :self.embedding.input_ch, :, :, :]
            
            x_in = self.embedding(x_next_input)
            x_hidden, last_hidden_outs = self.uniphy_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt)
            if isinstance(self.decoder, DiffusionHead):
                x_noisy_t = torch.randn(B, 1, self.decoder.out_ch, H, W, device=x.device, dtype=x.dtype)
                x_dec = self.decoder(x_hidden, x_noisy_t, torch.zeros(B, device=x.device))
                x_dec = x_dec.permute(0, 2, 1, 3, 4)
            else:
                x_dec = self.decoder(x_hidden)
            
            x_step_dist = x_dec
            out_list.append(x_step_dist.cpu())
            curr_x_phys = x_step_dist

        return torch.cat(out_list, dim=2), last_hidden_outs

