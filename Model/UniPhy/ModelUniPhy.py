import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=3),
            triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=4),
            triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=4),
            triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}, num_stages=4),
        ],
        key=['B', 'R', 'C', 'H', 'W']
    )
    @triton.jit
    def koopman_kernel(
        h_real_ptr, h_imag_ptr,
        nu_ptr, theta_ptr,
        dt_ptr,
        out_real_ptr, out_imag_ptr,
        B, R, C, H, W,
        stride_h_b, stride_h_r, stride_h_c, stride_h_h, stride_h_w,
        stride_p_b, stride_p_r, stride_p_c, stride_p_h, stride_p_w,
        stride_dt_b,
        stride_o_b, stride_o_r, stride_o_c, stride_o_h, stride_o_w,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        num_elements = B * R * C * H * W
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        idx_w = offsets % W
        tmp = offsets // W
        idx_h = tmp % H
        tmp = tmp // H
        idx_c = tmp % C
        tmp = tmp // C
        idx_r = tmp % R
        idx_b = tmp // R

        dt_val = tl.load(dt_ptr + idx_b * stride_dt_b, mask=mask, other=0.0)
        
        p_offset = (idx_b * stride_p_b + idx_r * stride_p_r + idx_c * stride_p_c + idx_w * stride_p_w)
        nu = tl.load(nu_ptr + p_offset, mask=mask, other=0.0)
        theta = tl.load(theta_ptr + p_offset, mask=mask, other=0.0)
        
        h_offset = (idx_b * stride_h_b + idx_r * stride_h_r + idx_c * stride_h_c + idx_h * stride_h_h + idx_w * stride_h_w)
        h_real = tl.load(h_real_ptr + h_offset, mask=mask, other=0.0)
        h_imag = tl.load(h_imag_ptr + h_offset, mask=mask, other=0.0)

        decay = tl.exp(-nu * dt_val)
        angle = theta * dt_val
        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)
        
        rot_real = h_real * cos_a - h_imag * sin_a
        rot_imag = h_real * sin_a + h_imag * cos_a
        
        res_real = rot_real * decay
        res_imag = rot_imag * decay
        
        out_offset = (idx_b * stride_o_b + idx_r * stride_o_r + idx_c * stride_o_c + idx_h * stride_o_h + idx_w * stride_o_w)
        tl.store(out_real_ptr + out_offset, res_real, mask=mask)
        tl.store(out_imag_ptr + out_offset, res_imag, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=3),
            triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=4),
            triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=4),
        ],
        key=['total_groups', 'H', 'W']
    )
    @triton.jit
    def warp_grid_kernel(
        flow_ptr, dt_ptr, out_grid_ptr,
        total_groups, H, W,
        stride_f_g, stride_f_c, stride_f_h, stride_f_w,
        stride_dt_b,
        stride_o_g, stride_o_h, stride_o_w, stride_o_c,
        groups_per_batch,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        num_pixels = total_groups * H * W
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_pixels

        idx_w = offsets % W
        tmp = offsets // W
        idx_h = tmp % H
        idx_g = tmp // H 

        idx_b = idx_g // groups_per_batch
        
        dt_val = tl.load(dt_ptr + idx_b * stride_dt_b, mask=mask, other=0.0)

        off_u = idx_g * stride_f_g + 0 * stride_f_c + idx_h * stride_f_h + idx_w * stride_f_w
        flow_u = tl.load(flow_ptr + off_u, mask=mask, other=0.0)
        
        off_v = idx_g * stride_f_g + 1 * stride_f_c + idx_h * stride_f_h + idx_w * stride_f_w
        flow_v = tl.load(flow_ptr + off_v, mask=mask, other=0.0)

        w_norm = (idx_w.to(tl.float32) / (W - 1)) * 2.0 - 1.0
        h_norm = (idx_h.to(tl.float32) / (H - 1)) * 2.0 - 1.0

        grid_x_raw = w_norm - flow_u * dt_val
        grid_y = h_norm - flow_v * dt_val

        grid_x_shifted = grid_x_raw + 1.0
        grid_x_mod = grid_x_shifted - 2.0 * tl.floor(grid_x_shifted * 0.5)
        grid_x = grid_x_mod - 1.0

        off_out_x = idx_g * stride_o_g + idx_h * stride_o_h + idx_w * stride_o_w + 0 * stride_o_c
        tl.store(out_grid_ptr + off_out_x, grid_x, mask=mask)
        
        off_out_y = idx_g * stride_o_g + idx_h * stride_o_h + idx_w * stride_o_w + 1 * stride_o_c
        tl.store(out_grid_ptr + off_out_y, grid_y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=3),
            triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=4),
            triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}, num_stages=4),
        ],
        key=['n_elements_out', 'C_out']
    )
    @triton.jit
    def glu_kernel(
        x_ptr, out_ptr,
        n_elements_out, C_out,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_out
        
        idx_c = offsets % C_out
        idx_row = offsets // C_out
        
        idx_val = idx_row * (2 * C_out) + idx_c
        idx_gate = idx_val + C_out
        
        val = tl.load(x_ptr + idx_val, mask=mask, other=0.0)
        gate = tl.load(x_ptr + idx_gate, mask=mask, other=0.0)
        
        res = val * tl.sigmoid(gate)
        tl.store(out_ptr + offsets, res, mask=mask)

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

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

class SpatialGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, L, H, W = x.shape
            y = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
            y = super().forward(y)
            return y.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return super().forward(x)

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

class PeriodicConv3d(nn.Module):
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

class LatitudeGating(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = int(channels)
        mid_channels = max(self.channels // reduction, 4)
        self.fc1 = nn.Linear(self.channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, self.channels)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, L, H, W = x.shape
            y = x.mean(dim=-1)
            y = y.permute(0, 2, 3, 1)
            y = self.fc1(y)
            y = self.act(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            y = y.permute(0, 3, 1, 2).unsqueeze(-1)
        else:
            B, C, H, W = x.shape
            y = x.mean(dim=-1)
            y = y.permute(0, 2, 1)
            y = self.fc1(y)
            y = self.act(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y

class GradientOperator(nn.Module):
    def __init__(self, mode: str = "sobel"):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode == "sobel":
            self.register_buffer("kernel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
            self.register_buffer("kernel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        elif self.mode in ["learnable", "diffconv"]:
            self.conv_x = nn.Conv2d(1, 1, 3, padding=0, bias=False)
            self.conv_y = nn.Conv2d(1, 1, 3, padding=0, bias=False)
            nn.init.xavier_normal_(self.conv_x.weight)
            nn.init.xavier_normal_(self.conv_y.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x.to(x.device, x.dtype))
            grad_y = F.conv2d(x, self.kernel_y.to(x.device, x.dtype))
            return grad_x, grad_y
        if self.mode == "diffconv":
            w_x = self.conv_x.weight - self.conv_x.weight.mean(dim=(2, 3), keepdim=True)
            w_y = self.conv_y.weight - self.conv_y.weight.mean(dim=(2, 3), keepdim=True)
            grad_x = F.conv2d(x, w_x)
            grad_y = F.conv2d(x, w_y)
            return grad_x, grad_y
        return self.conv_x(x), self.conv_y(x)

class FlowFieldGenerator(nn.Module):
    def __init__(self, channels: int, height: int, groups: int = 1, diff_mode: str = "sobel", max_velocity: float = 5.0):
        super().__init__()
        ch = int(channels)
        self.groups = int(groups)
        self.max_velocity = float(max_velocity)
        mid = max(ch // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, mid, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid, 2 * self.groups, 3, padding=1),
        )
        self.grad_op = GradientOperator(mode=diff_mode)
        self.height = int(height)
        lat = torch.linspace(-math.pi / 2, math.pi / 2, self.height).view(1, 1, self.height, 1)
        metric_correction = 1.0 / (torch.cos(lat) + 1e-6)
        self.register_buffer("metric_correction", metric_correction)
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x_flat = x.reshape(B * L, C, H, W)
        field = self.conv(x_flat)
        
        if H != self.height:
             lat = torch.linspace(-math.pi / 2, math.pi / 2, H, device=x.device).view(1, 1, H, 1)
             metric_correction = 1.0 / (torch.cos(lat) + 1e-6)
        else:
             metric_correction = self.metric_correction

        BL = B * L
        G = self.groups
        field_folded = field.view(BL, G, 2, H, W).reshape(BL * G, 2, H, W)
        
        phi = field_folded[:, 0:1]
        psi = field_folded[:, 1:2]
        
        pad_h = (1, 1, 0, 0)
        pad_w = (0, 0, 1, 1)
        
        phi_pad = F.pad(phi, pad_h, mode="circular")
        phi_pad = F.pad(phi_pad, pad_w, mode="replicate")
        psi_pad = F.pad(psi, pad_h, mode="circular")
        psi_pad = F.pad(psi_pad, pad_w, mode="replicate")
        
        grad_x_phi, grad_y_phi = self.grad_op(phi_pad)
        grad_x_psi, grad_y_psi = self.grad_op(psi_pad)
        
        u = grad_x_phi * metric_correction - grad_y_psi
        v = grad_y_phi + grad_x_psi * metric_correction
        
        flows_flat = torch.cat([u, v], dim=1)
        flows_flat = self.max_velocity * torch.tanh(flows_flat)
        
        flows_stack = flows_flat.view(BL, G, 2, H, W).reshape(B, L, G, 2, H, W)
        return flows_stack

class LieAdvection(nn.Module):
    def __init__(self, rank: int, groups: int = 1, integration_mode: str = 'rk2'):
        super().__init__()
        self.rank = int(rank)
        self.groups = int(groups)
        self.integration_mode = integration_mode.lower()
        if self.rank % self.groups != 0:
            raise ValueError(f"Rank {self.rank} must be divisible by groups {self.groups}")
        self.ranks_per_group = self.rank // self.groups

    def _get_grid(self, flows: torch.Tensor, dt: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B_G = flows.shape[0]
        base_grid = _get_base_grid(H, W, flows.device, flows.dtype)
        base_grid = base_grid.expand(B_G, -1, -1, -1)

        if self.integration_mode == 'euler':
            flow_dt = flows * dt.view(B_G, 1, 1, 1)
            grid = base_grid - flow_dt.permute(0, 2, 3, 1)
            
        elif self.integration_mode == 'rk2':
            dt_view = dt.view(B_G, 1, 1, 1)
            k1 = flows
            grid_mid = base_grid - 0.5 * k1.permute(0, 2, 3, 1) * dt_view
            k2 = F.grid_sample(flows, grid_mid, mode='bilinear', padding_mode='border', align_corners=False)
            grid = base_grid - k2.permute(0, 2, 3, 1) * dt_view
            
        else:
            raise NotImplementedError
            
        return grid

    def forward(self, h_prev: torch.Tensor, flows: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B, C, H, W, R = h_prev.shape
        dt_scalar = dt.view(B, 1)
        
        h_reshaped = h_prev.view(B, C, H, W, self.groups, self.ranks_per_group)
        h_flat = h_reshaped.permute(0, 4, 1, 5, 2, 3).reshape(B * self.groups, C * self.ranks_per_group, H, W)
        flows_flat = flows.view(B * self.groups, 2, H, W)
        dt_flat = dt_scalar.repeat_interleave(self.groups, dim=0)

        use_triton_grid = (HAS_TRITON and h_prev.device.type == "cuda" and self.integration_mode == 'euler')
        
        if use_triton_grid:
            sampling_grid = torch.empty((B * self.groups, H, W, 2), device=h_prev.device, dtype=h_prev.dtype)
            dt_view = dt_scalar.view(B)
            grid = lambda META: (triton.cdiv(B * self.groups * H * W, META['BLOCK_SIZE']), )
            warp_grid_kernel[grid](
                flows_flat, dt_view, sampling_grid,
                B * self.groups, H, W,
                *flows_flat.stride(),
                dt_view.stride(0),
                *sampling_grid.stride(),
                self.groups,
            )
        else:
            sampling_grid = self._get_grid(flows_flat, dt_flat, H, W)
            
        h_warped_flat = F.grid_sample(h_flat, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)
        h_warped = h_warped_flat.view(B, self.groups, C, self.ranks_per_group, H, W).permute(0, 2, 4, 5, 1, 3).reshape(B, C, H, W, R)
        return h_warped

class SpectralMixer(nn.Module):
    def __init__(self, rank: int, w_freq: int):
        super().__init__()
        self.rank = int(rank)
        self.w_freq = int(w_freq)
        self.mix_linear = nn.Linear(self.rank * 2, self.rank * 2)
        nn.init.zeros_(self.mix_linear.weight)
        nn.init.zeros_(self.mix_linear.bias)
        with torch.no_grad():
            self.mix_linear.weight.view(self.rank * 2, self.rank * 2).diagonal().fill_(1.0)

    def forward(self, h_freq: torch.Tensor) -> torch.Tensor:
        B, C, H, Wf, R = h_freq.shape
        h_real = h_freq.real
        h_imag = h_freq.imag
        h_stacked = torch.cat([h_real, h_imag], dim=-1)
        h_mixed = self.mix_linear(h_stacked)
        h_real_out, h_imag_out = torch.chunk(h_mixed, 2, dim=-1)
        return torch.complex(h_real_out, h_imag_out)

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
        self.head = nn.Linear(ch, self.emb_ch * self.rank * 3)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.view(self.emb_ch, self.rank, 3)[:, :, 0].fill_(1.0)
            self.head.bias.view(self.emb_ch, self.rank, 3)[:, :, 1].fill_(0.0)
            self.head.bias.view(self.emb_ch, self.rank, 3)[:, :, 2].fill_(-5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        feat = self.conv(x)
        glob = self.global_pool(feat).flatten(1)
        glob = torch.sigmoid(self.global_fc(glob)).view(B, C, 1, 1)
        feat = feat * glob
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 3, 1).reshape(B, 1, self.w_freq, C)
        params = self.head(feat)
        params = params.view(B, 1, self.w_freq, self.emb_ch, self.rank, 3)
        params = params.permute(0, 1, 3, 2, 4, 5).contiguous()
        nu_rate = F.softplus(params[..., 0])
        theta_rate = torch.tanh(params[..., 1]) * math.pi
        sigma = torch.sigmoid(params[..., 2])
        return nu_rate, theta_rate, sigma

class SpectralDynamics(nn.Module):
    def __init__(self, channels: int, rank: int, w_freq: int, use_noise: bool = False, noise_scale: float = 1.0):
        super().__init__()
        self.estimator = DynamicsParameterEstimator(int(channels), int(channels), int(rank), int(w_freq))
        self.channels = int(channels)
        self.rank = int(rank)
        self.use_noise = bool(use_noise)
        self.noise_scale = float(noise_scale)
        self.mixer = SpectralMixer(rank, w_freq)

    def compute_params(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(B * L, C, H, W)
        nu_rate, theta_rate, sigma = self.estimator(x_flat)
        _, _, C_emb, W_freq, Rank = nu_rate.shape
        nu_rate = nu_rate.view(B, L, 1, C_emb, W_freq, Rank)
        theta_rate = theta_rate.view(B, L, 1, C_emb, W_freq, Rank)
        sigma = sigma.view(B, L, 1, C_emb, W_freq, Rank)
        return nu_rate, theta_rate, sigma

    def forward_step(self, h_trans: torch.Tensor, dt: torch.Tensor, params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, C, H, W, R = h_trans.shape
        h_freq = torch.fft.rfft2(h_trans.permute(0, 4, 1, 2, 3).float(), norm="ortho").permute(0, 2, 3, 4, 1).contiguous()
        W_freq = h_freq.shape[-2]
        h_freq = self.mixer(h_freq)
        h_freq_in = h_freq.permute(0, 4, 1, 2, 3).contiguous()
        nu_rate, theta_rate, sigma_val = params

        if HAS_TRITON and h_freq_in.device.type == "cuda":
            nu_in = nu_rate.permute(0, 4, 2, 1, 3) 
            theta_in = theta_rate.permute(0, 4, 2, 1, 3)
            nu_in = nu_in.expand(B, R, C, H, W_freq)
            theta_in = theta_in.expand(B, R, C, H, W_freq)
            h_real = h_freq_in.real.contiguous()
            h_imag = h_freq_in.imag.contiguous()
            out_real = torch.empty_like(h_real)
            out_imag = torch.empty_like(h_imag)
            dt_in = dt.view(B).contiguous()
            grid = lambda META: (triton.cdiv(B * R * C * H * W_freq, META['BLOCK_SIZE']), )
            koopman_kernel[grid](
                h_real, h_imag,
                nu_in, theta_in, dt_in,
                out_real, out_imag,
                B, R, C, H, W_freq,
                *h_real.stride(),
                *nu_in.stride(),
                dt_in.stride(0),
                *out_real.stride(),
            )
            h_evolved = torch.complex(out_real, out_imag)
            if self.use_noise:
                 sigma_in = sigma_val.permute(0, 4, 2, 1, 3).expand(B, R, C, H, W_freq)
                 dt_sqrt = torch.sqrt(dt.view(B, 1, 1, 1, 1).clamp_min(0.0))
                 amp = sigma_in * (self.noise_scale * dt_sqrt)
                 noise = torch.randn_like(h_evolved) * amp
                 h_evolved = h_evolved + noise
        else:
            dt_ = dt.view(B, -1)
            dt_expanded = dt_.view(B, 1, 1, 1, 1, 1)
            nu = nu_rate.permute(0, 4, 2, 1, 3) * dt_expanded.view(B, 1, 1, 1, 1)
            theta = theta_rate.permute(0, 4, 2, 1, 3) * dt_expanded.view(B, 1, 1, 1, 1)
            decay = torch.exp(-nu)
            rotate = torch.exp(1j * theta)
            lambda_k = decay * rotate
            h_evolved = h_freq_in * lambda_k
            if self.use_noise:
                sigma_b = sigma_val.permute(0, 4, 2, 1, 3)
                dt_sqrt = torch.sqrt(dt_.clamp_min(0.0)).view(B, 1, 1, 1, 1)
                amp = sigma_b * (self.noise_scale * dt_sqrt)
                noise = torch.randn_like(h_freq_in) * amp
                h_evolved = h_evolved + noise

        h_out = torch.fft.irfft2(h_evolved, s=(H, W), norm="ortho")
        return h_out.permute(0, 2, 3, 4, 1)

class LearnableStateMap(nn.Module):
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

class PhysicalRecurrentLayer(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], rank: int = 64, use_noise: bool = False, noise_scale: float = 1.0, dt_ref: float = 1.0, inj_k: float = 2.0, diff_mode: str = "sobel", static_ch: int = 0, learnable_init_state: bool = False, max_velocity: float = 5.0):
        super().__init__()
        self.emb_ch = int(emb_ch)
        H, W = int(input_shape[0]), int(input_shape[1])
        self.rank = int(rank)
        self.W_freq = W // 2 + 1
        self.use_noise = bool(use_noise)
        self.noise_scale = float(noise_scale)
        self.dt_ref = float(dt_ref)
        if self.dt_ref <= 0:
            self.dt_ref = 1.0
        self.inj_k = float(inj_k)
        self.inj_k = max(self.inj_k, 0.0)
        
        diff_mode = str(diff_mode)
        self.flow_groups = 4
        if self.rank % self.flow_groups != 0:
            self.flow_groups = 1
            
        self.hamiltonian = FlowFieldGenerator(self.emb_ch, height=H, groups=self.flow_groups, diff_mode=diff_mode, max_velocity=max_velocity)
        self.lie_transport = LieAdvection(rank=self.rank, groups=self.flow_groups, integration_mode='rk2')
        self.koopman = SpectralDynamics(self.emb_ch, self.rank, self.W_freq, use_noise=self.use_noise, noise_scale=self.noise_scale)
        self.proj_out = nn.Linear(self.rank, 1)
        if bool(learnable_init_state) and int(static_ch) > 0:
            self.init_state = LearnableStateMap(int(static_ch), self.emb_ch, self.rank, H, W)
        else:
            self.init_state = None
        self.post_ifft_proj = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.norm = SpatialGroupNorm(4, self.emb_ch)
        self.gate = LatitudeGating(self.emb_ch)
        self.rec_norm = nn.LayerNorm(self.rank)

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
                h0 = self.init_state(static_feats)
                curr_h = h0
            else:
                curr_h = torch.zeros(B, C, H, W, self.rank, device=x.device, dtype=x.dtype)
        else:
            curr_h = last_hidden_in
        if listT is None:
            dt_seq = torch.ones(B, L, device=x.device, dtype=x.dtype)
        else:
            dt_seq = _match_dt_seq(listT.to(x.device, x.dtype), L)
        
        x_perm = x.permute(0, 2, 1, 3, 4)
        flows_all = self.hamiltonian(x_perm)
        koopman_params_all = self.koopman.compute_params(x_perm)
        dt_ref = torch.tensor(self.dt_ref, device=x.device, dtype=x.dtype).clamp_min(1e-6)
        
        h_stack = torch.empty((B, C, L, H, W, self.rank), device=x.device, dtype=x.dtype)
        
        for t in range(L):
            x_t = x[:, :, t : t + 1]
            dt_t = dt_seq[:, t : t + 1]
            flow_t = flows_all[:, t]
            
            nu_t = koopman_params_all[0][:, t]
            theta_t = koopman_params_all[1][:, t]
            sigma_t = koopman_params_all[2][:, t]
            params_t = (nu_t, theta_t, sigma_t)
            
            h_trans = self.lie_transport(curr_h, flow_t, dt_t)
            h_next = self.koopman.forward_step(h_trans, dt_t, params_t)
            
            x_inject = x_t.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, self.rank)
            dt_scaled = (dt_t / dt_ref).clamp_min(0.0)
            if self.inj_k > 0:
                g = 1.0 - torch.exp(-dt_scaled * torch.tensor(self.inj_k, device=x.device, dtype=x.dtype))
            else:
                g = dt_scaled
            curr_h = h_next + g.view(B, 1, 1, 1, 1) * x_inject
            curr_h = self.rec_norm(curr_h)
            h_stack[:, :, t] = curr_h
            
        out = self.proj_out(h_stack).squeeze(-1)
        out = self.post_ifft_proj(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        out = self.norm(out)
        out_gated = self.gate(out)
        x_out = x + out_gated
        
        return x_out, curr_h

class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: Tuple[int, int], kernel_size: int = 7, cond_channels: Optional[int] = None, conv_type: str = "conv"):
        super().__init__()
        self.dw_conv = PeriodicConv3d(int(channels), int(channels), kernel_size=int(kernel_size), conv_type=str(conv_type))
        self.norm = SpatialGroupNorm(4, int(channels))
        self.cond_channels_spatial = int(cond_channels) if cond_channels is not None else 0
        self.cond_proj = nn.Conv3d(self.cond_channels_spatial, int(channels) * 2, kernel_size=1) if self.cond_channels_spatial > 0 else None
        self.pw_conv_in = nn.Linear(int(channels), int(channels) * 2)
        self.pw_conv_out = nn.Linear(int(channels), int(channels))

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
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.pw_conv_in(x)
        
        use_triton_glu = HAS_TRITON and x.device.type == "cuda" and x.is_contiguous()
        if use_triton_glu:
            x_out = torch.empty_like(x[..., :x.shape[-1]//2])
            n_elements_out = x_out.numel()
            C_out = x_out.shape[-1]
            grid = lambda META: (triton.cdiv(n_elements_out, META['BLOCK_SIZE']), )
            glu_kernel[grid](
                x, x_out,
                n_elements_out, C_out,
            )
            x = x_out
        else:
            x1, x2 = torch.chunk(x, 2, dim=4)
            x = x1 * torch.sigmoid(x2)
            
        x = self.pw_conv_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], ffn_ratio: float = 4.0, static_ch: int = 0, conv_type: str = "conv"):
        super().__init__()
        self.emb_ch = int(emb_ch)
        self.ffn_ratio = float(ffn_ratio)
        self.hidden_dim = int(self.emb_ch * self.ffn_ratio)
        self.hidden_size = (int(input_shape[0]), int(input_shape[1]))
        self.static_ch = int(static_ch)
        self.conv_type = str(conv_type)
        self.c_in = nn.Linear(self.emb_ch, self.hidden_dim)
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        self.block = GatedConvBlock(self.hidden_dim, self.hidden_size, kernel_size=7, cond_channels=cond_ch, conv_type=self.conv_type)
        self.c_out = nn.Linear(self.hidden_dim, self.emb_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = x.permute(0, 2, 3, 4, 1)
        x = self.c_in(x)
        x = self.act(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.block(x, cond=cond)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.c_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x

class UniPhyBlock(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], lru_args: Dict[str, Any], ffn_args: Dict[str, Any]):
        super().__init__()
        self.lru_layer = PhysicalRecurrentLayer(emb_ch, input_shape, **lru_args)
        self.feed_forward = FeedForwardNetwork(emb_ch, input_shape, **ffn_args)

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
        x_attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
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

class UniPhyBackbone(nn.Module):
    def __init__(self, emb_ch: int, input_shape: Tuple[int, int], num_layers: int, arch_mode: str, down_mode: str, lru_args: Dict[str, Any], ffn_args: Dict[str, Any]):
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
            self.uniphy_blocks = nn.ModuleList([UniPhyBlock(C, (H, W), lru_args, ffn_args) for _ in range(layers)])
            self.upsample = None
            self.fusion = None
            self.mid_attention = None
        else:
            curr_H, curr_W = H, W
            encoder_res: List[Tuple[int, int]] = []
            for i in range(layers):
                self.down_blocks.append(UniPhyBlock(C, (curr_H, curr_W), lru_args, ffn_args))
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
                self.up_blocks.append(UniPhyBlock(C, (h_up, w_up), lru_args, ffn_args))
                if self.arch_mode == "bifpn":
                    self.csa_blocks.append(BiFPNFusion(C))
                else:
                    self.csa_blocks.append(CrossScaleGate(C))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
            self.fusion = nn.Conv3d(C * 2, C, 1) if self.arch_mode == "unet" else nn.Identity()
            self.uniphy_blocks = None

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
            assert self.uniphy_blocks is not None
            for idx, blk in enumerate(self.uniphy_blocks):
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

class FeatureEmbedding(nn.Module):
    def __init__(self, input_ch: int, input_size: Tuple[int, int], emb_ch: int, static_ch: int = 0, hidden_factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.input_ch = int(input_ch)
        self.input_size = tuple(input_size)
        self.emb_ch = int(emb_ch)
        self.emb_hidden_ch = self.emb_ch
        self.static_ch = int(static_ch)
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])
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
        self.register_buffer("grid_embed", self._make_grid(self.input_size), persistent=False)
        self.hidden_size = (int(self.input_downsp_shape[1]), int(self.input_downsp_shape[2]))
        if self.static_ch > 0:
            self.static_embed = nn.Sequential(
                nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.rH + 2, self.rW + 2), stride=(self.rH, self.rW), padding=(1, 1)),
                nn.SiLU(),
            )
        else:
            self.static_embed = None
        cond_ch = self.emb_ch if self.static_ch > 0 else None
        self.c_hidden = nn.ModuleList([GatedConvBlock(self.emb_hidden_ch, self.hidden_size, kernel_size=7, cond_channels=cond_ch)])
        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.norm = SpatialGroupNorm(4, self.emb_ch)

    def _make_grid(self, input_size: Tuple[int, int]) -> torch.Tensor:
        H, W = tuple(input_size)
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

class ProbabilisticDecoder(nn.Module):
    def __init__(self, out_ch: int, emb_ch: int, dist_mode: str = "gaussian", hidden_factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.dist_mode = str(dist_mode).lower()
        self.output_ch = int(out_ch)
        self.emb_ch = int(emb_ch)
        self.rH, self.rW = int(hidden_factor[0]), int(hidden_factor[1])
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
        
        self.final_out_ch = self.output_ch * 2
        if self.dist_mode == "mdn":
            self.num_mixtures = 3
            self.final_out_ch = self.output_ch * 3 * self.num_mixtures
        
        self.c_out = nn.Conv3d(out_ch_after_up, self.final_out_ch, kernel_size=(1, 1, 1), padding="same")
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_shuffle_conv(x)
        x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        x = self.c_out(x)
        
        if self.dist_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
                sigma = F.softplus(log_sigma.float()).to(mu.dtype) + 1e-6
            return torch.cat([mu, sigma], dim=1)
        
        if self.dist_mode == "laplace":
            mu, log_b = torch.chunk(x, 2, dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                log_b = torch.clamp(log_b, min=-5.0, max=5.0)
                b = F.softplus(log_b.float()).to(mu.dtype) + 1e-6
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
        static_ch = int(getattr(args, "static_ch", 0))
        hidden_factor = getattr(args, "hidden_factor", (2, 2))
        
        self.embedding = FeatureEmbedding(input_ch, input_size, emb_ch, static_ch, hidden_factor)
        
        num_blocks = int(getattr(args, "convlru_num_blocks", 2))
        arch = getattr(args, "Arch", "unet")
        down_mode = getattr(args, "down_mode", "avg")
        
        rank = int(getattr(args, "lru_rank", 64))
        koopman_use_noise = bool(getattr(args, "koopman_use_noise", False))
        koopman_noise_scale = float(getattr(args, "koopman_noise_scale", 1.0))
        dt_ref = float(getattr(args, "dt_ref", getattr(args, "T", 1.0)))
        inj_k = float(getattr(args, "inj_k", 2.0))
        diff_mode = str(getattr(args, "diff_mode", "sobel"))
        learnable_init_state = bool(getattr(args, "learnable_init_state", False))
        max_velocity = float(getattr(args, "max_velocity", 5.0))
        
        lru_args = {
            "rank": rank,
            "use_noise": koopman_use_noise,
            "noise_scale": koopman_noise_scale,
            "dt_ref": dt_ref,
            "inj_k": inj_k,
            "diff_mode": diff_mode,
            "static_ch": static_ch,
            "learnable_init_state": learnable_init_state,
            "max_velocity": max_velocity
        }
        
        ffn_ratio = float(getattr(args, "ffn_ratio", 4.0))
        conv_type = str(getattr(args, "ConvType", "conv"))
        ffn_args = {
            "ffn_ratio": ffn_ratio,
            "static_ch": static_ch,
            "conv_type": conv_type
        }
        
        self.uniphy_model = UniPhyBackbone(
            emb_ch,
            self.embedding.input_downsp_shape[1:],
            num_blocks,
            arch,
            down_mode,
            lru_args,
            ffn_args
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
            x_hid, last_hidden_outs = self.uniphy_model(x_emb, listT=listT, cond=cond, static_feats=static_feats)
            out = self.decoder(x_hid, cond=cond)
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
        if listT is None:
            listT0 = torch.ones(B, x.size(1), device=x.device, dtype=x.dtype)
        else:
            listT0 = listT
        
        stats = revin_stats if revin_stats is not None else self.revin.stats(x)
        if stats.mean.dim() == 5 and stats.mean.size(1) > 1:
            stats = RevINStats(mean=stats.mean[:, -1:], stdev=stats.stdev[:, -1:])
            
        out_list: List[torch.Tensor] = []
        x_norm = self.revin(x, "norm", stats=stats)
        x_emb, _ = self.embedding(x_norm, static_feats=static_feats)
        x_hidden, last_hidden_outs = self.uniphy_model(x_emb, listT=listT0, cond=cond, static_feats=static_feats)
        x_dec0 = self.decoder(x_hidden, cond=cond)
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
            if x_step_dist.size(2) >= self.revin.num_features:
                 x_step_mean = x_step_dist[:, :, :self.revin.num_features]
            else:
                 x_step_mean = x_step_dist

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
            x_hidden, last_hidden_outs = self.uniphy_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt, cond=cond, static_feats=static_feats)
            x_dec = self.decoder(x_hidden, cond=cond)
            x_step_dist = x_dec.permute(0, 2, 1, 3, 4).contiguous()
            x_step_dist = x_step_dist[:, -1:, :, :, :]
            
            if dist_mode in ["gaussian", "laplace"]:
                out_ch = int(getattr(self.args, "out_ch", x_step_dist.size(2) // 2))
                mu = x_step_dist[:, :, :out_ch, :, :]
                scale = x_step_dist[:, :, out_ch:, :, :]
                if mu.size(2) == self.revin.num_features:
                    mu = self.revin(mu, "denorm", stats=stats)
                    scale_denorm = scale * stats.stdev
                else:
                    mu_denorm = mu
                    scale_denorm = scale
                out_list.append(torch.cat([mu_denorm, scale_denorm], dim=2))
                x_step_mean = mu_denorm
            else:
                out_list.append(x_step_dist)
                if x_step_dist.size(2) >= self.revin.num_features:
                     x_step_mean = x_step_dist[:, :, :self.revin.num_features]
                else:
                     x_step_mean = x_step_dist
                     
        return torch.cat(out_list, dim=1)

