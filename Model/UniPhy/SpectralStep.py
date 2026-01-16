import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def spectral_fused_kernel(
    x_ptr,
    nu_ptr,
    theta_ptr,
    dt_ptr,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_nu_b, stride_nu_r, stride_nu_h, stride_nu_w,
    stride_dt,
    B, C, H, W, RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = B * C * H * W
    mask = offs < total_elements
    
    w_idx = offs % W
    h_idx = (offs // W) % H
    c_idx = (offs // (W * H)) % C
    b_idx = offs // (W * H * C)
    
    dt = tl.load(dt_ptr + b_idx * stride_dt, mask=mask, other=0.0)
    
    filter_r = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    filter_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    nu_base = b_idx * stride_nu_b + h_idx * stride_nu_h + w_idx * stride_nu_w
    theta_base = b_idx * stride_nu_b + h_idx * stride_nu_h + w_idx * stride_nu_w
    
    for r in range(RANK):
        nu_off = nu_base + r * stride_nu_r
        theta_off = theta_base + r * stride_nu_r
        
        nu_val = tl.load(nu_ptr + nu_off, mask=mask, other=0.0)
        theta_val = tl.load(theta_ptr + theta_off, mask=mask, other=0.0)
        
        decay = tl.exp(-nu_val * dt)
        angle = theta_val * dt
        
        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)
        
        filter_r += decay * cos_a
        filter_i += decay * sin_a

    x_off = b_idx * stride_x_b + c_idx * stride_x_c + h_idx * stride_x_h + w_idx * stride_x_w
    
    x_r = tl.load(x_ptr + 2 * x_off, mask=mask, other=0.0)
    x_i = tl.load(x_ptr + 2 * x_off + 1, mask=mask, other=0.0)
    
    out_r = x_r * filter_r - x_i * filter_i
    out_i = x_r * filter_i + x_i * filter_r
    
    tl.store(x_ptr + 2 * x_off, out_r, mask=mask)
    tl.store(x_ptr + 2 * x_off + 1, out_i, mask=mask)

class SpectralStep(nn.Module):
    def __init__(self, in_ch, rank=32, w_freq=64):
        super().__init__()
        self.in_ch = in_ch
        self.rank = rank
        self.w_freq = w_freq
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, w_freq)),
            nn.Conv2d(in_ch, rank * 2, 1)
        )
        nn.init.uniform_(self.estimator[-1].weight, -0.01, 0.01)

    def forward(self, x, dt):
        B, C, H, W = x.shape
        x_spec = torch.fft.rfft2(x, norm="ortho")
        
        params = self.estimator(x)
        nu, theta = torch.chunk(params, 2, dim=1)
        nu = F.softplus(nu)
        theta = torch.tanh(theta) * math.pi
        
        target_spec = x_spec[:, :, :, :self.w_freq].contiguous()
        
        if dt.dim() == 0:
            dt_tensor = dt.view(1).expand(B)
        elif dt.dim() > 1:
            dt_tensor = dt.view(B)
        else:
            dt_tensor = dt
            
        dt_tensor = dt_tensor.contiguous()
        
        total_elements = B * C * H * self.w_freq
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        
        target_spec_view = torch.view_as_real(target_spec)

        spectral_fused_kernel[grid](
            target_spec_view,
            nu,
            theta,
            dt_tensor,
            target_spec.stride(0), target_spec.stride(1), target_spec.stride(2), target_spec.stride(3),
            nu.stride(0), nu.stride(1), nu.stride(2), nu.stride(3),
            dt_tensor.stride(0),
            B, C, H, self.w_freq, self.rank,
            BLOCK_SIZE=512
        )
        
        x_spec[:, :, :, :self.w_freq] = target_spec
        
        return torch.fft.irfft2(x_spec, s=(H, W), norm="ortho")

