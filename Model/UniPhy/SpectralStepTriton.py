import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def spectral_fwd_kernel(
    out_r_ptr, out_i_ptr,
    nu_ptr, theta_ptr,
    dt_ptr,
    stride_out_b, stride_out_h, stride_out_w,
    stride_nu_b, stride_nu_r, stride_nu_h, stride_nu_w,
    stride_dt,
    RANK: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_HW = tl.program_id(1)
    
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    idx_h = pid_HW // num_w_blocks
    idx_w_chunk = pid_HW % num_w_blocks
    
    w_start = idx_w_chunk * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask = offs_w < W
    
    dt = tl.load(dt_ptr + pid_b * stride_dt)
    
    acc_r = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_W], dtype=tl.float32)
    
    nu_base = pid_b * stride_nu_b + idx_h * stride_nu_h
    theta_base = pid_b * stride_nu_b + idx_h * stride_nu_h
    
    for r in range(RANK):
        p_nu = nu_ptr + nu_base + r * stride_nu_r + offs_w * stride_nu_w
        p_theta = theta_ptr + theta_base + r * stride_nu_r + offs_w * stride_nu_w
        
        nu_val = tl.load(p_nu, mask=mask, other=0.0)
        theta_val = tl.load(p_theta, mask=mask, other=0.0)
        
        decay = tl.exp(-nu_val * dt)
        angle = theta_val * dt
        
        acc_r += decay * tl.cos(angle)
        acc_i += decay * tl.sin(angle)

    out_off = pid_b * stride_out_b + idx_h * stride_out_h + offs_w * stride_out_w
    tl.store(out_r_ptr + out_off, acc_r, mask=mask)
    tl.store(out_i_ptr + out_off, acc_i, mask=mask)

@triton.jit
def spectral_bwd_kernel(
    dnu_ptr, dtheta_ptr,
    grad_r_ptr, grad_i_ptr,
    nu_ptr, theta_ptr,
    dt_ptr,
    stride_nu_b, stride_nu_r, stride_nu_h, stride_nu_w,
    stride_grad_b, stride_grad_h, stride_grad_w,
    stride_dt,
    RANK: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_HW = tl.program_id(1)
    
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    idx_h = pid_HW // num_w_blocks
    idx_w_chunk = pid_HW % num_w_blocks
    
    w_start = idx_w_chunk * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask = offs_w < W
    
    dt = tl.load(dt_ptr + pid_b * stride_dt)
    
    grad_off = pid_b * stride_grad_b + idx_h * stride_grad_h + offs_w * stride_grad_w
    g_r = tl.load(grad_r_ptr + grad_off, mask=mask, other=0.0)
    g_i = tl.load(grad_i_ptr + grad_off, mask=mask, other=0.0)
    
    base_idx = pid_b * stride_nu_b + idx_h * stride_nu_h
    
    for r in range(RANK):
        curr_offset = base_idx + r * stride_nu_r + offs_w * stride_nu_w
        
        nu_val = tl.load(nu_ptr + curr_offset, mask=mask, other=0.0)
        theta_val = tl.load(theta_ptr + curr_offset, mask=mask, other=0.0)
        
        decay = tl.exp(-nu_val * dt)
        angle = theta_val * dt
        
        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)
        
        term_r = decay * cos_a
        term_i = decay * sin_a
        
        d_term_real_dnu = -dt * term_r
        d_term_imag_dnu = -dt * term_i
        
        d_term_real_dtheta = -dt * term_i
        d_term_imag_dtheta = dt * term_r
        
        d_nu = g_r * d_term_real_dnu + g_i * d_term_imag_dnu
        d_theta = g_r * d_term_real_dtheta + g_i * d_term_imag_dtheta
        
        tl.store(dnu_ptr + curr_offset, d_nu, mask=mask)
        tl.store(dtheta_ptr + curr_offset, d_theta, mask=mask)

class SpectralFilterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu, theta, dt, w_freq):
        B, R, H, W_in = nu.shape
        ctx.w_freq = w_freq
        ctx.rank = R
        
        dt = dt.contiguous()
        nu = nu.contiguous()
        theta = theta.contiguous()
        
        filter_real = torch.empty((B, H, w_freq), device=nu.device, dtype=torch.float32)
        filter_imag = torch.empty((B, H, w_freq), device=nu.device, dtype=torch.float32)
        
        BLOCK_W = 128
        grid = (B, H * triton.cdiv(w_freq, BLOCK_W))
        
        spectral_fwd_kernel[grid](
            filter_real, filter_imag,
            nu, theta,
            dt,
            filter_real.stride(0), filter_real.stride(1), filter_real.stride(2),
            nu.stride(0), nu.stride(1), nu.stride(2), nu.stride(3),
            dt.stride(0),
            RANK=R, W=w_freq, BLOCK_W=BLOCK_W
        )
        
        ctx.save_for_backward(nu, theta, dt)
        return filter_real, filter_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        nu, theta, dt = ctx.saved_tensors
        w_freq = ctx.w_freq
        R = ctx.rank
        B, _, H, _ = nu.shape
        
        dnu = torch.empty_like(nu)
        dtheta = torch.empty_like(theta)
        
        BLOCK_W = 128
        grid = (B, H * triton.cdiv(w_freq, BLOCK_W))
        
        spectral_bwd_kernel[grid](
            dnu, dtheta,
            grad_real, grad_imag,
            nu, theta,
            dt,
            nu.stride(0), nu.stride(1), nu.stride(2), nu.stride(3),
            grad_real.stride(0), grad_real.stride(1), grad_real.stride(2),
            dt.stride(0),
            RANK=R, W=w_freq, BLOCK_W=BLOCK_W
        )
        
        return dnu, dtheta, None, None

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
        
        if dt.dim() == 0: dt = dt.view(1).expand(B)
        elif dt.dim() > 1: dt = dt.view(B)
        dt = dt.contiguous()
        
        filter_real, filter_imag = SpectralFilterFunction.apply(nu, theta, dt, self.w_freq)
        
        target_spec = x_spec[:, :, :, :self.w_freq]
        filter_complex = torch.complex(filter_real, filter_imag).unsqueeze(1)
        
        out_spec = target_spec * filter_complex
        
        x_final_spec = torch.zeros_like(x_spec)
        x_final_spec[:, :, :, :self.w_freq] = out_spec
        x_final_spec[:, :, :, self.w_freq:] = x_spec[:, :, :, self.w_freq:]
        
        return torch.fft.irfft2(x_final_spec, s=(H, W), norm="ortho")

