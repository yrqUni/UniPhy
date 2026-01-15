import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_hamiltonian_kernel(
    z_real_ptr, z_imag_ptr, h_real_ptr, h_imag_ptr,
    noise_real_ptr, noise_imag_ptr, dt_ptr,
    n_elements, sample_size, sigma,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    batch_idx = offsets // sample_size
    
    idx_in_sample = offsets % sample_size
    
    dt = tl.load(dt_ptr + batch_idx, mask=mask, other=0.0)
    
    z_r = tl.load(z_real_ptr + offsets, mask=mask)
    z_i = tl.load(z_imag_ptr + offsets, mask=mask)
    
    h_r = tl.load(h_real_ptr + idx_in_sample, mask=mask)
    h_i = tl.load(h_imag_ptr + idx_in_sample, mask=mask)
    
    real_arg = -h_i * dt
    imag_arg = h_r * dt
    mag = tl.exp(real_arg)
    cos_v = tl.cos(imag_arg)
    sin_v = tl.sin(imag_arg)
    
    prop_r = mag * cos_v
    prop_i = mag * sin_v
    
    out_r = z_r * prop_r - z_i * prop_i
    out_i = z_r * prop_i + z_i * prop_r
    
    noise_r = tl.load(noise_real_ptr + offsets, mask=mask)
    noise_i = tl.load(noise_imag_ptr + offsets, mask=mask)
    
    scale = sigma * tl.sqrt(dt)
    out_r = out_r + scale * noise_r
    out_i = out_i + scale * noise_i
    
    tl.store(z_real_ptr + offsets, out_r, mask=mask)
    tl.store(z_imag_ptr + offsets, out_i, mask=mask)

class FusedHamiltonian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_real, z_imag, h_real, h_imag, dt, sigma_tensor):
        sigma_val = sigma_tensor.item()
        noise_real = torch.randn_like(z_real)
        noise_imag = torch.randn_like(z_imag)
        ctx.save_for_backward(z_real, z_imag, h_real, h_imag, noise_real, noise_imag, dt)
        ctx.sigma_val = sigma_val
        n_elements = z_real.numel()
        B = z_real.shape[0]
        sample_size = n_elements // B
        
        out_real = z_real.clone()
        out_imag = z_imag.clone()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        dt_flat = dt.view(-1).contiguous()
        
        fused_hamiltonian_kernel[grid](
            out_real, out_imag, h_real, h_imag,
            noise_real, noise_imag, dt_flat,
            n_elements, sample_size, float(sigma_val),
            BLOCK_SIZE=1024
        )
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        z_r, z_i, h_r, h_i, n_r, n_i, dt = ctx.saved_tensors
        dt_expanded = dt.view(-1, 1, 1, 1)
        H = torch.complex(h_r, h_i)
        prop = torch.exp(1j * H * dt_expanded)
        
        grad_out = torch.complex(grad_out_real, grad_out_imag)
        grad_z = grad_out * torch.conj(prop)
        
        z_in = torch.complex(z_r, z_i)
        z_out_det = z_in * prop
        grad_H = grad_out * torch.conj(z_out_det) * 1j * dt_expanded
        
        if grad_H.shape[0] > 1:
            grad_H = grad_H.sum(dim=0)
            
        sqrt_dt = torch.sqrt(dt_expanded)
        grad_sigma = torch.sum(grad_out_real * (sqrt_dt * n_r) + grad_out_imag * (sqrt_dt * n_i))
        
        return grad_z.real, grad_z.imag, grad_H.real, grad_H.imag, None, grad_sigma

@triton.jit
def stencil_curl_kernel(
    psi_ptr, u_ptr, v_ptr,
    stride_slice, stride_h, stride_w,
    H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_slice = tl.program_id(2)

    offset_slice = pid_slice * stride_slice
    
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_h = off_h < H
    mask_w = off_w < W
    
    idx_up = tl.maximum(off_h - 1, 0)
    idx_down = tl.minimum(off_h + 1, H - 1)
    idx_left = tl.maximum(off_w - 1, 0)
    idx_right = tl.minimum(off_w + 1, W - 1)
    
    base_ptr = psi_ptr + offset_slice
    ptr_up = base_ptr + (idx_up[:, None] * stride_h + off_w[None, :] * stride_w)
    ptr_down = base_ptr + (idx_down[:, None] * stride_h + off_w[None, :] * stride_w)
    ptr_left = base_ptr + (off_h[:, None] * stride_h + idx_left[None, :] * stride_w)
    ptr_right = base_ptr + (off_h[:, None] * stride_h + idx_right[None, :] * stride_w)
    
    val_up = tl.load(ptr_up, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    val_down = tl.load(ptr_down, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    val_left = tl.load(ptr_left, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    val_right = tl.load(ptr_right, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    
    grad_y = (val_down - val_up) * 0.5
    grad_x = (val_right - val_left) * 0.5
    
    offset_out = off_h[:, None] * stride_h + off_w[None, :] * stride_w
    
    u_out_ptr = u_ptr + offset_slice
    v_out_ptr = v_ptr + offset_slice
    
    tl.store(u_out_ptr + offset_out, grad_y, mask=mask_h[:, None] & mask_w[None, :])
    tl.store(v_out_ptr + offset_out, -grad_x, mask=mask_h[:, None] & mask_w[None, :])

def fused_curl_2d(psi):
    psi = psi.contiguous()
    B, C, H, W = psi.shape
    u = torch.empty_like(psi)
    v = torch.empty_like(psi)
    
    total_slices = B * C
    psi_flat = psi.view(total_slices, H, W)
    u_flat = u.view(total_slices, H, W)
    v_flat = v.view(total_slices, H, W)
    
    grid = lambda meta: (triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(W, meta['BLOCK_W']), total_slices)
    
    stride_slice = psi_flat.stride(0)
    stride_h = psi_flat.stride(1)
    stride_w = psi_flat.stride(2)
    
    stencil_curl_kernel[grid](
        psi_flat, u_flat, v_flat,
        stride_slice, stride_h, stride_w,
        H, W, BLOCK_H=16, BLOCK_W=16
    )
    return u, v

class CliffordConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dim = dim // 4
        if self.dim * 4 != dim:
            raise ValueError("Clifford channels must be divisible by 4")
        self.conv_s = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_x = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_y = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_b = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)

    def forward(self, x):
        s, vx, vy, b = torch.chunk(x, 4, dim=1)
        out_s = self.conv_s(s) - self.conv_x(vx) - self.conv_y(vy) - self.conv_b(b)
        out_x = self.conv_x(s) + self.conv_s(vx) - self.conv_b(vy) + self.conv_y(b)
        out_y = self.conv_y(s) + self.conv_b(vx) + self.conv_s(vy) - self.conv_x(b)
        out_b = self.conv_b(s) + self.conv_y(vx) - self.conv_x(vy) + self.conv_s(b)
        return torch.cat([out_s, out_x, out_y, out_b], dim=1)

class StreamFunctionMixing(nn.Module):
    def __init__(self, in_ch, h, w):
        super().__init__()
        self.psi_net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, 1, 3, padding=1)
        )
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        self.register_buffer('grid_base', torch.stack((xx, yy), dim=-1))

    def forward(self, z, dt):
        B, C, H, W = z.shape
        dt_view = dt.view(B, 1, 1, 1)
        psi = self.psi_net(z) * dt_view
        if not self.training and z.is_cuda:
            u, v = fused_curl_2d(psi)
        else:
            u, v = self._curl_pytorch(psi)
        flow_norm = torch.cat([u / (W/2), v / (H/2)], dim=1).permute(0, 2, 3, 1)
        grid = self.grid_base.unsqueeze(0).expand(B, -1, -1, -1)
        sampling_grid = grid - flow_norm
        z_out = F.grid_sample(z, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        return z_out

    def _curl_pytorch(self, psi):
        psi_pad = F.pad(psi, (1, 1, 1, 1), mode='replicate')
        u = (psi_pad[:, :, 2:, 1:-1] - psi_pad[:, :, :-2, 1:-1]) / 2.0
        v = -(psi_pad[:, :, 1:-1, 2:] - psi_pad[:, :, 1:-1, :-2]) / 2.0
        return u, v

class AdvectionStep(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_ch, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, dt):
        B, C, H, W = x.shape
        dt_view = dt.view(B, 1, 1, 1)
        flow = torch.tanh(self.net(x)) * dt_view
        
        y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(y, x_coord, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        flow_perm = flow.permute(0, 2, 3, 1)
        
        grid = base_grid - flow_perm
        
        x_adv = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return x_adv

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
        theta = (torch.tanh(theta) * math.pi)
        
        dt_view = dt.view(B, 1, 1, 1)
        decay = torch.exp(-nu * dt_view)
        angle = theta * dt_view
        
        operator = torch.complex(decay * torch.cos(angle), decay * torch.sin(angle))
        
        feat_spec = x_spec[:, :, :, :self.w_freq]
        
        feat_spec = feat_spec.unsqueeze(1) * operator.unsqueeze(2)
        
        feat_spec = feat_spec.sum(dim=1)
        
        x_out_spec = torch.zeros_like(x_spec)
        x_out_spec[:, :, :, :self.w_freq] = feat_spec
        return torch.fft.irfft2(x_out_spec, s=(H, W), norm="ortho")

