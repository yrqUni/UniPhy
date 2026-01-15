import torch
import triton
import triton.language as tl

@triton.jit
def fused_hamiltonian_kernel(
    z_real_ptr, z_imag_ptr,
    h_real_ptr, h_imag_ptr,
    noise_real_ptr, noise_imag_ptr,
    dt_ptr,
    n_elements,
    sample_size,
    sigma,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    batch_idx = offsets // sample_size
    
    dt = tl.load(dt_ptr + batch_idx, mask=mask, other=0.0)

    z_r = tl.load(z_real_ptr + offsets, mask=mask)
    z_i = tl.load(z_imag_ptr + offsets, mask=mask)
    h_r = tl.load(h_real_ptr + offsets, mask=mask)
    h_i = tl.load(h_imag_ptr + offsets, mask=mask)
    
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
            out_real, out_imag,
            h_real, h_imag,
            noise_real, noise_imag,
            dt_flat,
            n_elements, 
            sample_size,
            float(sigma_val),
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
        
        sqrt_dt = torch.sqrt(dt_expanded)
        grad_sigma = torch.sum(grad_out_real * (sqrt_dt * n_r) + grad_out_imag * (sqrt_dt * n_i))
        
        return grad_z.real, grad_z.imag, grad_H.real, grad_H.imag, None, grad_sigma

@triton.jit
def stencil_curl_kernel(
    psi_ptr,
    u_ptr, v_ptr,
    stride_b, stride_c, stride_h, stride_w,
    H, W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    mask_h = off_h < H
    mask_w = off_w < W
    
    idx_up = tl.maximum(off_h - 1, 0)
    idx_down = tl.minimum(off_h + 1, H - 1)
    idx_left = tl.maximum(off_w - 1, 0)
    idx_right = tl.minimum(off_w + 1, W - 1)
    
    base_ptr = psi_ptr
    
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
    tl.store(u_ptr + offset_out, grad_y, mask=mask_h[:, None] & mask_w[None, :])
    tl.store(v_ptr + offset_out, -grad_x, mask=mask_h[:, None] & mask_w[None, :])

def fused_curl_2d(psi):
    psi = psi.contiguous()
    B, C, H, W = psi.shape
    u = torch.empty_like(psi)
    v = torch.empty_like(psi)
    
    total_slices = B * C
    psi_flat = psi.view(total_slices, H, W)
    u_flat = u.view(total_slices, H, W)
    v_flat = v.view(total_slices, H, W)
    
    grid = lambda meta: (triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(W, meta['BLOCK_W']))
    
    for i in range(total_slices):
        stencil_curl_kernel[grid](
            psi_flat[i], u_flat[i], v_flat[i],
            0, 0, psi_flat.stride(1), psi_flat.stride(2),
            H, W,
            BLOCK_H=16, BLOCK_W=16
        )
    return u, v

