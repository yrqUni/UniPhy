import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def spectral_prep_kernel(
    zeta_ptr, inv_lap_k_ptr, kx_ptr, ky_ptr,
    u_ptr, v_ptr, gx_ptr, gy_ptr,
    H, Wc,
    stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = H * Wc
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements
    
    y_idx = offs // Wc
    x_idx = offs % Wc
    
    kx = tl.load(kx_ptr + x_idx, mask=mask, other=0.0)
    ky = tl.load(ky_ptr + y_idx, mask=mask, other=0.0)
    inv_lap = tl.load(inv_lap_k_ptr + offs, mask=mask, other=0.0)
    
    zeta_real_ptr = zeta_ptr + 2 * offs
    zeta_imag_ptr = zeta_ptr + 2 * offs + 1
    
    zr = tl.load(zeta_real_ptr, mask=mask, other=0.0)
    zi = tl.load(zeta_imag_ptr, mask=mask, other=0.0)
    
    pr = zr * inv_lap
    pi = zi * inv_lap
    
    ur = ky * pi
    ui = -ky * pr
    
    vr = -kx * pi
    vi = kx * pr
    
    gxr = -kx * zi
    gxi = kx * zr
    
    gyr = -ky * zi
    gyi = ky * zr
    
    base_out = 2 * offs
    
    tl.store(u_ptr + base_out, ur, mask=mask)
    tl.store(u_ptr + base_out + 1, ui, mask=mask)
    
    tl.store(v_ptr + base_out, vr, mask=mask)
    tl.store(v_ptr + base_out + 1, vi, mask=mask)
    
    tl.store(gx_ptr + base_out, gxr, mask=mask)
    tl.store(gx_ptr + base_out + 1, gxi, mask=mask)
    
    tl.store(gy_ptr + base_out, gyr, mask=mask)
    tl.store(gy_ptr + base_out + 1, gyi, mask=mask)

@triton.jit
def advection_kernel(
    u_ptr, v_ptr, gx_ptr, gy_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    u = tl.load(u_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    gx = tl.load(gx_ptr + offs, mask=mask)
    gy = tl.load(gy_ptr + offs, mask=mask)
    
    res = u * gx + v * gy
    tl.store(out_ptr + offs, res, mask=mask)

@triton.jit
def rk4_combine_kernel(
    z_ptr, k1_ptr, k2_ptr, k3_ptr, k4_ptr, out_ptr,
    dt,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    z = tl.load(z_ptr + offs, mask=mask)
    k1 = tl.load(k1_ptr + offs, mask=mask)
    k2 = tl.load(k2_ptr + offs, mask=mask)
    k3 = tl.load(k3_ptr + offs, mask=mask)
    k4 = tl.load(k4_ptr + offs, mask=mask)
    
    res = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    tl.store(out_ptr + offs, res, mask=mask)

class BarotropicVorticitySolver(nn.Module):
    def __init__(self, height: int, width: int, dt: float = 1.0, viscosity: float = 1e-3):
        super().__init__()
        self.H = int(height)
        self.W = int(width)
        self.dt = float(dt)
        self.viscosity = float(viscosity)
        
        ky = torch.fft.fftfreq(self.H).view(-1, 1) * 2 * math.pi
        kx = torch.fft.rfftfreq(self.W).view(1, -1) * 2 * math.pi
        
        self.register_buffer("ky", ky.to(torch.float32))
        self.register_buffer("kx", kx.to(torch.float32))
        
        self.register_buffer("laplacian_k", -(self.kx**2 + self.ky**2))
        
        inv_lap_k = self.laplacian_k.clone()
        inv_lap_k[0, 0] = 1.0 
        inv_lap_val = 1.0 / inv_lap_k
        inv_lap_val[0, 0] = 0.0
        self.register_buffer("inv_laplacian_k", inv_lap_val.to(torch.float32))
        
        mask_h = torch.ones_like(ky)
        mask_w = torch.ones_like(kx)
        mask_h[int(self.H/3):int(2*self.H/3)] = 0
        mask_w[:, int(self.W/3):] = 0
        self.register_buffer("dealias_mask", mask_h * mask_w)

    def compute_rhs(self, zeta_hat: torch.Tensor) -> torch.Tensor:
        psi_hat = -zeta_hat * self.inv_laplacian_k
        u_hat = -1j * self.ky * psi_hat
        v_hat = 1j * self.kx * psi_hat
        grad_x_hat = 1j * self.kx * zeta_hat
        grad_y_hat = 1j * self.ky * zeta_hat
        
        u = torch.fft.irfft2(u_hat, s=(self.H, self.W))
        v = torch.fft.irfft2(v_hat, s=(self.H, self.W))
        gx = torch.fft.irfft2(grad_x_hat, s=(self.H, self.W))
        gy = torch.fft.irfft2(grad_y_hat, s=(self.H, self.W))
        
        advection = torch.empty_like(u)
        n_elements = u.numel()
        advection_kernel[(triton.cdiv(n_elements, 1024),)](
            u, v, gx, gy, advection,
            n_elements, BLOCK_SIZE=1024
        )
        
        advection_hat = torch.fft.rfft2(advection) * self.dealias_mask
        diffusion_hat = self.viscosity * self.laplacian_k * zeta_hat
        
        return -advection_hat + diffusion_hat

    def rk4_step(self, zeta_hat: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.compute_rhs(zeta_hat)
        k2 = self.compute_rhs(zeta_hat + 0.5 * dt * k1)
        k3 = self.compute_rhs(zeta_hat + 0.5 * dt * k2)
        k4 = self.compute_rhs(zeta_hat + dt * k3)
        
        out = torch.empty_like(zeta_hat)
        
        z_view = zeta_hat.view(torch.float32)
        k1_view = k1.view(torch.float32)
        k2_view = k2.view(torch.float32)
        k3_view = k3.view(torch.float32)
        k4_view = k4.view(torch.float32)
        out_view = out.view(torch.float32)
        
        n_elements = z_view.numel()
        
        rk4_combine_kernel[(triton.cdiv(n_elements, 1024),)](
            z_view, k1_view, k2_view, k3_view, k4_view, out_view,
            dt,
            n_elements, BLOCK_SIZE=1024
        )
        return out

    def forward(self, x_phys: torch.Tensor, steps: int = 1) -> torch.Tensor:
        zeta_hat = torch.fft.rfft2(x_phys)
        for _ in range(steps):
            zeta_hat = self.rk4_step(zeta_hat, self.dt)
        return torch.fft.irfft2(zeta_hat, s=(self.H, self.W))

