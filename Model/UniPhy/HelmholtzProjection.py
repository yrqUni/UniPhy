import torch
import torch.nn as nn
import torch.fft

class HelmholtzProjection(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.h = h
        self.w = w
        kx = torch.fft.fftfreq(w)
        ky = torch.fft.fftfreq(h)
        grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')
        
        k2 = grid_kx**2 + grid_ky**2
        k2[0, 0] = 1.0
        self.register_buffer('inv_laplacian', 1.0 / k2)
        self.register_buffer('kx', grid_kx)
        self.register_buffer('ky', grid_ky)

    def forward(self, v_star):
        B, C, H, W = v_star.shape
        if C < 2:
            return v_star
            
        u = v_star[:, 0:1, :, :]
        v = v_star[:, 1:2, :, :]
        
        u_f = torch.fft.fftn(u, dim=(-2, -1))
        v_f = torch.fft.fftn(v, dim=(-2, -1))
        
        div_f = 1j * self.kx * u_f + 1j * self.ky * v_f
        p_f = div_f * self.inv_laplacian
        
        grad_px_f = 1j * self.kx * p_f
        grad_py_f = 1j * self.ky * p_f
        
        u_proj = u - torch.fft.ifftn(grad_px_f, dim=(-2, -1)).real
        v_proj = v - torch.fft.ifftn(grad_py_f, dim=(-2, -1)).real
        
        v_final = v_star.clone()
        v_final[:, 0:1, :, :] = u_proj
        v_final[:, 1:2, :, :] = v_proj
        
        return v_final

