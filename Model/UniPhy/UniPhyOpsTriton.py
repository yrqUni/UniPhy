import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import triton
import triton.language as tl
from PScan import PScanTriton

@triton.jit
def _apply_constraints_kernel(
    xf_ptr, ky_ptr, kx_ptr, k2_ptr,
    stride_b, stride_t, stride_c, stride_h, stride_w,
    B, T, H, Wf,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx_w = pid % Wf
    idx_h = (pid // Wf) % H
    idx_t = (pid // (Wf * H)) % T
    idx_b = pid // (Wf * H * T)
    base_off = idx_b * stride_b + idx_t * stride_t + idx_h * stride_h + idx_w * stride_w
    u_off = base_off + 0 * stride_c
    v_off = base_off + 1 * stride_c
    u_re = tl.load(xf_ptr + u_off * 2)
    u_im = tl.load(xf_ptr + u_off * 2 + 1)
    v_re = tl.load(xf_ptr + v_off * 2)
    v_im = tl.load(xf_ptr + v_off * 2 + 1)
    ky = tl.load(ky_ptr + idx_h)
    kx = tl.load(kx_ptr + idx_w)
    k2 = tl.load(k2_ptr + idx_h * Wf + idx_w)
    div_re = (ky * u_re + kx * v_re) / k2
    div_im = (ky * u_im + kx * v_im) / k2
    tl.store(xf_ptr + u_off * 2, u_re - ky * div_re)
    tl.store(xf_ptr + u_off * 2 + 1, u_im - ky * div_im)
    tl.store(xf_ptr + v_off * 2, v_re - kx * div_re)
    tl.store(xf_ptr + v_off * 2 + 1, v_im - kx * div_im)

def triton_apply_constraints(xf, ky, kx, k2):
    B, T, C, H, Wf = xf.shape
    if C < 2: return xf
    xf_real = torch.view_as_real(xf)
    grid = (B * T * H * Wf,)
    _apply_constraints_kernel[grid](
        xf_real, ky, kx, k2,
        xf.stride(0), xf.stride(1), xf.stride(2), xf.stride(3), xf.stride(4),
        B, T, H, Wf,
        BLOCK_SIZE=1
    )
    return xf

class LearnedPropagator(nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, channels)
        )
        ky = torch.fft.fftfreq(h)
        kx = torch.fft.rfftfreq(w)
        grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')
        self.register_buffer('k_grid', torch.stack([grid_ky, grid_kx], dim=-1))
        k2_val = torch.sqrt(grid_ky**2 + grid_kx**2)
        self.register_buffer('k_phys', k2_val.unsqueeze(-1))
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, dt):
        h, wf, _ = self.k_grid.shape
        learned_phase = self.net(self.k_grid.view(-1, 2)).view(h, wf, self.channels)
        omega = learned_phase + self.k_phys.to(learned_phase.dtype)
        phase = omega * dt.view(-1, 1, 1, 1)
        return torch.complex(torch.cos(phase), torch.sin(phase))

class FluidInteractionNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(channels * 2, channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return self.conv3(x)

    def get_force_analytical(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.forward(x)
            force = torch.autograd.grad(y.sum(), x)[0]
        return force

class UniPhyFluidScan(nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        self.h, self.w, self.channels = h, w, channels
        self.pscan = PScanTriton.apply
        self.propagator = LearnedPropagator(channels, h, w)
        self.interaction = FluidInteractionNet(channels)
        ky = torch.fft.fftfreq(h)
        kx = torch.fft.rfftfreq(w)
        grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')
        k2 = grid_ky**2 + grid_kx**2
        k2[0, 0] = 1.0
        self.register_buffer('ky_b', ky)
        self.register_buffer('kx_b', kx)
        self.register_buffer('k2_b', k2)

    def forward(self, x_seq, dt):
        B, T, C, H, W = x_seq.shape
        Wf = W // 2 + 1
        A = self.propagator(dt)
        A_scan = A.unsqueeze(1).expand(B, T, H, Wf, C).permute(0, 2, 3, 1, 4).reshape(-1, T, C)
        
        v_inter = self.interaction.get_force_analytical(x_seq.reshape(B * T, C, H, W)).view(B, T, C, H, W)
        Xf = torch.fft.rfft2(x_seq + v_inter * dt.view(B, 1, 1, 1, 1), dim=(-2, -1), norm="ortho")
        Xf = triton_apply_constraints(Xf, self.ky_b, self.kx_b, self.k2_b)
        
        Xf_scan = Xf.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
        h_f_scan = self.pscan(A_scan, Xf_scan)
        
        h_f = h_f_scan.view(B, H, Wf, T, C).permute(0, 3, 4, 1, 2)
        h_f = triton_apply_constraints(h_f, self.ky_b, self.kx_b, self.k2_b)
        
        n_in = torch.linalg.vector_norm(Xf, ord=2, dim=(2,3,4), keepdim=True)
        n_out = torch.linalg.vector_norm(h_f, ord=2, dim=(2,3,4), keepdim=True)
        h_f = h_f * (n_in / (n_out + 1e-6))
        
        return torch.fft.irfft2(h_f, s=(H, W), dim=(-2, -1), norm="ortho")

