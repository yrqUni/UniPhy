import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumSpectralPropagator(nn.Module):
    def __init__(self, hidden_dim, h, w):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_h = h
        self.freq_w = w 
        
        self.hamiltonian_real = nn.Parameter(torch.randn(hidden_dim, self.freq_h, self.freq_w) * 0.01)
        self.hamiltonian_imag = nn.Parameter(torch.randn(hidden_dim, self.freq_h, self.freq_w) * 0.01)

    def forward(self, z, dt):
        B, C, H, W = z.shape
        z_spec = torch.fft.fft2(z, norm='ortho')
        
        H_op = torch.complex(self.hamiltonian_real, self.hamiltonian_imag)
        
        propagator = torch.exp(1j * H_op * dt)
        
        z_shifted_spec = z_spec * propagator
        z_shifted = torch.fft.ifft2(z_shifted_spec, s=(H, W), norm='ortho')
        return z_shifted

class StreamFunctionAdvector(nn.Module):
    def __init__(self, in_ch, h, w):
        super().__init__()
        self.h = h
        self.w = w
        
        self.psi_net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        self.register_buffer('grid_base', torch.stack((xx, yy), dim=-1))

    def compute_curl(self, psi):
        psi_pad = F.pad(psi, (1, 1, 1, 1), mode='replicate')
        
        d_psi_dy = (psi_pad[:, :, 2:, 1:-1] - psi_pad[:, :, :-2, 1:-1]) / 2.0
        d_psi_dx = (psi_pad[:, :, 1:-1, 2:] - psi_pad[:, :, 1:-1, :-2]) / 2.0
        
        u = d_psi_dy
        v = -d_psi_dx
        
        return torch.cat([u, v], dim=1)

    def forward(self, z, dt):
        B, C, H, W = z.shape
        
        psi = self.psi_net(z.real) * dt
        
        velocity = self.compute_curl(psi) 
        
        flow_norm = torch.cat([
            velocity[:, 0:1] / (W/2), 
            velocity[:, 1:2] / (H/2)
        ], dim=1).permute(0, 2, 3, 1)
        
        grid = self.grid_base.unsqueeze(0).expand(B, -1, -1, -1)
        sampling_grid = grid - flow_norm
        
        z_real = F.grid_sample(z.real, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        z_imag = F.grid_sample(z.imag, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        
        return torch.complex(z_real, z_imag)

class ConservativeSymplecticNet(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim=16, h=64, w=128):
        super().__init__()
        
        self.encoder_real = nn.Conv2d(in_ch, hidden_dim, 1)
        self.encoder_imag = nn.Conv2d(in_ch, hidden_dim, 1)
        
        self.kinetic_op = QuantumSpectralPropagator(hidden_dim, h, w)
        self.advection_op = StreamFunctionAdvector(hidden_dim, h, w)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x, dt=1.0):
        z_real = self.encoder_real(x)
        z_imag = self.encoder_imag(x)
        z = torch.complex(z_real, z_imag)
        
        z_half = self.kinetic_op(z, dt / 2.0)
        z_adv = self.advection_op(z_half, dt)
        z_final_latent = self.kinetic_op(z_adv, dt / 2.0)
        
        z_cat = torch.cat([z_final_latent.real, z_final_latent.imag], dim=1)
        out = self.decoder(z_cat)
        
        return x + out

