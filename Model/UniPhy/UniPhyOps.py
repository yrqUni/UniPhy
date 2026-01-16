import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from PScan import PScanTriton

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
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.main:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.main(x)

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
        self.register_buffer('ky', grid_ky)
        self.register_buffer('kx', grid_kx)
        self.register_buffer('k2', k2)

    def apply_constraints(self, xf):
        if self.channels >= 2:
            u_f = xf[..., 0, :, :]
            v_f = xf[..., 1, :, :]
            div_f = (self.ky * u_f + self.kx * v_f) / self.k2
            xf_new = xf.clone()
            xf_new[..., 0, :, :] = u_f - self.ky * div_f
            xf_new[..., 1, :, :] = v_f - self.kx * div_f
            return xf_new
        return xf

    def get_interaction_force(self, x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            h_pot = self.interaction(x).sum()
            force = torch.autograd.grad(h_pot, x, create_graph=True, retain_graph=True)[0]
        return force

    def forward(self, x_seq, dt):
        B, T, C, H, W = x_seq.shape
        Wf = W // 2 + 1
        A = self.propagator(dt)
        A_scan = A.unsqueeze(1).expand(B, T, H, Wf, C)
        A_scan = A_scan.permute(0, 2, 3, 1, 4).reshape(-1, T, C)
        
        x_in = x_seq.reshape(B * T, C, H, W)
        v_inter = self.get_interaction_force(x_in)
        v_inter = v_inter.view(B, T, C, H, W)
        
        X_total = x_seq + v_inter * dt.view(B, 1, 1, 1, 1)
        Xf = torch.fft.rfft2(X_total, dim=(-2, -1), norm="ortho")
        Xf = self.apply_constraints(Xf)
        
        Xf_scan = Xf.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
        h_f_scan = self.pscan(A_scan, Xf_scan)
        
        h_f = h_f_scan.view(B, H, Wf, T, C).permute(0, 3, 4, 1, 2)
        h_f = self.apply_constraints(h_f)
        
        norm_in = torch.linalg.vector_norm(Xf, ord=2, dim=(2,3,4), keepdim=True)
        norm_out = torch.linalg.vector_norm(h_f, ord=2, dim=(2,3,4), keepdim=True)
        scaling = norm_in / (norm_out + 1e-6)
        h_f = h_f * scaling
        
        out = torch.fft.irfft2(h_f, s=(H, W), dim=(-2, -1), norm="ortho")
        return out

