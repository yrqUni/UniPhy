import torch
import torch.nn as nn
import torch.nn.functional as F

class HamiltonianPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, conserve_energy=True):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.conserve_energy = conserve_energy
        
        assert dim % 2 == 0
        
        q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.cfloat))
        self.V_real = nn.Parameter(q.real)
        self.V_imag = nn.Parameter(q.imag)
        
        half_dim = dim // 2
        self.log_freq = nn.Parameter(torch.randn(half_dim))
        self.log_decay = nn.Parameter(torch.linspace(-5, -1, half_dim))

    def get_operators(self, dt):
        V = torch.complex(self.V_real, self.V_imag)
        V = F.normalize(V, dim=0)
        V_inv = V.conj().T
        
        freqs = torch.exp(self.log_freq).repeat_interleave(2)
        
        if self.conserve_energy:
            decay = torch.zeros_like(freqs)
        else:
            decay = -torch.exp(self.log_decay).repeat_interleave(2)
            
        lambda_modes = torch.complex(decay, -freqs)
        
        mask = torch.ones_like(lambda_modes)
        mask[0] = 0
        mask[1] = 0
        lambda_modes = lambda_modes * mask
        
        dt_expanded = dt.unsqueeze(-1)
        evolution_diag = torch.exp(lambda_modes.unsqueeze(0).unsqueeze(0) * dt_expanded)
        
        return V, V_inv, evolution_diag

    def forward(self, z, dt):
        V, V_inv, evo_diag = self.get_operators(dt)
        
        z_eigen = torch.matmul(z, V_inv.T)
        z_eigen_next = z_eigen * evo_diag
        z_out = torch.matmul(z_eigen_next, V.T)
        
        return z_out

