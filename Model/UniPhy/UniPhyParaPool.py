import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxConservingSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

        nn.init.zeros_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x):
        x_in = x.permute(0, 2, 3, 1)
        
        x1 = self.w1(x_in)
        x2 = self.w2(x_in)
        hidden = F.silu(x1) * x2
        
        delta = self.w3(hidden)
        delta = delta.permute(0, 3, 1, 2)
        
        delta_mean = delta.mean(dim=(2, 3), keepdim=True)
        delta = delta - delta_mean
        
        return delta

class EquivariantVectorMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        assert dim % 2 == 0
        self.n_vectors = dim // 2
        
        self.gate_proj = nn.Linear(self.n_vectors, hidden_dim)
        self.feat_proj = nn.Linear(self.n_vectors, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.n_vectors)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        vectors = x.view(B, self.n_vectors, 2, H, W)
        
        ke_in = torch.sum(vectors ** 2, dim=2)
        magnitudes = torch.sqrt(ke_in + 1e-8)
        mag_flat = magnitudes.permute(0, 2, 3, 1)
        
        gate = self.gate_proj(mag_flat)
        feat = self.feat_proj(mag_flat)
        hidden = F.silu(gate) * feat
        
        raw_scales = self.out_proj(hidden)
        scales = torch.tanh(raw_scales) 
        
        scales = scales.permute(0, 3, 1, 2).unsqueeze(2)
        
        out_vectors = vectors * scales
        
        return out_vectors.view(B, C, H, W)

class LieAlgebraRotation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.skew_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        A = self.skew_generator.triu(diagonal=1)
        A = A - A.t()
        R = torch.linalg.matrix_exp(A)
        
        x_flat = x.permute(0, 2, 3, 1)
        out = F.linear(x_flat, R)
        return out.permute(0, 3, 1, 2)

class SymplecticExchange(nn.Module):
    def __init__(self, scalar_dim, vector_dim):
        super().__init__()
        self.s_dim = scalar_dim
        self.v_dim = vector_dim
        
        ctrl_hidden = max(vector_dim // 2, 2)
        
        self.coupling_controller = nn.Sequential(
            nn.Linear(1, ctrl_hidden), 
            nn.SiLU(),
            nn.Linear(ctrl_hidden, 1),
            nn.Tanh()
        )
        
        nn.init.zeros_(self.coupling_controller[2].weight)
        nn.init.zeros_(self.coupling_controller[2].bias)

    def forward(self, scalar, vector):
        vector_energy = (vector ** 2).mean(dim=1, keepdim=True)
        theta = self.coupling_controller(vector_energy.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        s_norm = torch.sqrt(torch.sum(scalar ** 2, dim=1, keepdim=True) + 1e-8)
        v_norm = torch.sqrt(torch.sum(vector ** 2, dim=1, keepdim=True) + 1e-8)
        
        total_amp = torch.sqrt(s_norm**2 + v_norm**2 + 1e-8)
        current_angle = torch.atan2(v_norm, s_norm)
        
        new_angle = current_angle + theta * 0.5 
        
        new_s_norm = total_amp * torch.cos(new_angle)
        new_v_norm = total_amp * torch.sin(new_angle)
        
        scalar_out = scalar * (new_s_norm / (s_norm + 1e-8))
        vector_out = vector * (new_v_norm / (v_norm + 1e-8))
        
        return scalar_out, vector_out

class UniPhyParaPool(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.dim = dim
        self.scalar_dim = dim // 4
        self.vector_dim = dim - self.scalar_dim
        
        self.norm = nn.LayerNorm(dim)
        
        scalar_hidden = int(self.scalar_dim * expand)
        vector_hidden = int(self.vector_dim * expand)
        
        self.scalar_op = FluxConservingSwiGLU(self.scalar_dim, scalar_hidden)
        self.vector_mixing = LieAlgebraRotation(self.vector_dim)
        self.vector_op = EquivariantVectorMLP(self.vector_dim, vector_hidden)
        
        self.symplectic_exchange = SymplecticExchange(self.scalar_dim, self.vector_dim)

    def forward(self, x):
        x_in = x.permute(0, 2, 3, 1)
        x_in = self.norm(x_in)
        x = x_in.permute(0, 3, 1, 2)
        
        x_scalar = x[:, :self.scalar_dim, :, :]
        x_vector = x[:, self.scalar_dim:, :, :]
        
        x_vector_rot = self.vector_mixing(x_vector)
        
        delta_vector = self.vector_op(x_vector_rot)
        delta_scalar = self.scalar_op(x_scalar)
        
        delta_scalar, delta_vector = self.symplectic_exchange(delta_scalar, delta_vector)
        
        out = torch.cat([delta_scalar, delta_vector], dim=1)
        return out

