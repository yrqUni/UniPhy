import torch
import torch.nn as nn
import torch.nn.functional as F

class MassConservingSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        resid = x
        
        x_in = x.permute(0, 2, 3, 1)
        
        x1 = self.w1(x_in)
        x2 = self.w2(x_in)
        hidden = F.silu(x1) * x2
        hidden = self.dropout(hidden)
        delta = self.w3(hidden)
        
        delta = delta.permute(0, 3, 1, 2)
        
        delta_mean = delta.mean(dim=(2, 3), keepdim=True)
        delta = delta - delta_mean
            
        return resid + delta

class HighCapacityVectorMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        assert dim % 2 == 0
        self.n_vectors = dim // 2
        
        self.gate_proj = nn.Linear(self.n_vectors, hidden_dim)
        self.feat_proj = nn.Linear(self.n_vectors, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.n_vectors)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        vectors = x.view(B, self.n_vectors, 2, H, W)
        magnitudes = torch.norm(vectors, dim=2)
        
        mag_flat = magnitudes.permute(0, 2, 3, 1)
        
        gate = self.gate_proj(mag_flat)
        feat = self.feat_proj(mag_flat)
        hidden = F.silu(gate) * feat
        hidden = self.dropout(hidden)
        
        scales = self.out_proj(hidden)
        scales = torch.sigmoid(scales) * 2.0
        
        scales = scales.permute(0, 3, 1, 2).unsqueeze(2)
        
        out_vectors = vectors * scales
        return out_vectors.view(B, C, H, W)

class LieAlgebraRotation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.skew_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        
        A = self.skew_generator.triu(diagonal=1)
        A = A - A.t()
        
        R = torch.linalg.matrix_exp(A)
        
        x_flat = x.permute(0, 2, 3, 1)
        out = F.linear(x_flat, R)
        return out.permute(0, 3, 1, 2)

class UniPhyParaPool(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.scalar_dim = dim // 4
        self.vector_dim = dim - self.scalar_dim
        
        scalar_hidden = int(self.scalar_dim * expand)
        vector_hidden = int(self.vector_dim * expand)
        
        self.scalar_op = MassConservingSwiGLU(self.scalar_dim, scalar_hidden, dropout)
        
        self.vector_mixing = LieAlgebraRotation(self.vector_dim)
        self.vector_op = HighCapacityVectorMLP(self.vector_dim, vector_hidden, dropout)

    def forward(self, x):
        x_scalar = x[:, :self.scalar_dim, :, :]
        x_vector = x[:, self.scalar_dim:, :, :]
        
        out_scalar = self.scalar_op(x_scalar)
        
        x_vector_rot = self.vector_mixing(x_vector)
        out_vector = self.vector_op(x_vector_rot)
        
        out = torch.cat([out_scalar, out_vector], dim=1)
        return out

