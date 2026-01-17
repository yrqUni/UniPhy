import torch
import torch.nn as nn
from PScan import PScanTriton
from HamiltonianPropagator import HamiltonianPropagator
from CliffordConv2d import CliffordConv2d
from SpectralStep import SpectralStep

class UniPhyTransformerBlock(nn.Module):
    def __init__(self, dim, state_dim, kernel_size=3, expand=2, dropout=0.0, img_height=64, img_width=64):
        super().__init__()
        self.dim = dim
        self.expand = expand
        
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = CliffordConv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.spatial_spec = SpectralStep(dim, img_height, img_width)
        self.spatial_gate = nn.Parameter(torch.ones(1) * 0.5)

        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = HamiltonianPropagator(dim, dt_ref=1.0, conserve_energy=False)
        self.pscan = PScanTriton()
        
        self.norm_mlp = nn.LayerNorm(dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * expand * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand * 2, dim * 2),
            nn.Dropout(dropout)
        )

    def _complex_norm(self, z, norm_layer):
        z_cat = torch.cat([z.real, z.imag], dim=-1)
        z_norm = norm_layer(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        return torch.complex(r, i)

    def _complex_mlp(self, z):
        z_cat = torch.cat([z.real, z.imag], dim=-1)
        z_out = self.mlp(z_cat)
        r, i = torch.chunk(z_out, 2, dim=-1)
        return torch.complex(r, i)

    def _spatial_op(self, x):
        B, C, H, W = x.shape
        x_real_imag = torch.cat([x.real, x.imag], dim=1)
        
        out_cliff = self.spatial_cliff(x_real_imag)
        r, i = torch.chunk(out_cliff, 2, dim=1)
        out_cliff_c = torch.complex(r, i)
        
        out_spec = self.spatial_spec(x)
        
        return self.spatial_gate * out_cliff_c + (1.0 - self.spatial_gate) * out_spec

    def forward_parallel(self, x, dt):
        B, T, D, H, W = x.shape
        resid = x
        
        x_s = x.view(B * T, D, H, W)
        x_s_perm = x_s.permute(0, 2, 3, 1)
        x_s_norm = self._complex_norm(x_s_perm, self.norm_spatial)
        x_s = x_s_norm.permute(0, 3, 1, 2)
        
        x_s = self._spatial_op(x_s)
        x = x_s.view(B, T, D, H, W) + resid
        
        resid = x
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        dt_expanded = dt.repeat_interleave(H * W, dim=0)
        V, V_inv, evo_diag = self.prop.get_operators(dt_expanded)
        
        x_eigen = torch.matmul(x_t, V_inv.T)
        h_eigen = self.pscan(evo_diag, x_eigen)
        x_t_out = torch.matmul(h_eigen, V.T)
        
        x_t_out = x_t_out.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        x = x_t_out + resid
        
        resid = x
        x_c = x.permute(0, 1, 3, 4, 2)
        x_c = self._complex_norm(x_c, self.norm_mlp)
        x_c = self._complex_mlp(x_c)
        x_c = x_c.permute(0, 1, 4, 2, 3)
        x = x_c + resid
        
        return x

    def step_serial(self, x_curr, state, dt):
        B, D, H, W = x_curr.shape
        
        resid = x_curr
        x_s = x_curr.permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial)
        x_s = x_s.permute(0, 3, 1, 2)
        
        x_s = self._spatial_op(x_s)
        x = x_s + resid
        
        resid = x
        x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        if isinstance(dt, float):
            dt_tensor = torch.tensor([dt], device=x.device).expand(B * H * W, 1)
        elif dt.ndim == 0:
            dt_tensor = dt.expand(B * H * W, 1)
        else:
            dt_tensor = dt.repeat_interleave(H * W, dim=0)
            
        V, V_inv, evo_diag = self.prop.get_operators(dt_tensor)
        evo_diag = evo_diag.squeeze(1)
        
        x_eigen = torch.matmul(x_t, V_inv.T)
        
        if state is None:
            h_prev = torch.zeros_like(x_eigen)
        else:
            h_prev = state.reshape(B * H * W, D)
            
        h_curr = evo_diag * h_prev + x_eigen
        
        x_t_out = torch.matmul(h_curr, V.T)
        
        x_t_out = x_t_out.view(B, H, W, D).permute(0, 3, 1, 2)
        x = x_t_out + resid
        
        new_state = h_curr.view(B, H, W, D)
        
        resid = x
        x_c = x.permute(0, 2, 3, 1)
        x_c = self._complex_norm(x_c, self.norm_mlp)
        x_c = self._complex_mlp(x_c)
        x_c = x_c.permute(0, 3, 1, 2)
        x = x_c + resid
        
        return x, new_state

