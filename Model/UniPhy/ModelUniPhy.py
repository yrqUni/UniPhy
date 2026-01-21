import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from UniPhyOps import RiemannianCliffordConv2d, SpectralStep, MetriplecticPropagator
from UniPhyIO import UniPhyInputEmbedding, UniPhyEnsembleDecoder, GlobalConservationConstraint
from PScan import PScanTriton

class PortHamiltonianLayer(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expand)
        self.potential_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.J_generator = nn.Linear(dim, dim * dim)
        self.R_generator = nn.Linear(dim, dim * dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, dt=1.0):
        x_in = self.norm(x)
        B_size = x.shape[:-1]
        dim = x.shape[-1]
        
        with torch.enable_grad():
            x_in.requires_grad_(True)
            H = self.potential_net(x_in).sum()
            grads = torch.autograd.grad(H, x_in, create_graph=True)[0]
            
        M = self.J_generator(x_in).view(*B_size, dim, dim)
        J = M - M.transpose(-1, -2)
        
        P = self.R_generator(x_in).view(*B_size, dim, dim)
        R = torch.matmul(P.transpose(-1, -2), P) + 1e-4 * torch.eye(dim, device=x.device)
        
        update = torch.matmul(J - R, grads.unsqueeze(-1)).squeeze(-1)
        
        out = x + update * dt
        return out

class UniPhyBlock(nn.Module):
    def __init__(self, dim, img_height, img_width, kernel_size=3, expand=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.H = img_height
        self.W = img_width
        
        self.spatial_mixer = RiemannianCliffordConv2d(
            dim, dim, kernel_size, padding=kernel_size//2, 
            img_height=img_height, img_width=img_width
        )
        
        self.spectral_mixer = SpectralStep(dim, img_height, img_width)
        
        self.propagator = MetriplecticPropagator(dim, img_height, img_width)
        self.pscan = PScanTriton()
        
        self.ph_layer = PortHamiltonianLayer(dim * 2, expand=expand, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)

    def forward(self, x, dt):
        residual = x
        
        x_spatial = self.spatial_mixer(x)
        x = x + x_spatial
        
        x_spectral = self.spectral_mixer(x)
        x = x + x_spectral
        
        u_flat, A_flat, dt_eff, shape_info = self.propagator.forward_spectral(x, dt)
        B_out, T, C, H, W_f = shape_info
        
        h_flat = self.pscan(A_flat, u_flat)
        
        h_freq = h_flat.view(B_out, H, W_f, T, C).permute(0, 3, 4, 1, 2)
        
        x_temporal = torch.fft.irfft2(h_freq, s=(self.H, self.W), norm='ortho')
        
        if B_out == 2 * x.shape[0]:
            x_r, x_i = torch.chunk(x_temporal, 2, dim=0)
            x_temporal = torch.complex(x_r, x_i)
        
        noise = self.propagator.inject_noise(x, dt_eff)
        
        x = x + x_temporal + noise
        
        x_perm = x.permute(0, 1, 3, 4, 2)
        x_cat = torch.cat([x_perm.real, x_perm.imag], dim=-1)
        
        dt_scalar = dt_eff.mean() if isinstance(dt_eff, torch.Tensor) else dt
        x_ph = self.ph_layer(x_cat, dt=dt_scalar)
        
        x_ph = x_ph.permute(0, 1, 4, 2, 3)
        x_r, x_i = torch.chunk(x_ph, 2, dim=2)
        x_out = torch.complex(x_r, x_i)
        
        return x_out

class UniPhyModel(nn.Module):
    def __init__(self, 
                 in_channels, 
                 dim, 
                 out_channels, 
                 img_height, 
                 img_width, 
                 patch_size=16, 
                 depth=4, 
                 kernel_size=3, 
                 expand=4, 
                 dropout=0.0,
                 ensemble_size=1):
        super().__init__()
        
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        
        latent_h = (img_height + pad_h) // patch_size
        latent_w = (img_width + pad_w) // patch_size

        self.embedding = UniPhyInputEmbedding(
            in_channels, dim, 
            patch_size=patch_size, 
            img_height=img_height, 
            img_width=img_width
        )
        
        self.blocks = nn.ModuleList([
            UniPhyBlock(dim, latent_h, latent_w, kernel_size, expand, dropout)
            for _ in range(depth)
        ])
        
        self.decoder = UniPhyEnsembleDecoder(
            dim, out_channels, 
            ensemble_size=ensemble_size, 
            patch_size=patch_size, 
            img_height=img_height
        )

    def forward(self, x, dt=1.0, pred_len=1):
        x_raw = x 
        
        x = self.embedding(x)
        
        B, T, C, H, W = x.shape
        if pred_len > T:
             pad_len = pred_len - T
             x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_len))
        
        for block in self.blocks:
            x = block(x, dt)
            
        out = self.decoder(x, x_raw)
        return out

