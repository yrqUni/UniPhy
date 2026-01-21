import torch
import torch.nn as nn

from PScan import PScanTriton
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import MetriplecticPropagator, RiemannianCliffordConv2d, SpectralStep, GatedChannelMixer

class UniPhyBlock(nn.Module):
    def __init__(self, dim, img_height, img_width, kernel_size=3, expand=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.img_height = img_height
        self.img_width = img_width
        
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(
            dim * 2, dim * 2, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            img_height=img_height,
            img_width=img_width 
        )
        self.spatial_spec = SpectralStep(dim, img_height, img_width)
        self.spatial_gate = nn.Parameter(torch.ones(1) * 0.5)

        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = MetriplecticPropagator(dim, img_height, img_width, dt_ref=1.0, stochastic=True)
        self.pscan = PScanTriton()

        self.mlp_norm = nn.LayerNorm(dim * 2)
        self.mlp = GatedChannelMixer(dim * 2, expand=expand, dropout=dropout)

    def forward(self, x, dt):
        B, T, C, H, W = x.shape
        residual = x
        
        x_flat = x.view(B * T, C, H, W)
        x_perm = x_flat.permute(0, 2, 3, 1)
        x_cat = torch.cat([x_perm.real, x_perm.imag], dim=-1)
        x_norm = self.norm_spatial(x_cat)
        r, i = torch.chunk(x_norm, 2, dim=-1)
        x_norm_c = torch.complex(r, i).permute(0, 3, 1, 2)
        
        x_cat_conv = torch.cat([x_norm_c.real, x_norm_c.imag], dim=1)
        out_cliff = self.spatial_cliff(x_cat_conv)
        r_c, i_c = torch.chunk(out_cliff, 2, dim=1)
        out_cliff_c = torch.complex(r_c, i_c)
        out_spec = self.spatial_spec(x_norm_c)
        
        z = self.spatial_gate * out_cliff_c + (1.0 - self.spatial_gate) * out_spec
        z = z.view(B, T, C, H, W) + residual
        
        residual = z
        
        z_perm = z.permute(0, 1, 3, 4, 2)
        z_cat = torch.cat([z_perm.real, z_perm.imag], dim=-1)
        z_norm = self.norm_temporal(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        z_in = torch.complex(r, i).permute(0, 1, 4, 2, 3)
        
        C_op, B_op, A_op, dt_eff = self.prop.get_operators(dt, x_context=z_in)
        
        if B_op is None:
            u = z_in
        elif B_op.ndim >= 2 and B_op.shape[-1] == C and B_op.shape[-2] == C:
             z_flat = z_in.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, C)
             if not B_op.is_complex():
                 B_op = B_op.to(dtype=z_in.dtype)
             u_flat = torch.matmul(z_flat, B_op.transpose(-1, -2)) 
             u = u_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        else:
             if B_op.ndim == 3:
                 B_op_cast = B_op.unsqueeze(-1).unsqueeze(-1)
             elif B_op.ndim == z_in.ndim:
                 B_op_cast = B_op
             else:
                 B_op_cast = B_op
                 
             if not B_op_cast.is_complex() and z_in.is_complex():
                 B_op_cast = B_op_cast.to(dtype=z_in.dtype)
                 
             u = z_in * B_op_cast

        u_t = u.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        if A_op.ndim == 5:
            A_expanded = A_op.expand(B, H, W, T, C)
        elif A_op.ndim == 3:
            A_expanded = A_op.unsqueeze(1).unsqueeze(1).expand(B, H, W, T, C)
        else:
            A_expanded = A_op.view(B, 1, 1, -1, C).expand(B, H, W, T, C)
            
        A_flat = A_expanded.reshape(B * H * W, T, C)
        h = self.pscan(A_flat, u_t)
        
        if C_op is None:
            y = h
        else:
            if not C_op.is_complex():
                 C_op = C_op.to(dtype=y.dtype)
            y = torch.matmul(h, C_op.transpose(-1, -2))
            
        x_drift = y.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        noise = self.prop.inject_noise(z, dt_eff)
        
        x = x_drift + noise + residual
        
        residual = x
        
        if x.shape[2] == self.dim * 2 or x.shape[2] == self.dim:
             x_in = x.permute(0, 1, 3, 4, 2)
        else:
             x_in = x
             
        x_cat = torch.cat([x_in.real, x_in.imag], dim=-1)
        x_norm = self.mlp_norm(x_cat)
        x_mlp = self.mlp(x_norm)
        
        x_r, x_i = torch.chunk(x_mlp, 2, dim=-1)
        x_out = torch.complex(x_r, x_i)
        
        if x_out.ndim == 5 and x_out.shape[1] == T and x_out.shape[2] == H:
             x_out = x_out.permute(0, 1, 4, 2, 3)
             
        x = x + x_out
        return x

class UniPhyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embed_dim=64, depth=4, patch_size=16, img_height=64, img_width=128, dropout=0.0):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim = (img_height + pad_h) // patch_size
        w_dim = (img_width + pad_w) // patch_size
        
        self.encoder = UniPhyEncoder(in_channels, embed_dim, patch_size, img_height, img_width)
        self.blocks = nn.ModuleList([
            UniPhyBlock(embed_dim, h_dim, w_dim, dropout=dropout) 
            for _ in range(depth)
        ])
        self.decoder = UniPhyEnsembleDecoder(out_channels, embed_dim, patch_size, img_height=img_height)

    def forward(self, x, dt):
        z = self.encoder(x)
        for block in self.blocks:
            z = block(z, dt)
        out = self.decoder(z, x)
        return out

