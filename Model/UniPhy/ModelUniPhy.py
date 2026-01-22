import torch
import torch.nn as nn

from PScan import PScanTriton 
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import AnalyticSpectralPropagator, RiemannianCliffordConv2d, SpectralStep
from UniPhyParaPool import UniPhyParaPool

class UniPhyBlock(nn.Module):
    def __init__(self, dim, img_height, img_width, kernel_size=3, expand=4):
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
        self.prop = AnalyticSpectralPropagator(dim, dt_ref=1.0)
        
        self.pscan = PScanTriton()
        
        self.para_pool = UniPhyParaPool(dim * 2, expand=expand)
        self.last_h_state = None

    def _complex_norm(self, z, norm_layer):
        z_cat = torch.cat([z.real, z.imag], dim=-1)
        z_norm = norm_layer(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        return torch.complex(r, i)

    def _spatial_op(self, x):
        x_real_imag = torch.cat([x.real, x.imag], dim=1)
        out_cliff = self.spatial_cliff(x_real_imag)
        r, i = torch.chunk(out_cliff, 2, dim=1)
        out_cliff_c = torch.complex(r, i)
        out_spec = self.spatial_spec(x)
        return self.spatial_gate * out_cliff_c + (1.0 - self.spatial_gate) * out_spec

    def forward_step(self, x_step, h_prev, dt_step):
        B, D, H, W = x_step.shape
        resid = x_step
        
        x_s = x_step.permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x = x_s + resid
        resid = x
        
        x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        op_decay, op_forcing = self.prop.get_transition_operators(dt_step)
        
        target_shape = (B * H * W, 1, D)
        if op_decay.shape[0] == B:
             op_decay = op_decay.unsqueeze(1).unsqueeze(1).expand(B, H, W, D).reshape(*target_shape)
             op_forcing = op_forcing.unsqueeze(1).unsqueeze(1).expand(B, H, W, D).reshape(*target_shape)
        else:
             op_decay = op_decay.view(1, 1, D).expand(*target_shape)
             op_forcing = op_forcing.view(1, 1, D).expand(*target_shape)

        x_eigen = self.prop.basis.encode(x_t)
        
        h_next = h_prev.unsqueeze(1) * op_decay + x_eigen * op_forcing
        h_next = h_next.squeeze(1)
        
        x_out_complex = self.prop.basis.decode(h_next.unsqueeze(1))
        x_drift = x_out_complex.real.view(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
        
        x = x_drift + resid
        resid = x
        
        x_p = x.permute(0, 2, 3, 1)
        x_p_flat = torch.cat([x_p.real, x_p.imag], dim=-1)
        x_pool_out = self.para_pool(x_p_flat.permute(0, 3, 1, 2))
        x_pool_out = x_pool_out.permute(0, 2, 3, 1)
        r, i = torch.chunk(x_pool_out, 2, dim=-1)
        x = torch.complex(r, i).permute(0, 3, 1, 2) + resid
        
        return x, h_next

    def forward(self, x, dt):
        B, T, D, H, W = x.shape
        resid = x
        
        x_s = x.view(B * T, D, H, W).permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x = x_s.view(B, T, D, H, W) + resid
        
        resid = x
        
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        op_decay, op_forcing = self.prop.get_transition_operators(dt)
        
        target_shape = (B * H * W, T, D)
        
        if op_decay.shape[0] == B * T: 
            op_decay = op_decay.view(B, T, D).unsqueeze(1).unsqueeze(1)
            op_decay = op_decay.expand(B, H, W, T, D).reshape(*target_shape)
            
            op_forcing = op_forcing.view(B, T, D).unsqueeze(1).unsqueeze(1)
            op_forcing = op_forcing.expand(B, H, W, T, D).reshape(*target_shape)
            
        elif op_decay.ndim == 3 and op_decay.shape[0] == B:
            op_decay = op_decay.unsqueeze(1).unsqueeze(1).expand(B, H, W, T, D).reshape(*target_shape)
            op_forcing = op_forcing.unsqueeze(1).unsqueeze(1).expand(B, H, W, T, D).reshape(*target_shape)
            
        else:
            op_decay = op_decay.view(1, 1, D).expand(*target_shape)
            op_forcing = op_forcing.view(1, 1, D).expand(*target_shape)
            
        x_eigen = self.prop.basis.encode(x_t)
        
        u_t = x_eigen * op_forcing
        
        h_eigen = self.pscan(op_decay, u_t)
        
        self.last_h_state = h_eigen[:, -1, :] 
        
        x_t_out_complex = self.prop.basis.decode(h_eigen)
        
        x_drift = x_t_out_complex.real.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        x = x_drift + resid
        
        resid = x
        
        x_p = x.permute(0, 1, 3, 4, 2) 
        x_p_flat = torch.cat([x_p.real, x_p.imag], dim=-1)
        B_p, T_p, H_p, W_p, C_p = x_p_flat.shape
        x_p_in = x_p_flat.view(B_p * T_p, H_p, W_p, C_p).permute(0, 3, 1, 2)
        x_pool_out = self.para_pool(x_p_in)
        x_pool_out = x_pool_out.permute(0, 2, 3, 1).view(B_p, T_p, H_p, W_p, C_p)
        r, i = torch.chunk(x_pool_out, 2, dim=-1)
        x = torch.complex(r, i).permute(0, 1, 4, 2, 3) + resid
        
        return x

class UniPhyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embed_dim=64, depth=4, patch_size=16, img_height=64, img_width=128):
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
            UniPhyBlock(embed_dim, h_dim, w_dim) 
            for _ in range(depth)
        ])
        
        self.decoder = UniPhyEnsembleDecoder(out_channels, embed_dim, patch_size, img_height=img_height)

    def forward(self, x, dt):
        z = self.encoder(x)
        for block in self.blocks:
            z = block(z, dt)
        out = self.decoder(z, x)
        return out
        
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        B, L, C, H, W = x_cond.shape
        z = self.encoder(x_cond)
        
        states = []
        for block in self.blocks:
            z = block(z, dt_cond)
            states.append(block.last_h_state)
            
        predictions = []
        z_curr = z[:, -1]
        
        for k in range(k_steps):
            if isinstance(dt_future, torch.Tensor) and dt_future.ndim > 0:
                dt_k = dt_future[:, k] if dt_future.shape[1] > 1 else dt_future[:, 0]
            else:
                dt_k = dt_future
            
            z_next = z_curr
            new_states = []
            for i, block in enumerate(self.blocks):
                h_prev = states[i]
                z_next, h_next = block.forward_step(z_next, h_prev, dt_k)
                new_states.append(h_next)
            
            states = new_states
            z_curr = z_next
            
            pred_pixel = self.decoder(z_curr.unsqueeze(1), None).squeeze(1)
            predictions.append(pred_pixel)
            
        return torch.stack(predictions, dim=1)

