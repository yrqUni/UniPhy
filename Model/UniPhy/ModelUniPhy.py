import torch
import torch.nn as nn
from UniPhyOps import HamiltonianPropagator
from UniPhyParaPool import UniPhyParaPool
from CliffordConv2d import CliffordConv2d
from SpectralStepTriton import SpectralStep
from UniPhyIO import UniPhyEncoder, UniPhyDiffusionDecoder

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2, eps=eps)

    def forward(self, x):
        x_real = torch.view_as_real(x)
        shape = x_real.shape
        x_flat = x_real.flatten(start_dim=-2)
        x_norm = self.norm(x_flat)
        return torch.view_as_complex(x_norm.view(shape))

class UniPhyTransformerBlock(nn.Module):
    def __init__(self, dim, input_shape, expansion_factor=4):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_shape
        
        self.norm_time = ComplexLayerNorm(dim)
        self.time_mixer = HamiltonianPropagator(dim * 2, conserve_energy=True)
        
        self.norm_space = ComplexLayerNorm(dim)
        self.space_clifford = CliffordConv2d(dim * 2, kernel_size=3, padding=1)
        self.space_spectral = SpectralStep(dim * 2, rank=32, w_freq=self.W // 2 + 1)
        
        self.norm_feat = ComplexLayerNorm(dim)
        self.feature_mixer = UniPhyParaPool(dim, expansion_factor=expansion_factor)

    def forward_parallel(self, x, dt, initial_state=None):
        residual = x
        x = self.norm_time(x)
        
        B, T, H, W, C = x.shape
        x_real_flat = torch.view_as_real(x).flatten(start_dim=-2)
        
        x_out_real, h_final = self.time_mixer.forward_parallel(x_real_flat, dt, initial_state)
        
        x = torch.view_as_complex(x_out_real.view(B, T, H, W, C, 2))
        x = x + residual
        
        residual = x
        x = self.norm_space(x)
        
        x_real_view = torch.view_as_real(x)
        # [B, T, H, W, C, 2] -> [B, T, C, 2, H, W]
        x_real_view = x_real_view.permute(0, 1, 4, 5, 2, 3).contiguous()
        # -> [B*T, 2C, H, W]
        x_real_view = x_real_view.view(B * T, C * 2, H, W)
        
        x_cliff = self.space_clifford(x_real_view)
        
        dt_flat = dt.view(-1).contiguous()
        x_spec = self.space_spectral(x_real_view, dt_flat)
        
        x_space = x_cliff + x_spec
        
        # [B*T, 2C, H, W] -> [B, T, C, 2, H, W]
        x_space = x_space.view(B, T, C, 2, H, W)
        # -> [B, T, H, W, C, 2]
        x_space = x_space.permute(0, 1, 4, 5, 2, 3).contiguous()
        x_space = torch.view_as_complex(x_space)
        
        x = x_space + residual

        residual = x
        x = self.norm_feat(x)
        
        B, T, H, W, C = x.shape
        x_flat = x.view(B * T * H * W, C)[:, :, None, None]
        x_feat = self.feature_mixer(x_flat)
        x_feat = x_feat.view(B, T, H, W, C)
        x = x_feat + residual
        
        return x, h_final

    def step_serial(self, x, dt, h_prev):
        residual = x
        x = self.norm_time(x)
        
        B, H, W, C = x.shape
        x_real_flat = torch.view_as_real(x).flatten(start_dim=-2)
        
        x_out_real, h_next = self.time_mixer.step_serial(x_real_flat, dt, h_prev)
        
        x = torch.view_as_complex(x_out_real.view(B, H, W, C, 2))
        x = x + residual
        
        residual = x
        x = self.norm_space(x)
        
        x_real_view = torch.view_as_real(x)
        # [B, H, W, C, 2] -> [B, C, 2, H, W]
        x_real_view = x_real_view.permute(0, 3, 4, 1, 2).contiguous()
        # -> [B, 2C, H, W]
        x_real_view = x_real_view.view(B, C * 2, H, W)
        
        x_cliff = self.space_clifford(x_real_view)
        x_spec = self.space_spectral(x_real_view, dt)
        
        x_space = x_cliff + x_spec
        
        # [B, 2C, H, W] -> [B, C, 2, H, W]
        x_space = x_space.view(B, C, 2, H, W)
        # -> [B, H, W, C, 2]
        x_space = x_space.permute(0, 3, 4, 1, 2).contiguous()
        x_space = torch.view_as_complex(x_space)
        
        x = x_space + residual

        residual = x
        x = self.norm_feat(x)
        
        x_flat = x.view(B * H * W, C)[:, :, None, None]
        x_feat = self.feature_mixer(x_flat)
        x_feat = x_feat.view(B, H, W, C)
        x = x_feat + residual
        
        return x, h_next

class UniPhyModel(nn.Module):
    def __init__(self, 
                 input_shape=(721, 1440),
                 in_channels=30,
                 dim=64,
                 patch_size=4,
                 num_layers=8,
                 para_pool_expansion=4,
                 conserve_energy=True):
        super().__init__()
        
        self.H_raw, self.W_raw = input_shape
        self.pad_h = (patch_size - self.H_raw % patch_size) % patch_size
        self.pad_w = (patch_size - self.W_raw % patch_size) % patch_size
        self.H_latent = (self.H_raw + self.pad_h) // patch_size
        self.W_latent = (self.W_raw + self.pad_w) // patch_size
        
        self.encoder = UniPhyEncoder(in_channels, dim, patch_size=patch_size)
        
        self.blocks = nn.ModuleList([
            UniPhyTransformerBlock(
                dim=dim,
                input_shape=(self.H_latent, self.W_latent),
                expansion_factor=para_pool_expansion
            ) for _ in range(num_layers)
        ])
        
        self.decoder = UniPhyDiffusionDecoder(
            out_ch=in_channels,
            latent_dim=dim,
            patch_size=patch_size,
            model_channels=dim
        )
        
        self.norm_final = ComplexLayerNorm(dim)

    def forward(self, x, dt):
        B, T, C, H, W = x.shape
        
        x_flat = x.view(B * T, C, H, W)
        z = self.encoder(x_flat)
        _, D, Hl, Wl = z.shape
        z = z.view(B, T, D, Hl, Wl).permute(0, 1, 3, 4, 2)
        
        final_states = []
        for block in self.blocks:
            z, h_final = block.forward_parallel(z, dt)
            final_states.append(h_final)
            
        z = self.norm_final(z)
        z = z.permute(0, 1, 4, 2, 3)
        
        return z, final_states

    @torch.no_grad()
    def inference(self, context_x, context_dt, future_steps, future_dt, diffusion_steps=20):
        B, T, C, H, W = context_x.shape
        
        z_ctx, current_states = self.forward(context_x, context_dt)
        curr_z = z_ctx[:, -1].permute(0, 2, 3, 1)
        
        predictions = []
        
        if isinstance(future_dt, float):
            dt_val = torch.full((B,), future_dt, device=context_x.device)
        else:
            dt_val = future_dt

        for _ in range(future_steps):
            next_states = []
            for i, block in enumerate(self.blocks):
                curr_z, h_next = block.step_serial(curr_z, dt_val, current_states[i])
                next_states.append(h_next)
            current_states = next_states
            
            z_out = self.norm_final(curr_z)
            z_out_perm = z_out.permute(0, 3, 1, 2)
            
            x_sample = self.decoder.sample(z_out_perm, (B, C, H, W), context_x.device, steps=diffusion_steps)
            predictions.append(x_sample)
            
        return torch.stack(predictions, dim=1)

