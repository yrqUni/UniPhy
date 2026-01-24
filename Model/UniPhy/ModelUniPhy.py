import torch
import torch.nn as nn

from PScan import PScanTriton 
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork

class UniPhyBlock(nn.Module):
    def __init__(self, dim, expand, num_experts, img_height, img_width, kernel_size=3, dt_ref=1.0):
        super().__init__()
        self.dim = dim
        self.img_height = img_height
        self.img_width = img_width
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=kernel_size//2, img_height=img_height, img_width=img_width)
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(dim, dt_ref=dt_ref, noise_scale=0.01)
        self.pscan = PScanTriton()
        self.ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts)
        self.last_h_state = None
        self.last_flux_state = None

    def _complex_norm(self, z, norm_layer):
        z_cat = torch.cat([z.real, z.imag], dim=-1)
        z_norm = norm_layer(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        return torch.complex(r, i)

    def _spatial_op(self, x):
        x_real_imag = torch.cat([x.real, x.imag], dim=1)
        out_cliff = self.spatial_cliff(x_real_imag)
        r, i = torch.chunk(out_cliff, 2, dim=1)
        return torch.complex(r, i)

    def forward_step(self, x_step, h_prev, dt_step, flux_prev):
        B, D, H, W = x_step.shape
        resid = x_step
        x_s = x_step.permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x = x_s + resid
        resid = x
        x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        dt_t = torch.as_tensor(dt_step, device=x.device, dtype=x.real.dtype)
        if dt_t.numel() == B:
            dt_expanded = dt_t.reshape(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
        else:
            dt_expanded = dt_t.reshape(-1, 1)
        
        x_encoded_local = self.prop.basis.encode(x_t)
        x_global_mean_encoded = x_encoded_local.view(B, H * W, D).mean(dim=1)
        
        h_next, flux_next = self.prop.forward_step(h_prev, x_t, x_global_mean_encoded, dt_expanded, flux_prev)
        
        x_drift = h_next.real.reshape(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
        x = x_drift + resid
        x = x + self.ffn(x)
        return x, h_next, flux_next

    def forward(self, x, dt):
        B, T, D, H, W = x.shape
        resid = x
        x_s = x.reshape(B * T, D, H, W).permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x = x_s.reshape(B, T, D, H, W) + resid
        resid = x
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        dt_expanded = torch.as_tensor(dt, device=x.device).reshape(B, 1, 1, T).expand(B, H, W, T).reshape(B * H * W, T)
        
        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        
        x_eigen = self.prop.basis.encode(x_t)
        
        x_eigen_input = x_eigen.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2) 
        x_mean_seq = x_eigen_input.mean(dim=(-2, -1))
        
        flux_A, flux_X = self.prop.flux_tracker.get_operators(x_mean_seq)
        flux_states = self.pscan(flux_A, flux_X)
        source_seq = self.prop.flux_tracker.project(flux_states)
        
        source_expanded = source_seq.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, D)
        source_flat = source_expanded.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        
        forcing_term = x_eigen + source_flat
        
        u_t = forcing_term * op_forcing
        noise = self.prop.generate_stochastic_term(u_t.shape, dt_expanded, u_t.dtype)
        u_t = u_t + noise
        
        N = u_t.shape[0]
        
        A = op_decay.unsqueeze(2).expand(N, T, D).permute(0, 2, 1).contiguous()
        X = u_t.permute(0, 2, 1).contiguous()
        
        h_eigen_perm = self.pscan(A, X)
        h_eigen = h_eigen_perm.permute(0, 2, 1)
        
        self.last_h_state = self.prop.basis.decode(h_eigen[:, -1, :])
        self.last_flux_state = flux_states[:, :, -1].permute(0, 1) 
        
        x_drift = self.prop.basis.decode(h_eigen).real.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        x = x_drift + resid
        x_in = x.reshape(B * T, D, H, W)
        delta_p = self.ffn(x_in)
        x = x + delta_p.reshape(B, T, D, H, W)
        return x

class UniPhyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embed_dim=64, expand=4, num_experts=4, depth=4, patch_size=16, img_height=64, img_width=128, dt_ref=1.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim, w_dim = (img_height + pad_h) // patch_size, (img_width + pad_w) // patch_size
        self.encoder = UniPhyEncoder(in_channels, embed_dim, patch_size, img_height, img_width)
        self.blocks = nn.ModuleList([UniPhyBlock(embed_dim, expand, num_experts, h_dim, w_dim, dt_ref=dt_ref) for _ in range(depth)])
        self.decoder = UniPhyEnsembleDecoder(out_channels, embed_dim, patch_size, img_height=img_height)

    def forward(self, x, dt):
        z = self.encoder(x)
        for block in self.blocks: z = block(z, dt)
        return self.decoder(z)
        
    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        z = self.encoder(x_cond)
        
        states = []
        for block in self.blocks:
            z = block(z, dt_cond)
            
            curr_flux = torch.zeros(x_cond.shape[0], block.dim, device=device, dtype=torch.cdouble)
            if block.last_flux_state is not None:
                curr_flux = block.last_flux_state
            elif z.shape[1] > 0:
                z_perm = z.permute(0, 1, 3, 4, 2)
                x_encoded = block.prop.basis.encode(z_perm)
                x_eigen_last_seq = x_encoded.mean(dim=(2, 3))
                for t in range(x_eigen_last_seq.shape[1]):
                    curr_flux, _ = block.prop.flux_tracker.forward_step(curr_flux, x_eigen_last_seq[:, t])

            states.append((block.last_h_state.to("cpu"), curr_flux.to("cpu")))
            block.last_h_state = None 
            block.last_flux_state = None
            
        z_curr = z[:, -1].detach()
        del z            
        
        predictions = []
        for k in range(k_steps):
            dt_k = dt_future[:, k] if (isinstance(dt_future, torch.Tensor) and dt_future.ndim > 0) else dt_future
            z_next = z_curr
            new_states = []
            
            for i, block in enumerate(self.blocks):
                h_prev = states[i][0].to(device, non_blocking=True)
                flux_prev = states[i][1].to(device, non_blocking=True)
                
                z_next, h_next, flux_next = block.forward_step(z_next, h_prev, dt_k, flux_prev)
                new_states.append((h_next.to("cpu", non_blocking=True), flux_next.to("cpu", non_blocking=True)))
                del h_prev
                del h_next
                
            states = new_states
            z_curr = z_next
            pred_pixel = self.decoder(z_curr.unsqueeze(1)).squeeze(1).to("cpu", non_blocking=True)
            predictions.append(pred_pixel)
            
        return torch.stack(predictions, dim=1)
    