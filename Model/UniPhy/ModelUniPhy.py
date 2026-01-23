import torch
import torch.nn as nn
import torch.nn.functional as F

from PScan import PScanTriton 
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork

class UniPhyBlock(nn.Module):
    def __init__(self, dim, expand, num_experts, img_height, img_width, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.img_height = img_height
        self.img_width = img_width
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=kernel_size//2, img_height=img_height, img_width=img_width)
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(dim, dt_ref=1.0, noise_scale=0.01)
        self.pscan = PScanTriton()
        self.ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts)
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
        return torch.complex(r, i)

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
        dt_t = torch.as_tensor(dt_step, device=x.device, dtype=x.real.dtype)
        if dt_t.numel() == B:
            dt_expanded = dt_t.view(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
        else:
            dt_expanded = dt_t.reshape(-1, 1)
        prop_out = self.prop.forward(h_prev, x_t, dt_expanded)
        if isinstance(prop_out, tuple):
            h_next = prop_out[0]
        else:
            h_next = prop_out
        x_drift = h_next.real.view(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
        x = x_drift + resid
        x = x + self.ffn(x)
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
        dt_expanded = torch.as_tensor(dt, device=x.device).view(B, 1, 1, T).expand(B, H, W, T).reshape(B * H * W, T)
        ops = self.prop.get_transition_operators(dt_expanded, x_t)
        if len(ops) == 3:
            op_decay, op_forcing, _ = ops
        else:
            op_decay, op_forcing = ops
        bias = self.prop._get_source_bias()
        x_encoded = self.prop.basis.encode(x_t)
        u_t = (x_encoded + bias) * op_forcing
        noise = self.prop.generate_stochastic_term(u_t.shape, dt_expanded, u_t.dtype)
        u_t = u_t + noise
        h_eigen = self.pscan(op_decay, u_t)
        self.last_h_state = self.prop.basis.decode(h_eigen[:, -1, :])
        x_drift = self.prop.basis.decode(h_eigen).real.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        x = x_drift + resid
        x_in = x.view(B * T, D, H, W)
        delta_p = self.ffn(x_in)
        x = x + delta_p.view(B, T, D, H, W)
        return x

class UniPhyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embed_dim=64, expand=4, num_experts=4, depth=4, patch_size=16, img_height=64, img_width=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim, w_dim = (img_height + pad_h) // patch_size, (img_width + pad_w) // patch_size
        self.encoder = UniPhyEncoder(in_channels, embed_dim, patch_size, img_height, img_width)
        self.blocks = nn.ModuleList([UniPhyBlock(embed_dim, expand, num_experts, h_dim, w_dim) for _ in range(depth)])
        self.decoder = UniPhyEnsembleDecoder(out_channels, embed_dim, patch_size, img_height=img_height)
        self.fusion_weights = nn.Parameter(torch.ones(depth, dtype=torch.float32))
        self.ic_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x, dt):
        z = self.encoder(x)
        z_ic = z.clone()
        block_outputs = []
        for block in self.blocks:
            z = z + z_ic * self.ic_scale
            z = block(z, dt)
            block_outputs.append(z)
        weights = F.softmax(self.fusion_weights, dim=0)
        z_fused = 0
        for w, out in zip(weights, block_outputs):
            z_fused = z_fused + w * out
        return self.decoder(z_fused, x)
        
    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        z = self.encoder(x_cond)
        z_ic = z.clone()
        states = []
        for block in self.blocks:
            z = z + z_ic * self.ic_scale
            z = block(z, dt_cond)
            states.append(block.last_h_state.to("cpu"))
            block.last_h_state = None 
        z_curr = z[:, -1].detach()
        del z            
        predictions = []
        for k in range(k_steps):
            dt_k = dt_future[:, k] if (isinstance(dt_future, torch.Tensor) and dt_future.ndim > 0) else dt_future
            z_next = z_curr
            new_states = []
            step_outputs = []
            for i, block in enumerate(self.blocks):
                h_prev = states[i].to(device, non_blocking=True)
                z_next, h_next = block.forward_step(z_next, h_prev, dt_k)
                step_outputs.append(z_next)
                new_states.append(h_next.to("cpu", non_blocking=True))
                del h_prev
                del h_next
            states = new_states
            z_curr = z_next
            weights = F.softmax(self.fusion_weights, dim=0)
            z_fused_step = 0
            for w, out in zip(weights, step_outputs):
                z_fused_step = z_fused_step + w * out
            pred_pixel = self.decoder(z_fused_step.unsqueeze(1), None).squeeze(1).to("cpu", non_blocking=True)
            predictions.append(pred_pixel)
        return torch.stack(predictions, dim=1)
    