import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
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

    def forward(self, x, dt):
        B, T, D, H, W = x.shape
        x_s = x.view(B * T, D, H, W).permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x_spatial = x_s.view(B, T, D, H, W).add(x)
        
        x_t = x_spatial.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        dt_tensor = torch.as_tensor(dt, device=x.device)
        dt_expanded = dt_tensor.view(B, 1, 1, T).expand(B, H, W, T).reshape(B * H * W, T)
        
        x_encoded = self.prop.basis.encode(x_t)
        gate = F.silu(self.prop.input_gate(x_encoded.real))
        x_gated = x_encoded.mul(torch.complex(gate, torch.zeros_like(gate)))
        
        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded, x_gated)
        bias = self.prop._get_source_bias()
        u_t = x_gated.add(bias).mul(op_forcing)
        
        noise = self.prop.generate_stochastic_term(u_t.shape, dt_expanded, u_t.dtype)
        u_t_noisy = u_t.add(noise)
        
        h_eigen = self.pscan(op_decay, u_t_noisy)
        self.last_h_state = self.prop.basis.decode(h_eigen[:, -1, :]).detach()
        
        x_drift = self.prop.basis.decode(h_eigen).real.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        x_temporal = x_drift.add(x_spatial)
        
        delta_p = self.ffn(x_temporal.view(B * T, D, H, W))
        return x_temporal.add(delta_p.view(B, T, D, H, W))

    def forward_step(self, x_step, h_prev, dt_step):
        B, D, H, W = x_step.shape
        x_s = x_step.permute(0, 2, 3, 1)
        x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
        x_s = self._spatial_op(x_s)
        x_spatial = x_s.add(x_step)
        
        x_t = x_spatial.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
        x_t = self._complex_norm(x_t, self.norm_temporal)
        
        dt_t = torch.as_tensor(dt_step, device=x_step.device).view(-1, 1)
        x_encoded = self.prop.basis.encode(x_t)
        gate = F.silu(self.prop.input_gate(x_encoded.real))
        x_gated = x_encoded.mul(torch.complex(gate, torch.zeros_like(gate)))
        
        op_decay, op_forcing = self.prop.get_transition_operators(dt_t, x_gated)
        u_t = x_gated.add(self.prop._get_source_bias()).mul(op_forcing)
        
        noise = self.prop.generate_stochastic_term(u_t.shape, dt_t, u_t.dtype)
        u_t_noisy = u_t.add(noise)
        
        h_next = h_prev.mul(op_decay.squeeze(1)).add(u_t_noisy.squeeze(1))
        x_drift = self.prop.basis.decode(h_next).real.view(B, H, W, D).permute(0, 3, 1, 2)
        
        x_temporal = x_drift.add(x_spatial)
        delta_p = self.ffn(x_temporal)
        return x_temporal.add(delta_p), h_next

class UniPhyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, embed_dim=64, expand=4, num_experts=4, depth=4, patch_size=16, img_height=64, img_width=128, checkpointing=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = UniPhyEncoder(in_channels, embed_dim, patch_size, img_height, img_width)
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        feat_h, feat_w = (img_height + pad_h) // patch_size, (img_width + pad_w) // patch_size
        self.blocks = nn.ModuleList([UniPhyBlock(embed_dim, expand, num_experts, feat_h, feat_w) for _ in range(depth)])
        self.decoder = UniPhyEnsembleDecoder(out_channels, embed_dim, patch_size, img_height=img_height)
        self.fusion_weights = nn.Parameter(torch.ones(depth))
        self.ic_scale = nn.Parameter(torch.zeros(1))
        self.checkpointing = checkpointing

    def forward(self, x, dt):
        z_init = self.encoder(x)
        z_ic = z_init.clone()
        weights = F.softmax(self.fusion_weights, dim=0)
        
        block_outputs = []
        z_flow = z_init
        
        for i, block in enumerate(self.blocks):
            z_in = z_flow.clone().add(z_ic.mul(self.ic_scale))
            if self.training and self.checkpointing:
                z_flow = checkpoint.checkpoint(block, z_in, dt, use_reentrant=False)
            else:
                z_flow = block(z_in, dt)
            block_outputs.append(z_flow.mul(weights[i]))
            
        z_fused = torch.stack(block_outputs, dim=0).sum(dim=0)
        return self.decoder(z_fused, x)
        
    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        z_init = self.encoder(x_cond)
        z_ic = z_init.clone()
        states = []
        z_flow = z_init
        for block in self.blocks:
            z_in = z_flow.clone().add(z_ic.mul(self.ic_scale))
            z_flow = block(z_in, dt_cond)
            states.append(block.last_h_state.clone())
            
        z_curr = z_flow[:, -1]
        x_ref = x_cond[:, -1:]
        predictions = []
        
        for k in range(k_steps):
            dt_k = dt_future[:, k]
            z_step_flow = z_curr
            new_states = []
            step_outs = []
            
            for i, block in enumerate(self.blocks):
                z_in_step = z_step_flow.clone().add(z_ic[:, -1].mul(self.ic_scale))
                z_out, h_next = block.forward_step(z_in_step, states[i], dt_k)
                step_outs.append(z_out)
                new_states.append(h_next)
                z_step_flow = z_out
                
            states = new_states
            z_curr = z_step_flow
            weights = F.softmax(self.fusion_weights, dim=0)
            z_fused = torch.stack([out.mul(w) for w, out in zip(weights, step_outs)], dim=0).sum(dim=0)
            
            pred_pixel = self.decoder(z_fused.unsqueeze(1), x_ref)
            predictions.append(pred_pixel.squeeze(1))
            x_ref = pred_pixel
            
        return torch.stack(predictions, dim=1)
    