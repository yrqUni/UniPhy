import torch
import torch.nn as nn
from PScan import pscan
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork

class UniPhyBlock(nn.Module):
    def __init__(self, dim, expand, num_experts, img_height, img_width, kernel_size=3, dt_ref=1.0, init_noise_scale=0.01, sde_mode="sde", max_growth_rate=0.3):
        super().__init__()
        self.dim = dim
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, img_height=img_height, img_width=img_width)
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(dim, dt_ref=dt_ref, sde_mode=sde_mode, init_noise_scale=init_noise_scale, max_growth_rate=max_growth_rate)
        self.ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts)

    def _spatial_process(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, D, H, W = x.shape
            x_flat = x.reshape(B * T, D, H, W)
        else:
            x_flat = x
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        x_out = x_flat + torch.complex(x_re, x_im)
        return x_out.reshape(B, T, D, H, W) if is_5d else x_out

    def _temporal_decode(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, D, H, W = x.shape
            x_flat = x.reshape(B * T, D, H, W)
        else:
            x_flat = x
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_temporal(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)
        delta = self.ffn(x_complex)
        x_out = x_complex + delta
        return x_out.reshape(B, T, D, H, W) if is_5d else x_out

    def forward(self, x, h_prev, dt, flux_prev):
        B, T, D, H, W = x.shape
        x = self._spatial_process(x)
        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        A_flux, X_flux = self.prop.flux_tracker.get_scan_operators(
            x_mean, dt, self.prop.dt_ref
        )
        flux_seq = pscan(A_flux, X_flux).squeeze(-1)
        
        decay_seq = A_flux.squeeze(-1)
        decay_cum = torch.cumprod(decay_seq, dim=1)
        flux_seq = flux_seq + (flux_prev.unsqueeze(1) * decay_cum)
        
        source_seq, gate_seq = self.prop.flux_tracker.compute_output(flux_seq)
        flux_out = flux_seq[:, -1]

        source_exp = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        gate_exp = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        forcing = x_eigen * gate_exp + source_exp * (1 - gate_exp)

        dt_exp = dt.unsqueeze(0).expand(B, T) if dt.ndim == 1 else dt
        op_decay, op_forcing = self.prop.get_transition_operators(dt_exp)

        op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        u_t = forcing * op_forcing

        if self.prop.sde_mode == "sde":
            dt_noise = dt_exp.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            u_t = u_t + self.prop.generate_stochastic_term(
                u_t.shape, dt_noise, u_t.dtype, x_eigen
            )

        if h_prev is not None:
            h_contrib_t0 = h_prev.reshape(B, H, W, D) * op_decay[:, 0]
            u_t_list = list(torch.unbind(u_t, dim=1))
            u_t_list[0] = u_t_list[0] + h_contrib_t0
            u_t = torch.stack(u_t_list, dim=1)

        A = op_decay.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        u_out = pscan(A, X).reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        h_out = u_out[:, -1].reshape(B * H * W, 1, D)
        x_out = self._temporal_decode(
            self.prop.basis.decode(u_out).permute(0, 1, 4, 2, 3)
        )

        return x + x_out, h_out, flux_out

    def forward_step(self, x_curr, h_prev, dt, flux_prev):
        B, D, H, W = x_curr.shape
        x_real = torch.cat([x_curr.real, x_curr.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_s_re, x_s_im = torch.chunk(self.spatial_cliff(x_norm), 2, dim=1)
        x_curr = x_curr + torch.complex(x_s_re, x_s_im)

        x_eigen = self.prop.basis.encode(x_curr.permute(0, 2, 3, 1))
        x_mean = x_eigen.mean(dim=(1, 2))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=x_curr.device, dtype=x_curr.dtype)

        flux_next, source, gate = self.prop.flux_tracker.forward_step(
            flux_prev, x_mean, dt, self.prop.dt_ref
        )

        source_exp = source.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        gate_exp = gate.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        forcing = x_eigen * gate_exp + source_exp * (1 - gate_exp)

        op_decay, op_forcing = self.prop.get_transition_operators(dt)
        if op_decay.ndim == 2:
            op_decay = op_decay.unsqueeze(1).unsqueeze(1)
        if op_forcing.ndim == 2:
            op_forcing = op_forcing.unsqueeze(1).unsqueeze(1)

        h_prev_reshaped = h_prev.reshape(B, H, W, D)
        noise = 0
        if self.prop.sde_mode == "sde":
            dt_noise = dt.view(B, 1, 1, 1) if dt.ndim >= 1 else dt
            noise = self.prop.generate_stochastic_term(
                h_prev_reshaped.shape, dt_noise, h_prev_reshaped.dtype, x_eigen
            )

        h_next = h_prev_reshaped * op_decay + forcing * op_forcing + noise
        x_out = self.prop.basis.decode(h_next).permute(0, 3, 1, 2)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)

        delta = self.ffn(torch.complex(x_out_re, x_out_im))
        z_next = x_curr + torch.complex(x_out_re, x_out_im) + delta

        return z_next, h_next.reshape(B * H * W, 1, D), flux_next


class UniPhyModel(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, expand=4, num_experts=8, depth=8, patch_size=16, img_height=64, img_width=64, dt_ref=1.0, sde_mode="sde", init_noise_scale=0.01, max_growth_rate=0.3):
        super().__init__()
        self.embed_dim, self.depth, self.img_height, self.img_width = embed_dim, depth, img_height, img_width
        self.h_patches = (img_height + (patch_size - img_height % patch_size) % patch_size) // patch_size
        self.w_patches = (img_width + (patch_size - img_width % patch_size) % patch_size) // patch_size
        self.encoder = UniPhyEncoder(in_ch=in_channels, embed_dim=embed_dim, patch_size=patch_size, img_height=img_height, img_width=img_width)
        self.decoder = UniPhyEnsembleDecoder(out_ch=out_channels, latent_dim=embed_dim, patch_size=patch_size, model_channels=embed_dim, ensemble_size=10, img_height=img_height, img_width=img_width)
        self.blocks = nn.ModuleList([UniPhyBlock(dim=embed_dim, expand=expand, num_experts=num_experts, img_height=self.h_patches, img_width=self.w_patches, dt_ref=dt_ref, sde_mode=sde_mode, init_noise_scale=init_noise_scale, max_growth_rate=max_growth_rate) for _ in range(depth)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _init_states(self, B, device, dtype):
        return [(torch.zeros(B * self.h_patches * self.w_patches, 1, self.embed_dim, device=device, dtype=dtype), torch.zeros(B, self.embed_dim, device=device, dtype=dtype)) for _ in range(self.depth)]

    def forward(self, x, dt, member_idx=None):
        z = self.encoder(x)
        states = self._init_states(x.shape[0], x.device, z.dtype if z.dtype.is_complex else torch.complex64)
        for i, block in enumerate(self.blocks):
            z, h_next, flux_next = block(z, states[i][0], dt, states[i][1])
            states[i] = (h_next, flux_next)
        out = self.decoder(z, member_idx=member_idx)
        return out[..., :self.img_height, :self.img_width]

    @torch.no_grad()
    def forward_rollout(self, x_context, dt_context, dt_list):
        B, T_in = x_context.shape[0], x_context.shape[1]
        z_ctx = self.encoder(x_context)
        states = self._init_states(B, x_context.device, z_ctx.dtype if z_ctx.dtype.is_complex else torch.complex64)
        
        if isinstance(dt_context, (float, int)):
             dt_ctx_tensor = torch.full((B, T_in), float(dt_context), device=x_context.device)
        elif dt_context.ndim == 0:
             dt_ctx_tensor = dt_context.expand(B, T_in)
        else:
             dt_ctx_tensor = dt_context

        for i, block in enumerate(self.blocks):
            z_ctx, h_f, f_f = block(z_ctx, states[i][0], dt_ctx_tensor, states[i][1])
            states[i] = (h_f, f_f)
            
        z_curr = z_ctx[:, -1]
        preds = []
        for dt_k in dt_list:
            new_states = []
            for i, block in enumerate(self.blocks):
                z_curr, h_n, f_n = block.forward_step(z_curr, states[i][0], dt_k, states[i][1])
                new_states.append((h_n, f_n))
            states = new_states
            pred = self.decoder(z_curr.unsqueeze(1)).squeeze(1)
            preds.append(pred[..., :self.img_height, :self.img_width])
        return torch.stack(preds, dim=1)
    