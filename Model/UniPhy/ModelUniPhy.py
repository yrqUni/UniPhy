import torch
import torch.nn as nn
from PScan import pscan
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork


class SpatialGateModulator(nn.Module):
    def __init__(self, dim, h_patches, w_patches):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(
                dim, dim, kernel_size=3, padding=1,
                groups=dim, bias=False,
            ),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate_global, source_global, x_local):
        x_cat = torch.cat([x_local.real, x_local.imag], dim=-1)
        B, H, W, D2 = x_cat.shape
        spatial_feat = x_cat.permute(0, 3, 1, 2)
        spatial_gate = self.spatial_proj(spatial_feat)
        spatial_gate = spatial_gate.permute(0, 2, 3, 1)
        gate_combined = gate_global.unsqueeze(1).unsqueeze(2) * spatial_gate
        source_exp = source_global.unsqueeze(1).unsqueeze(2).expand(
            B, H, W, D2 // 2,
        )
        return x_local * gate_combined + source_exp * (1 - gate_combined)


class UniPhyBlock(nn.Module):
    def __init__(self, dim, expand, img_height, img_width, kernel_size,
                 dt_ref, init_noise_scale, sde_mode):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(
            dim * 2, dim * 2, kernel_size=kernel_size,
            padding=kernel_size // 2, img_height=img_height,
            img_width=img_width,
        )
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(
            dim, dt_ref=dt_ref, sde_mode=sde_mode,
            init_noise_scale=init_noise_scale,
        )
        self.ffn = UniPhyFeedForwardNetwork(dim, expand)
        self.spatial_gate = SpatialGateModulator(dim, img_height, img_width)

    def _apply_spatial(self, x_4d):
        x_real = torch.cat([x_4d.real, x_4d.imag], dim=1)
        x_norm = self.norm_spatial(
            x_real.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        return x_4d + torch.complex(x_re, x_im)

    def _apply_temporal_decode(self, x_4d):
        x_real = torch.cat([x_4d.real, x_4d.imag], dim=1)
        x_norm = self.norm_temporal(
            x_real.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)
        delta = self.ffn(x_complex)
        return x_4d + delta

    def forward(self, x, h_prev, dt_seq, flux_prev):
        B, T, D, H, W = x.shape

        x_flat = x.reshape(B * T, D, H, W)
        x_flat = self._apply_spatial(x_flat)
        x = x_flat.reshape(B, T, D, H, W)

        basis_W, basis_W_inv = self.prop.basis.get_matrix(torch.complex64)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode_with(x_perm, basis_W)
        x_mean = x_eigen.mean(dim=(2, 3))

        A_flux, X_flux = self.prop.flux_tracker.get_scan_operators(
            x_mean, dt_seq,
        )
        flux_seq = pscan(A_flux, X_flux).squeeze(-1)

        decay_seq = A_flux.squeeze(-1)
        decay_cum = torch.cumprod(decay_seq, dim=1)
        flux_seq = flux_seq + flux_prev.unsqueeze(1) * decay_cum

        source_seq, gate_seq = self.prop.flux_tracker.compute_output_seq(
            flux_seq,
        )
        flux_out = flux_seq[:, -1]

        forcing_list = []
        for t_idx in range(T):
            f_t = self.spatial_gate(
                gate_seq[:, t_idx],
                source_seq[:, t_idx],
                x_eigen[:, t_idx],
            )
            forcing_list.append(f_t)
        forcing = torch.stack(forcing_list, dim=1)

        op_decay, op_forcing = self.prop.get_transition_operators_seq(dt_seq)
        op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(
            B, T, H, W, D,
        )

        u_t = forcing * op_forcing

        if self.prop.sde_mode == "sde":
            u_t = u_t + self.prop.generate_stochastic_term_seq(
                u_t.shape, dt_seq, u_t.dtype, x_eigen,
            )

        h_prev_hw = h_prev.reshape(B, H, W, D)
        h_contrib_t0 = h_prev_hw * op_decay[:, 0]
        u_t_0 = u_t[:, 0] + h_contrib_t0
        u_t = torch.cat([u_t_0.unsqueeze(1), u_t[:, 1:]], dim=1)

        A = op_decay.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        u_out = pscan(A, X).reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        h_out = u_out[:, -1].reshape(B * H * W, 1, D)

        decoded = self.prop.basis.decode_with(
            u_out, basis_W_inv,
        ).permute(0, 1, 4, 2, 3)
        decoded_flat = decoded.reshape(B * T, D, H, W)
        x_out_flat = self._apply_temporal_decode(decoded_flat)
        x_out = x_out_flat.reshape(B, T, D, H, W)

        return x + x_out, h_out, flux_out

    def forward_step(self, x_curr, h_prev, dt_step, flux_prev):
        B, D, H, W = x_curr.shape

        x_curr = self._apply_spatial(x_curr)

        basis_W, basis_W_inv = self.prop.basis.get_matrix(torch.complex64)

        x_eigen = self.prop.basis.encode_with(
            x_curr.permute(0, 2, 3, 1), basis_W,
        )
        x_mean = x_eigen.mean(dim=(1, 2))

        flux_next, source, gate = self.prop.flux_tracker.forward_step(
            flux_prev, x_mean, dt_step,
        )

        forcing = self.spatial_gate(gate, source, x_eigen)

        op_decay, op_forcing = self.prop.get_transition_operators_step(
            dt_step,
        )
        op_decay = op_decay.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        op_forcing = op_forcing.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)

        h_prev_hw = h_prev.reshape(B, H, W, D)

        noise = torch.zeros_like(h_prev_hw)
        if self.prop.sde_mode == "sde":
            noise = self.prop.generate_stochastic_term_step(
                h_prev_hw.shape, dt_step, h_prev_hw.dtype, x_eigen,
            )

        h_next = h_prev_hw * op_decay + forcing * op_forcing + noise
        x_out = self.prop.basis.decode_with(
            h_next, basis_W_inv,
        ).permute(0, 3, 1, 2)

        x_out = self._apply_temporal_decode(x_out)

        z_next = x_curr + x_out
        return z_next, h_next.reshape(B * H * W, 1, D), flux_next


class UniPhyModel(nn.Module):
    def __init__(
        self, in_channels, out_channels, embed_dim, expand=4, depth=8,
        patch_size=16, img_height=64, img_width=64, dt_ref=1.0,
        sde_mode="sde", init_noise_scale=0.01, ensemble_size=10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        if isinstance(patch_size, (tuple, list)):
            ph, pw = patch_size
        else:
            ph = pw = patch_size
        self.h_patches = (
            (img_height + (ph - img_height % ph) % ph) // ph
        )
        self.w_patches = (
            (img_width + (pw - img_width % pw) % pw) // pw
        )
        self.encoder = UniPhyEncoder(
            in_ch=in_channels, embed_dim=embed_dim, patch_size=patch_size,
            img_height=img_height, img_width=img_width,
        )
        self.decoder = UniPhyEnsembleDecoder(
            out_ch=out_channels, latent_dim=embed_dim, patch_size=patch_size,
            model_channels=embed_dim, ensemble_size=ensemble_size,
            img_height=img_height, img_width=img_width,
        )
        self.blocks = nn.ModuleList([
            UniPhyBlock(
                dim=embed_dim, expand=expand, img_height=self.h_patches,
                img_width=self.w_patches, kernel_size=3, dt_ref=dt_ref,
                init_noise_scale=init_noise_scale, sde_mode=sde_mode,
            )
            for _ in range(depth)
        ])
        self._init_encoder_decoder_weights()

    def _init_encoder_decoder_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.blocks.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_states(self, B, device, dtype):
        H, W, D = self.h_patches, self.w_patches, self.embed_dim
        return [
            (
                torch.zeros(B * H * W, 1, D, device=device, dtype=dtype),
                torch.zeros(B, D, device=device, dtype=dtype),
            )
            for _ in range(self.depth)
        ]

    def _normalize_dt(self, dt, B, T, device):
        if isinstance(dt, (float, int)):
            return torch.full((B, T), float(dt), device=device)
        if dt.ndim == 0:
            return dt.expand(B, T).contiguous()
        if dt.ndim == 1:
            if dt.shape[0] == B:
                return dt.unsqueeze(1).expand(B, T).contiguous()
            return dt.unsqueeze(0).expand(B, T).contiguous()
        return dt

    def forward(self, x, dt, member_idx=None, return_latent=False):
        B, T = x.shape[0], x.shape[1]
        dt_seq = self._normalize_dt(dt, B, T, x.device)
        z = self.encoder(x)
        dtype = z.dtype if z.is_complex() else torch.complex64
        states = self._init_states(B, x.device, dtype)
        for i, block in enumerate(self.blocks):
            z, h_next, flux_next = block(
                z, states[i][0], dt_seq, states[i][1],
            )
            states[i] = (h_next, flux_next)
        out = self.decoder(z, member_idx=member_idx)
        if return_latent:
            return out, z
        return out

    def forward_rollout(self, x_context, dt_context, dt_list):
        B, T_in = x_context.shape[0], x_context.shape[1]
        device = x_context.device
        dt_ctx_seq = self._normalize_dt(dt_context, B, T_in, device)
        z_ctx = self.encoder(x_context)
        dtype = z_ctx.dtype if z_ctx.is_complex() else torch.complex64
        states = self._init_states(B, device, dtype)
        for i, block in enumerate(self.blocks):
            z_ctx, h_f, f_f = block(
                z_ctx, states[i][0], dt_ctx_seq, states[i][1],
            )
            states[i] = (h_f, f_f)
        z_curr = z_ctx[:, -1]
        n_steps = len(dt_list)
        preds = []
        for step_idx, dt_k in enumerate(dt_list):
            dt_step = self._normalize_dt(dt_k, B, 1, device).squeeze(1)
            new_states = []
            for i, block in enumerate(self.blocks):
                z_curr, h_n, f_n = block.forward_step(
                    z_curr, states[i][0], dt_step, states[i][1],
                )
                new_states.append((h_n, f_n))
            states = new_states
            x_pred = self.decoder(z_curr)
            preds.append(x_pred)
            if step_idx < n_steps - 1:
                z_curr = self.encoder(x_pred)
        return torch.stack(preds, dim=1)
