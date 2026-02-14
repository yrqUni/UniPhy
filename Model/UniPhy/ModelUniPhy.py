import torch
import torch.nn as nn

from PScan import pscan
from UniPhyFFN import UniPhyFeedForwardNetwork
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import RiemannianCliffordConv2d, TemporalPropagator


class UniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand,
        img_height,
        img_width,
        kernel_size=3,
        tau_ref_hours=1.0,
        init_noise_scale=0.01,
        sde_mode="sde",
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.spatial_cliff = RiemannianCliffordConv2d(
            dim * 2,
            dim * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            img_height=img_height,
            img_width=img_width,
        )
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(
            dim,
            tau_ref_hours=tau_ref_hours,
            sde_mode=sde_mode,
            init_noise_scale=init_noise_scale,
            max_growth_rate=max_growth_rate,
        )
        self.ffn = UniPhyFeedForwardNetwork(dim, expand)

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
            bsz, t_len, dim, hgt, wdt = x.shape
            x_flat = x.reshape(bsz * t_len, dim, hgt, wdt)
        else:
            x_flat = x

        x_out = x_flat + self.ffn(x_flat)

        if is_5d:
            return x_out.reshape(bsz, t_len, dim, hgt, wdt)
        return x_out


    def forward(self, x, h_prev, dt, flux_prev):
        B, T, D, H, W = x.shape
        x = self._spatial_process(x)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        A_flux, X_flux = self.prop.flux_tracker.get_scan_operators(x_mean, dt)
        flux_seq = pscan(A_flux, X_flux).squeeze(-1)
        decay_seq = A_flux.squeeze(-1)
        decay_cum = torch.cumprod(decay_seq, dim=1)
        flux_seq = flux_seq + flux_prev.unsqueeze(1) * decay_cum

        source_seq, gate_seq = self.prop.flux_tracker.compute_output(flux_seq)
        flux_out = flux_seq[:, -1]

        source_exp = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        gate_exp = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        forcing = x_eigen * gate_exp + source_exp * (1.0 - gate_exp)

        op_decay, op_forcing = self.prop.get_transition_operators(dt)
        op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        u_t = forcing * op_forcing
        if self.prop.sde_mode == "sde":
            u_t = u_t + self.prop.generate_stochastic_term(u_t.shape, dt, u_t.dtype)

        if h_prev is not None:
            h_contrib_t0 = h_prev.reshape(B, H, W, D) * op_decay[:, 0]
            u0 = u_t[:, 0] + h_contrib_t0
            u_t = torch.cat([u0.unsqueeze(1), u_t[:, 1:]], dim=1)

        A = op_decay.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        u_out = pscan(A, X).reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        h_out = u_out[:, -1].reshape(B * H * W, 1, D)
        x_out = self._temporal_decode(self.prop.basis.decode(u_out).permute(0, 1, 4, 2, 3))
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

        decay_flux, forcing_flux = self.prop.flux_tracker.get_transition_operators(dt)
        if decay_flux.ndim == 1:
            decay_flux = decay_flux.unsqueeze(0).expand(B, D)
            forcing_flux = forcing_flux.unsqueeze(0).expand(B, D)

        flux_next = flux_prev * decay_flux + x_mean * forcing_flux
        source, gate = self.prop.flux_tracker.compute_output(flux_next)

        source_exp = source.reshape(B, 1, 1, D).expand(B, H, W, D)
        gate_exp = gate.reshape(B, 1, 1, D).expand(B, H, W, D)
        forcing = x_eigen * gate_exp + source_exp * (1.0 - gate_exp)

        op_decay, op_forcing = self.prop.get_transition_operators(dt)
        if op_decay.ndim == 1:
            op_decay = op_decay.unsqueeze(0).expand(B, D)
            op_forcing = op_forcing.unsqueeze(0).expand(B, D)

        u = forcing * op_forcing.reshape(B, 1, 1, D)
        if self.prop.sde_mode == "sde":
            u = u + self.prop.generate_stochastic_term(u.shape, dt, u.dtype)

        if h_prev is not None:
            u = u + h_prev.reshape(B, H, W, D) * op_decay.reshape(B, 1, 1, D)

        h_out = u.reshape(B * H * W, 1, D)
        x_out = self._temporal_decode(self.prop.basis.decode(u).permute(0, 3, 1, 2))
        return x_curr + x_out, h_out, flux_next


class UniPhyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        expand=4,
        depth=8,
        patch_size=16,
        img_height=64,
        img_width=64,
        tau_ref_hours=6.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
        ensemble_size=1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.tau_ref_hours = float(tau_ref_hours)

        if isinstance(patch_size, (tuple, list)):
            ph, pw = int(patch_size[0]), int(patch_size[1])
        else:
            ph = pw = int(patch_size)

        self.h_patches = (self.img_height + (ph - self.img_height % ph) % ph) // ph
        self.w_patches = (self.img_width + (pw - self.img_width % pw) % pw) // pw

        self.encoder = UniPhyEncoder(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )
        self.decoder = UniPhyEnsembleDecoder(
            out_ch=out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
            ensemble_size=ensemble_size,
        )
        self.blocks = nn.ModuleList(
            [
                UniPhyBlock(
                    dim=embed_dim,
                    expand=expand,
                    img_height=self.h_patches,
                    img_width=self.w_patches,
                    kernel_size=3,
                    tau_ref_hours=self.tau_ref_hours,
                    init_noise_scale=init_noise_scale,
                    sde_mode=sde_mode,
                    max_growth_rate=max_growth_rate,
                )
                for _ in range(self.depth)
            ]
        )

    def _init_states(self):
        return [(None, None) for _ in range(self.depth)]

    def forward(self, x, dt, member_idx=None):
        B, T, _, _, _ = x.shape
        z = self.encoder(x)
        states = self._init_states()
        for i, block in enumerate(self.blocks):
            z, h_f, f_f = block(z, states[i][0], dt, states[i][1])
            states[i] = (h_f, f_f)
        return self.decoder(z, member_idx=member_idx)

    def forward_rollout(self, x_context, dt_context, dt_future, member_idx=None):
        B, t_in = x_context.shape[0], x_context.shape[1]
        z_ctx = self.encoder(x_context)
        states = self._init_states()

        for i, block in enumerate(self.blocks):
            z_ctx, h_f, f_f = block(z_ctx, states[i][0], dt_context, states[i][1])
            states[i] = (h_f, f_f)

        x_last = x_context[:, -1]
        z_curr = self.encoder(x_last)

        if isinstance(dt_future, list):
            dt_future = torch.stack(
                [
                    d.float().to(x_context.device)
                    if isinstance(d, torch.Tensor)
                    else torch.tensor(float(d), device=x_context.device)
                    for d in dt_future
                ]
            )

        if not isinstance(dt_future, torch.Tensor):
            dt_future = torch.tensor(dt_future, device=x_context.device, dtype=torch.float32)

        steps = int(dt_future.shape[0]) if dt_future.ndim == 1 else int(dt_future.shape[1])

        preds = []
        for k in range(steps):
            dt_k = dt_future[k] if dt_future.ndim == 1 else dt_future[:, k]
            new_states = []
            for i, block in enumerate(self.blocks):
                z_curr, h_n, f_n = block.forward_step(z_curr, states[i][0], dt_k, states[i][1])
                new_states.append((h_n, f_n))
            states = new_states
            x_pred = self.decoder(z_curr, member_idx=member_idx)
            preds.append(x_pred)
            z_curr = self.encoder(x_pred)
        return torch.stack(preds, dim=1)
