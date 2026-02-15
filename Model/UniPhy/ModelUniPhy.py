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
        self.prop = TemporalPropagator(
            dim,
            tau_ref_hours=tau_ref_hours,
            sde_mode=sde_mode,
            init_noise_scale=init_noise_scale,
            max_growth_rate=max_growth_rate,
        )
        self.ffn = UniPhyFeedForwardNetwork(dim, expand)

    def spatial_process_seq(self, x):
        bsz, t_len, dim, height, width = x.shape
        x_flat = x.reshape(bsz * t_len, dim, height, width)
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        y = x_flat + torch.complex(x_re, x_im)
        return y.reshape(bsz, t_len, dim, height, width)

    def spatial_process_step(self, x):
        bsz, dim, height, width = x.shape
        x_real = torch.cat([x.real, x.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        return x + torch.complex(x_re, x_im)

    def temporal_decode_seq(self, x):
        bsz, t_len, dim, height, width = x.shape
        x_flat = x.reshape(bsz * t_len, dim, height, width)
        y_flat = x_flat + self.ffn(x_flat)
        return y_flat.reshape(bsz, t_len, dim, height, width)

    def temporal_decode_step(self, x):
        return x + self.ffn(x)

    def forward(self, x, h_prev, dt_step, flux_prev):
        bsz, t_len, dim, height, width = x.shape
        x = self.spatial_process_seq(x)
        x_eigen = self.prop.basis.encode(x.permute(0, 1, 3, 4, 2))
        x_mean = x_eigen.mean(dim=(2, 3))
        if flux_prev is None:
            flux_prev = torch.zeros((bsz, dim), device=x.device, dtype=x.dtype)
        a_flux, u_flux = self.prop.flux_tracker.get_scan_operators(x_mean, dt_step)
        flux_seq = pscan(a_flux, u_flux)[..., 0]
        decay_flux_cum = torch.cumprod(a_flux[..., 0], dim=1)
        flux_seq = flux_seq + flux_prev.unsqueeze(1) * decay_flux_cum
        flux_out = flux_seq[:, -1]
        flux_forcing = torch.cat([flux_prev.unsqueeze(1), flux_seq[:, :-1]], dim=1)
        source_seq, gate_seq = self.prop.flux_tracker.compute_output_seq(flux_forcing)
        source_exp = source_seq.unsqueeze(2).unsqueeze(3).expand(bsz, t_len, height, width, dim)
        gate_exp = gate_seq.unsqueeze(2).unsqueeze(3).expand(bsz, t_len, height, width, dim)
        forcing = x_eigen * gate_exp + source_exp * (1.0 - gate_exp)
        op_decay_raw, op_forcing_raw = self.prop.get_transition_operators_seq(dt_step)
        decay_cum_raw = torch.cumprod(op_decay_raw, dim=1)
        op_decay = op_decay_raw.unsqueeze(2).unsqueeze(3).expand(bsz, t_len, height, width, dim)
        op_forcing = op_forcing_raw.unsqueeze(2).unsqueeze(3).expand(bsz, t_len, height, width, dim)
        u_t = forcing * op_forcing
        if self.prop.sde_mode == "sde":
            u_t = u_t + self.prop.generate_stochastic_term_seq(u_t.shape, dt_step, u_t.dtype)
        a = op_decay.permute(0, 2, 3, 1, 4).reshape(bsz * height * width, t_len, dim, 1)
        u = u_t.permute(0, 2, 3, 1, 4).reshape(bsz * height * width, t_len, dim, 1)
        y = pscan(a, u).reshape(bsz, height, width, t_len, dim).permute(0, 3, 1, 2, 4)
        if h_prev is not None:
            h0 = h_prev.reshape(bsz, height, width, dim)
            y = y + h0.unsqueeze(1) * decay_cum_raw.unsqueeze(2).unsqueeze(3)
        h_out = y[:, -1].reshape(bsz * height * width, 1, dim)
        y_dec = self.prop.basis.decode(y).permute(0, 1, 4, 2, 3)
        x_out = self.temporal_decode_seq(y_dec)
        return x + x_out, h_out, flux_out

    def forward_step(self, x_curr, h_prev, dt_step, flux_prev):
        bsz, dim, height, width = x_curr.shape
        x_curr = self.spatial_process_step(x_curr)
        x_eigen = self.prop.basis.encode(x_curr.permute(0, 2, 3, 1))
        x_mean = x_eigen.mean(dim=(1, 2))
        if flux_prev is None:
            flux_prev = torch.zeros((bsz, dim), device=x_curr.device, dtype=x_curr.dtype)
        source, gate = self.prop.flux_tracker.compute_output_step(flux_prev)
        source_exp = source.view(bsz, 1, 1, dim).expand(bsz, height, width, dim)
        gate_exp = gate.view(bsz, 1, 1, dim).expand(bsz, height, width, dim)
        forcing = x_eigen * gate_exp + source_exp * (1.0 - gate_exp)
        op_decay, op_forcing = self.prop.get_transition_operators_step(dt_step)
        u = forcing * op_forcing.view(bsz, 1, 1, dim)
        if self.prop.sde_mode == "sde":
            u = u + self.prop.generate_stochastic_term_step(u.shape, dt_step, u.dtype)
        if h_prev is not None:
            u = u + h_prev.reshape(bsz, height, width, dim) * op_decay.view(bsz, 1, 1, dim)
        decay_flux, forcing_flux = self.prop.flux_tracker.get_transition_operators_step(dt_step)
        flux_next = flux_prev * decay_flux + x_mean * forcing_flux
        h_out = u.reshape(bsz * height * width, 1, dim)
        x_out = self.temporal_decode_step(self.prop.basis.decode(u).permute(0, 3, 1, 2))
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
        tau_ref_hours=None,
        dt_ref=None,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
        ensemble_size=1,
        target_dt_hours=None,
        **kwargs,
    ):
        super().__init__()
        if tau_ref_hours is None:
            if dt_ref is None:
                tau_ref_hours = 6.0
            else:
                tau_ref_hours = float(dt_ref)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.tau_ref_hours = float(tau_ref_hours)
        if isinstance(patch_size, (tuple, list)):
            ph = int(patch_size[0])
            pw = int(patch_size[1])
        else:
            ph = int(patch_size)
            pw = int(patch_size)
        self.h_patches = (self.img_height + (ph - self.img_height % ph) % ph) // ph
        self.w_patches = (self.img_width + (pw - self.img_width % pw) % pw
                          ) // pw
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

    def forward(self, x_input, dt_step, member_idx=None):
        z = self.encoder(x_input)
        states = self._init_states()
        for idx, block in enumerate(self.blocks):
            z, h_f, f_f = block(z, states[idx][0], dt_step, states[idx][1])
            states[idx] = (h_f, f_f)
        return self.decoder(z, member_idx=member_idx)

    def forward_rollout(self, x_context, dt_context_step, dt_future_step, member_idx=None):
        bsz, t_ctx = x_context.shape[0], x_context.shape[1]
        if t_ctx >= 2:
            x_ctx_step = x_context[:, :-1]
            z = self.encoder(x_ctx_step)
            states = self._init_states()
            for idx, block in enumerate(self.blocks):
                z, h_f, f_f = block(z, states[idx][0], dt_context_step, states[idx][1])
                states[idx] = (h_f, f_f)
        else:
            states = self._init_states()
        z_curr = self.encoder(x_context[:, -1].unsqueeze(1))[:, 0]
        steps = int(dt_future_step.shape[1])
        preds = []
        for k in range(steps):
            dt_k = dt_future_step[:, k]
            new_states = []
            for idx, block in enumerate(self.blocks):
                z_curr, h_n, f_n = block.forward_step(z_curr, states[idx][0], dt_k, states[idx][1])
                new_states.append((h_n, f_n))
            states = new_states
            x_pred = self.decoder(z_curr.unsqueeze(1), member_idx=member_idx)[:, 0]
            preds.append(x_pred)
            z_curr = self.encoder(x_pred.unsqueeze(1))[:, 0]
        return torch.stack(preds, dim=1)
