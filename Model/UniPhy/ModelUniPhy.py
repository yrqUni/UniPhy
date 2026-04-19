import torch
import torch.nn as nn

from .PScan import pscan
from .UniPhyFFN import UniPhyFeedForwardNetwork
from .UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from .UniPhyOps import (
    GlobalFluxTracker,
    MultiScaleSpatialMixer,
    TemporalPropagator,
    complex_dtype_for,
)


def _dt_is_zero(dt, tol=1e-12):
    dt_real = dt.real if torch.is_complex(dt) else dt
    return dt_real.abs() <= tol


def _dt_has_negative(dt, tol=1e-12):
    dt_real = dt.real if torch.is_complex(dt) else dt
    return bool((dt_real < -tol).any().item())


def _expand_batch_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


class SpatialGateModulator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def _forward_single(self, gate_global, source_global, x_local):
        x_cat = torch.cat([x_local.real, x_local.imag], dim=-1)
        batch_size, height, width, doubled_dim = x_cat.shape
        dim = doubled_dim // 2
        src_re = source_global.real.unsqueeze(1).unsqueeze(2).expand(
            batch_size,
            height,
            width,
            dim,
        )
        src_im = source_global.imag.unsqueeze(1).unsqueeze(2).expand(
            batch_size,
            height,
            width,
            dim,
        )
        source_field = torch.complex(src_re, src_im)
        spatial_feat = x_cat.permute(0, 3, 1, 2)
        spatial_gate = self.spatial_proj(spatial_feat).permute(0, 2, 3, 1)
        gate_combined = gate_global.unsqueeze(1).unsqueeze(2) * spatial_gate
        return x_local * gate_combined + source_field * (1 - gate_combined)

    def forward(self, gate_global, source_global, x_local):
        if x_local.ndim == 5:
            batch_size, steps, height, width, dim = x_local.shape
            x_flat = x_local.reshape(batch_size * steps, height, width, dim)
            gate_flat = gate_global.reshape(batch_size * steps, -1)
            source_flat = source_global.reshape(batch_size * steps, -1)
            out = self._forward_single(gate_flat, source_flat, x_flat)
            return out.reshape(batch_size, steps, height, width, dim)
        return self._forward_single(gate_global, source_global, x_local)


class FluxSpatialPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stat_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.SiLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        squeeze_time = x.ndim == 4
        if squeeze_time:
            x = x.unsqueeze(1)
        mean_pool = x.mean(dim=(2, 3))
        magnitude = x.abs()
        mean_mag = magnitude.mean(dim=(2, 3))
        std_mag = magnitude.std(dim=(2, 3), correction=0)
        stats = torch.stack([mean_mag, std_mag], dim=-1)
        scale = torch.tanh(self.stat_net(stats).squeeze(-1)).to(x.real.dtype)
        pooled = mean_pool * (1.0 + scale)
        return pooled.squeeze(1) if squeeze_time else pooled


class UniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand,
        img_height,
        img_width,
        kernel_size,
        dt_ref,
        init_noise_scale,
    ):
        super().__init__()
        del img_height, img_width, kernel_size
        self.spatial_mixer = MultiScaleSpatialMixer(dim)
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.prop = TemporalPropagator(
            dim,
            dt_ref=dt_ref,
            init_noise_scale=init_noise_scale,
        )
        self.ffn = UniPhyFeedForwardNetwork(dim, expand)
        self.spatial_gate = SpatialGateModulator(dim)
        self.flux_pool = FluxSpatialPool(dim)

    def _combine_output(self, input_state, decoded_state, decoded_with_residual):
        del input_state, decoded_state
        return decoded_with_residual

    def _decode_state(self, h_state, basis_w_inv):
        return self.prop.basis.decode_with(h_state, basis_w_inv).permute(0, 3, 1, 2)

    def _decode_sequence(self, h_seq, basis_w_inv):
        return self.prop.basis.decode_with(h_seq, basis_w_inv).permute(0, 1, 4, 2, 3)

    def _apply_spatial(self, x_4d):
        return self.spatial_mixer(x_4d)

    def _apply_temporal_decode(self, x_4d, source=None, gate=None):
        del source, gate
        x_real = torch.cat([x_4d.real, x_4d.imag], dim=1)
        x_norm = self.norm_temporal(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)
        delta = self.ffn(x_complex)
        return x_4d + delta

    def _normalize_block_noise_seq(
        self, noise_seq, batch_size, steps, dim, height, width
    ):
        if noise_seq is None:
            return None
        if tuple(noise_seq.shape) == (batch_size, steps, height, width, dim):
            return noise_seq
        if tuple(noise_seq.shape) == (batch_size, steps, dim, height, width):
            return noise_seq.permute(0, 1, 3, 4, 2).contiguous()
        raise ValueError(
            "Noise shape mismatch: expected "
            f"{(batch_size, steps, height, width, dim)} or "
            f"{(batch_size, steps, dim, height, width)}, got {tuple(noise_seq.shape)}"
        )

    def _normalize_block_noise_step(self, noise_step, batch_size, dim, height, width):
        if noise_step is None:
            return None
        if tuple(noise_step.shape) == (batch_size, height, width, dim):
            return noise_step
        if tuple(noise_step.shape) == (batch_size, dim, height, width):
            return noise_step.permute(0, 2, 3, 1).contiguous()
        raise ValueError(
            "Noise shape mismatch: expected "
            f"{(batch_size, height, width, dim)} or "
            f"{(batch_size, dim, height, width)}, got {tuple(noise_step.shape)}"
        )

    def forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
        batch_size, steps, dim, height, width = x.shape

        x_flat = x.reshape(batch_size * steps, dim, height, width)
        x_flat = self._apply_spatial(x_flat)
        x = x_flat.reshape(batch_size, steps, dim, height, width)

        basis_dtype = complex_dtype_for(x.dtype)
        basis_w, basis_w_inv = self.prop.get_basis_matrices(basis_dtype)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode_with(x_perm, basis_w)
        x_mean = self.flux_pool(x_eigen)

        a_flux, x_flux = self.prop.flux_tracker.get_scan_operators(x_mean, dt_seq)
        flux_seq = pscan(a_flux, x_flux).squeeze(-1)
        decay_seq = a_flux.squeeze(-1)
        decay_cum = torch.cumprod(decay_seq, dim=1)
        flux_seq = flux_seq + flux_prev.unsqueeze(1) * decay_cum
        source_seq, gate_seq = self.prop.flux_tracker.compute_output_seq(flux_seq)
        flux_out = flux_seq[:, -1]

        forcing = self.spatial_gate(gate_seq, source_seq, x_eigen)

        op_decay, op_forcing = self.prop.get_transition_operators_seq(dt_seq)
        op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(
            batch_size, steps, height, width, dim
        )
        op_forcing = (
            op_forcing.unsqueeze(2)
            .unsqueeze(3)
            .expand(batch_size, steps, height, width, dim)
        )

        u_t = forcing * op_forcing
        noise_seq = self._normalize_block_noise_seq(
            noise_seq, batch_size, steps, dim, height, width
        )
        u_t = u_t + self.prop.generate_stochastic_term_seq(
            u_t.shape,
            dt_seq,
            u_t.dtype,
            x_eigen,
            noise_seq=noise_seq,
        )

        h_prev_hw = h_prev.reshape(batch_size, height, width, dim)
        h_contrib_t0 = h_prev_hw * op_decay[:, 0]
        u_t0 = u_t[:, 0] + h_contrib_t0
        u_t = torch.cat([u_t0.unsqueeze(1), u_t[:, 1:]], dim=1)

        a_scan = op_decay.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width, steps, dim, 1
        )
        x_scan = u_t.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width, steps, dim, 1
        )
        u_out = pscan(a_scan, x_scan).reshape(
            batch_size, height, width, steps, dim
        ).permute(0, 3, 1, 2, 4)

        h_out = u_out[:, -1].reshape(batch_size * height * width, 1, dim)
        decoded = self._decode_sequence(u_out, basis_w_inv)
        decoded_flat = decoded.reshape(batch_size * steps, dim, height, width)
        decoded_with_residual_flat = self._apply_temporal_decode(decoded_flat)
        decoded_with_residual = decoded_with_residual_flat.reshape(
            batch_size, steps, dim, height, width
        )
        combined = self._combine_output(x, decoded, decoded_with_residual)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x, combined)

        return combined, h_out, flux_out

    def forward_step(self, x_curr, h_prev, dt_step, flux_prev, noise_step=None):
        batch_size, dim, height, width = x_curr.shape

        x_curr = self._apply_spatial(x_curr)

        basis_dtype = complex_dtype_for(x_curr.dtype)
        basis_w, basis_w_inv = self.prop.get_basis_matrices(basis_dtype)

        x_eigen = self.prop.basis.encode_with(x_curr.permute(0, 2, 3, 1), basis_w)
        x_mean = self.flux_pool(x_eigen)
        flux_next, source, gate = self.prop.flux_tracker.forward_step(
            flux_prev,
            x_mean,
            dt_step,
        )

        forcing = self.spatial_gate(gate, source, x_eigen)

        op_decay, op_forcing = self.prop.get_transition_operators_step(dt_step)
        op_decay = op_decay.unsqueeze(1).unsqueeze(2).expand(
            batch_size, height, width, dim
        )
        op_forcing = op_forcing.unsqueeze(1).unsqueeze(2).expand(
            batch_size, height, width, dim
        )

        h_prev_hw = h_prev.reshape(batch_size, height, width, dim)
        noise_step = self._normalize_block_noise_step(
            noise_step, batch_size, dim, height, width
        )
        noise = self.prop.generate_stochastic_term_step(
            h_prev_hw.shape,
            dt_step,
            h_prev_hw.dtype,
            x_eigen,
            noise_step=noise_step,
        )

        h_next = h_prev_hw * op_decay + forcing * op_forcing + noise
        decoded = self._decode_state(h_next, basis_w_inv)
        decoded_with_residual = self._apply_temporal_decode(decoded)
        z_next = self._combine_output(x_curr, decoded, decoded_with_residual)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim)
        z_next = torch.where(zero_mask, x_curr, z_next)
        return z_next, h_next.reshape(batch_size * height * width, 1, dim), flux_next


class UniPhyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        expand,
        depth,
        patch_size,
        img_height,
        img_width,
        dt_ref,
        init_noise_scale,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        if isinstance(patch_size, (tuple, list)):
            ph, pw = patch_size
        else:
            ph = pw = patch_size
        self.h_patches = (img_height + (ph - img_height % ph) % ph) // ph
        self.w_patches = (img_width + (pw - img_width % pw) % pw) // pw
        self.encoder = UniPhyEncoder(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )
        self.decoder = UniPhyEnsembleDecoder(
            out_ch=out_channels,
            latent_dim=embed_dim,
            patch_size=patch_size,
            model_channels=embed_dim,
            img_height=img_height,
            img_width=img_width,
        )
        self.skip_context_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.skip_spatial_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )
        nn.init.zeros_(self.skip_spatial_proj[-1].weight)
        nn.init.zeros_(self.skip_spatial_proj[-1].bias)
        self.blocks = nn.ModuleList(
            [
                UniPhyBlock(
                    dim=embed_dim,
                    expand=expand,
                    img_height=self.h_patches,
                    img_width=self.w_patches,
                    kernel_size=3,
                    dt_ref=dt_ref,
                    init_noise_scale=init_noise_scale,
                )
                for _ in range(depth)
            ]
        )
        self._init_encoder_decoder_weights()

    def _init_encoder_decoder_weights(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                nonlinearity = "linear" if "stem" in name else "relu"
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity=nonlinearity,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.blocks.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_states(self, batch_size, device, dtype):
        height, width, dim = self.h_patches, self.w_patches, self.embed_dim
        states = []
        for block in self.blocks:
            h0 = block.prop.get_initial_h(batch_size, height, width, device, dtype)
            h0_flat = h0.reshape(batch_size * height * width, 1, dim)
            f0 = block.prop.flux_tracker.get_initial_state(batch_size, device, dtype)
            states.append((h0_flat, f0))
        return states

    def _normalize_dt(self, dt, batch_size, steps, device):
        dtype = self.skip_spatial_proj[0].weight.dtype
        if isinstance(dt, (float, int)):
            return torch.full(
                (batch_size, steps),
                float(dt),
                device=device,
                dtype=dtype,
            )
        dt = dt.detach().to(device=device, dtype=dtype)
        if dt.ndim == 0:
            return dt.expand(batch_size, steps).contiguous()
        if dt.ndim == 1:
            if dt.shape[0] == batch_size:
                return dt.unsqueeze(1).expand(batch_size, steps).contiguous()
            return dt.unsqueeze(0).expand(batch_size, steps).contiguous()
        return dt

    def _noise_shape_from_latent(self, latent):
        if latent.ndim in {4, 5}:
            return latent.shape
        raise ValueError(f"Unsupported latent rank for noise shape: {latent.ndim}")

    def _normalize_noise(self, latent, noise, allow_missing=False):
        del allow_missing
        expected_shape = self._noise_shape_from_latent(latent)
        if noise is None:
            if latent.ndim == 5:
                return torch.zeros(
                    (
                        latent.shape[0],
                        latent.shape[1],
                        latent.shape[3],
                        latent.shape[4],
                        latent.shape[2],
                    ),
                    device=latent.device,
                    dtype=latent.dtype,
                )
            return torch.zeros(
                (latent.shape[0], latent.shape[2], latent.shape[3], latent.shape[1]),
                device=latent.device,
                dtype=latent.dtype,
            )
        if tuple(noise.shape) != tuple(expected_shape):
            raise ValueError(
                "Noise shape mismatch: expected "
                f"{tuple(expected_shape)}, got {tuple(noise.shape)}"
            )
        if latent.ndim == 5:
            return noise.permute(0, 1, 3, 4, 2).contiguous()
        return noise.permute(0, 2, 3, 1).contiguous()

    def _normalize_rollout_noise(self, noise, steps, batch_size, allow_missing=False):
        if noise is None:
            if allow_missing:
                return None
            return torch.zeros(
                (batch_size, steps, self.h_patches, self.w_patches, self.embed_dim),
                device=self.skip_spatial_proj[0].weight.device,
                dtype=self.skip_spatial_proj[0].weight.dtype,
            )
        expected_shape = (
            batch_size,
            steps,
            self.embed_dim,
            self.h_patches,
            self.w_patches,
        )
        if tuple(noise.shape) != expected_shape:
            raise ValueError(
                "Rollout noise shape mismatch: expected "
                f"{expected_shape}, got {tuple(noise.shape)}"
            )
        return noise.permute(0, 1, 3, 4, 2).contiguous()

    def sample_noise(self, x):
        if x.ndim == 5:
            shape = (
                x.shape[0],
                x.shape[1],
                self.embed_dim,
                self.h_patches,
                self.w_patches,
            )
        elif x.ndim == 4:
            shape = (x.shape[0], self.embed_dim, self.h_patches, self.w_patches)
        else:
            raise ValueError(
                "Unsupported input rank for explicit noise sampling: "
                f"{x.ndim}"
            )
        return torch.randn(shape, device=x.device, dtype=x.dtype)

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        return torch.randn(
            (batch_size, steps, self.embed_dim, self.h_patches, self.w_patches),
            device=device,
            dtype=dtype,
        )

    def _noise_step(self, noise_seq, step_idx):
        if noise_seq is None:
            return None
        return noise_seq[:, step_idx]

    def _validate_dt(self, dt):
        if _dt_has_negative(dt):
            raise ValueError("dt must be non-negative")

    def _skip_gate_4d(self, z_dec, z_skip):
        dec_ctx = torch.cat(
            [
                z_dec.real.mean(dim=(-2, -1)),
                z_dec.imag.mean(dim=(-2, -1)),
                z_skip.real.mean(dim=(-2, -1)),
                z_skip.imag.mean(dim=(-2, -1)),
            ],
            dim=1,
        )
        gate = self.skip_context_proj(dec_ctx).to(z_dec.real.dtype)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        delta_mag = (z_skip - z_dec).abs()
        gate = torch.sigmoid(
            gate + self.skip_spatial_proj(delta_mag).to(z_dec.real.dtype)
        )
        return z_dec * (1.0 - gate) + z_skip * gate

    def _apply_decoder_skip(self, z_dec, z_skip):
        if z_skip is None:
            return z_dec
        if z_dec.ndim == 5:
            batch_size, steps, dim, height, width = z_dec.shape
            z_dec_flat = z_dec.contiguous().view(batch_size * steps, dim, height, width)
            z_skip_flat = z_skip.contiguous().view(
                batch_size * steps, dim, height, width
            )
            z_out_flat = self._skip_gate_4d(z_dec_flat, z_skip_flat)
            return z_out_flat.view(batch_size, steps, dim, height, width)
        return self._skip_gate_4d(z_dec, z_skip)

    def forward(self, x, dt, z=None, return_latent=False):
        batch_size, steps = x.shape[0], x.shape[1]
        dt_seq = self._normalize_dt(dt, batch_size, steps, x.device)
        self._validate_dt(dt_seq)
        latent = self.encoder(x)
        z_skip = latent
        noise_seq = self._normalize_noise(latent, z)
        dtype = complex_dtype_for(latent.dtype)
        states = self._init_states(batch_size, x.device, dtype)
        use_checkpoint = self.training and torch.is_grad_enabled()
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                latent, h_next, flux_next = torch.utils.checkpoint.checkpoint(
                    block,
                    latent,
                    states[i][0],
                    dt_seq,
                    states[i][1],
                    noise_seq,
                    use_reentrant=False,
                )
            else:
                latent, h_next, flux_next = block(
                    latent,
                    states[i][0],
                    dt_seq,
                    states[i][1],
                    noise_seq,
                )
            states[i] = (h_next, flux_next)
        latent = self._apply_decoder_skip(latent, z_skip)
        out = self.decoder(latent)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), out.ndim)
        out = torch.where(zero_mask, x, out)
        if return_latent:
            return out, latent
        return out

    def _rollout_chunk_fn(
        self, z_curr, z_skip, x_curr, chunk_dt_stacked, chunk_noise, *flat_states
    ):
        num_layers = self.depth
        states = [
            (flat_states[i * 2], flat_states[i * 2 + 1])
            for i in range(num_layers)
        ]
        chunk_preds = []
        for step_idx, dt_step in enumerate(chunk_dt_stacked.unbind(0)):
            noise_step = self._noise_step(chunk_noise, step_idx)
            new_states = []
            for i, block in enumerate(self.blocks):
                z_curr, h_next, flux_next = block.forward_step(
                    z_curr,
                    states[i][0],
                    dt_step,
                    states[i][1],
                    noise_step,
                )
                new_states.append((h_next, flux_next))
            states = new_states
            x_pred = self.decoder(self._apply_decoder_skip(z_curr, z_skip))
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), x_pred.ndim)
            x_pred = torch.where(zero_mask, x_curr, x_pred)
            x_curr = x_pred
            chunk_preds.append(x_pred)

        out_tensors = [z_curr, x_curr] + chunk_preds
        for h_state, flux_state in states:
            out_tensors.extend([h_state, flux_state])
        return tuple(out_tensors)

    def forward_rollout(
        self,
        x_context,
        dt_context,
        dt_list,
        z_context=None,
        z_rollout=None,
        chunk_size=1,
        output_stride=1,
        output_offset=0,
    ):
        batch_size, steps_in = x_context.shape[0], x_context.shape[1]
        device = x_context.device
        dt_ctx_seq_full = self._normalize_dt(dt_context, batch_size, steps_in, device)
        self._validate_dt(dt_ctx_seq_full)
        z_ctx_full = self.encoder(x_context)
        noise_ctx_full = self._normalize_noise(
            z_ctx_full, z_context, allow_missing=steps_in <= 1
        )
        z_skip = z_ctx_full[:, -1]
        dtype = complex_dtype_for(z_ctx_full.dtype)
        states = self._init_states(batch_size, device, dtype)
        use_checkpoint = self.training and torch.is_grad_enabled()

        if steps_in > 1:
            z_ctx = z_ctx_full[:, :-1]
            dt_ctx_seq = dt_ctx_seq_full[:, 1:]
            noise_ctx = noise_ctx_full[:, :-1] if noise_ctx_full is not None else None
            for i, block in enumerate(self.blocks):
                if use_checkpoint:
                    z_ctx, h_final, flux_final = torch.utils.checkpoint.checkpoint(
                        block,
                        z_ctx,
                        states[i][0],
                        dt_ctx_seq,
                        states[i][1],
                        noise_ctx,
                        use_reentrant=False,
                    )
                else:
                    z_ctx, h_final, flux_final = block(
                        z_ctx,
                        states[i][0],
                        dt_ctx_seq,
                        states[i][1],
                        noise_ctx,
                    )
                states[i] = (h_final, flux_final)
        z_curr = z_ctx_full[:, -1]
        x_curr = x_context[:, -1]

        n_steps = len(dt_list)
        rollout_noise = self._normalize_rollout_noise(
            z_rollout,
            n_steps,
            batch_size,
            allow_missing=n_steps == 0,
        )
        dt_steps = [
            self._normalize_dt(dt_k, batch_size, 1, device).squeeze(1)
            for dt_k in dt_list
        ]
        for dt_step in dt_steps:
            self._validate_dt(dt_step)
        output_stride = max(1, int(output_stride))
        output_offset = int(output_offset)
        if use_checkpoint:
            chunk_size = 1

        preds = []
        step = 0
        while step < n_steps:
            chunk_end = min(step + chunk_size, n_steps)
            chunk_len = chunk_end - step
            chunk_dt_stacked = torch.stack(dt_steps[step:chunk_end], dim=0)
            chunk_noise = (
                rollout_noise[:, step:chunk_end]
                if rollout_noise is not None
                else None
            )

            flat_states = []
            for h_state, flux_state in states:
                flat_states.extend([h_state, flux_state])

            outs = torch.utils.checkpoint.checkpoint(
                self._rollout_chunk_fn,
                z_curr,
                z_skip,
                x_curr,
                chunk_dt_stacked,
                chunk_noise,
                *flat_states,
                use_reentrant=False,
            )

            z_curr = outs[0]
            x_curr = outs[1]
            for offset in range(chunk_len):
                step_idx = step + offset
                if (
                    step_idx >= output_offset
                    and (step_idx - output_offset) % output_stride == 0
                ):
                    preds.append(outs[2 + offset])
            states = [
                (outs[2 + chunk_len + i * 2], outs[2 + chunk_len + i * 2 + 1])
                for i in range(self.depth)
            ]
            step = chunk_end

        return torch.stack(preds, dim=1)
