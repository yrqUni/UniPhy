import torch
import torch.nn as nn

from .PScan import pscan
from .UniPhyFFN import UniPhyFeedForwardNetwork
from .UniPhyIO import (
    UniPhyEncoder,
    UniPhyEnsembleDecoder,
    UniPhyRealDecoder,
    UniPhyRealEncoder,
)
from .UniPhyOps import (
    MultiScaleSpatialMixer,
    RealMultiScaleSpatialMixer,
    TemporalPropagator,
    complex_dtype_for,
)


def _dt_is_zero(dt, tol=1e-12):
    return dt.abs() <= tol


def _dt_has_negative(dt, tol=1e-12):
    return bool((dt < -tol).any().item())


def _dt_has_nonfinite(dt):
    return not bool(torch.isfinite(dt).all().item())


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
        batch_size, height, width, dim = x_local.real.shape
        spatial_feat = torch.cat(
            [x_local.real.permute(0, 3, 1, 2), x_local.imag.permute(0, 3, 1, 2)],
            dim=1,
        ).contiguous()
        spatial_gate = self.spatial_proj(spatial_feat).permute(0, 2, 3, 1).contiguous()
        gate_combined = gate_global.unsqueeze(1).unsqueeze(2) * spatial_gate

        src_re = (
            source_global.real.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, dim)
        )
        src_im = (
            source_global.imag.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, height, width, dim)
        )
        source_field = torch.complex(src_re, src_im)

        return x_local * gate_combined + source_field * (1.0 - gate_combined)

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


class RealTemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref, init_noise_scale):
        super().__init__()
        self.dim = dim
        self.dt_ref = float(dt_ref)
        self.decay_logit = nn.Parameter(torch.randn(dim) * 0.01)
        self.noise_scale = nn.Parameter(torch.ones(dim) * init_noise_scale)
        self.h0 = nn.Parameter(torch.zeros(dim))

    def get_initial_h(self, batch_size, height, width, device, dtype):
        h0 = self.h0.to(device=device, dtype=dtype)
        return h0.reshape(1, 1, 1, self.dim).expand(batch_size, height, width, -1)

    def decay(self, dt, target_ndim):
        dt_ratio = dt / self.dt_ref
        while dt_ratio.ndim < target_ndim:
            dt_ratio = dt_ratio.unsqueeze(-1)
        lam = torch.nn.functional.softplus(self.decay_logit).to(dt_ratio.dtype)
        while lam.ndim < target_ndim:
            lam = lam.unsqueeze(0)
        return torch.exp(-lam * dt_ratio)

    def noise(self, noise, shape, dt):
        if noise is None:
            return torch.zeros(shape, device=dt.device, dtype=dt.dtype)
        noise = noise.real if torch.is_complex(noise) else noise
        noise = noise.to(device=dt.device, dtype=dt.dtype).view(shape)
        scale = self.noise_scale.abs().to(noise.dtype)
        while scale.ndim < noise.ndim:
            scale = scale.unsqueeze(0)
        dt_scale = torch.sqrt(torch.clamp(dt / self.dt_ref, min=0.0))
        while dt_scale.ndim < noise.ndim:
            dt_scale = dt_scale.unsqueeze(-1)
        return noise * scale * dt_scale


class RealUniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand,
        dt_ref,
        init_noise_scale,
    ):
        super().__init__()
        self.prop = RealTemporalPropagator(dim, dt_ref, init_noise_scale)
        self.spatial_mixer = RealMultiScaleSpatialMixer(dim)
        del expand

    def _decode_sequence(self, seq, lead_time):
        del lead_time
        return seq.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
        batch_size, steps, dim, height, width = x.shape
        x_real = x.real if torch.is_complex(x) else x
        x_flat = x_real.contiguous().reshape(batch_size * steps, dim, height, width)
        x_flat = self.spatial_mixer(x_flat)
        x_seq = x_flat.reshape(batch_size, steps, dim, height, width)
        x_hw = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        rows = []
        for t in range(steps):
            dt_t = dt_seq[:, t]
            decay = self.prop.decay(dt_t, 4)
            noise_t = None if noise_seq is None else noise_seq[:, t]
            h_next = h_cur * decay + x_hw[:, t] * (1.0 - decay)
            h_next = h_next + self.prop.noise(noise_t, h_next.shape, dt_t)
            zero_mask = _expand_batch_mask(_dt_is_zero(dt_t), h_next.ndim)
            h_cur = torch.where(zero_mask, x_hw[:, t], h_next)
            rows.append(h_cur)
        seq = torch.stack(rows, dim=1)
        lead_time = torch.cumsum(dt_seq, dim=1)
        combined = self._decode_sequence(seq, lead_time)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x_real, combined)
        h_out = seq[:, -1].reshape(batch_size * height * width, 1, dim)
        flux_out = seq[:, -1].mean(dim=(1, 2))
        return combined, h_out, flux_out

    def forward_step(
        self,
        x_curr,
        x_next,
        h_prev,
        dt_step,
        dt_next,
        flux_prev,
        noise_step=None,
        lead_time=None,
    ):
        del x_next, dt_next, flux_prev
        batch_size, dim, height, width = x_curr.shape
        x_real = x_curr.real if torch.is_complex(x_curr) else x_curr
        x_mix = self.spatial_mixer(x_real.contiguous())
        x_hw = x_mix.permute(0, 2, 3, 1).contiguous()
        h_cur = h_prev.real if torch.is_complex(h_prev) else h_prev
        h_cur = h_cur.contiguous().reshape(batch_size, height, width, dim)
        decay = self.prop.decay(dt_step, 4)
        h_next = h_cur * decay + x_hw * (1.0 - decay)
        h_next = h_next + self.prop.noise(noise_step, h_next.shape, dt_step)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), h_next.ndim)
        h_next = torch.where(zero_mask, x_hw, h_next)
        if lead_time is None:
            lead_time = dt_step
        z_next = self._decode_sequence(h_next.unsqueeze(1), lead_time.unsqueeze(1))[
            :, 0
        ]
        z_next = torch.where(
            _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim),
            x_real,
            z_next,
        )
        flux_next = h_next.mean(dim=(1, 2))
        return z_next, h_next.reshape(batch_size * height * width, 1, dim), flux_next


class UniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand,
        dt_ref,
        init_noise_scale,
    ):
        super().__init__()
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

    def _decode_state(self, h_state, basis_w_inv):
        return (
            self.prop.basis.decode_with(h_state, basis_w_inv)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _decode_sequence(self, h_seq, basis_w_inv):
        return (
            self.prop.basis.decode_with(h_seq, basis_w_inv)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

    def _apply_spatial(self, x_4d):
        return self.spatial_mixer(x_4d)

    def _apply_temporal_decode(self, x_4d):
        x_real = torch.cat([x_4d.real, x_4d.imag], dim=1)
        x_norm = (
            self.norm_temporal(x_real.permute(0, 2, 3, 1).contiguous())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)
        delta = self.ffn(x_complex)
        return x_4d + delta

    def _normalize_block_noise_seq(
        self, noise_seq, batch_size, steps, dim, height, width
    ):
        if noise_seq is None:
            return None
        expected_shape = (batch_size, steps, height, width, dim)
        torch.empty(
            expected_shape,
            device=noise_seq.device,
            dtype=noise_seq.dtype,
        ).copy_(noise_seq)
        return noise_seq

    def _normalize_block_noise_step(self, noise_step, batch_size, dim, height, width):
        if noise_step is None:
            return None
        expected_shape = (batch_size, height, width, dim)
        torch.empty(
            expected_shape,
            device=noise_step.device,
            dtype=noise_step.dtype,
        ).copy_(noise_step)
        return noise_step

    def _shift_forcing_next(self, forcing):
        steps = forcing.shape[1]
        if steps == 1:
            return forcing
        forcing_next = forcing.clone()
        forcing_next[:, :-1] = forcing[:, 1:]
        if steps >= 2:
            forcing_next[:, -1] = 2.0 * forcing[:, -1] - forcing[:, -2]
        return forcing_next

    def _compute_forcing_seq(self, x, dt_seq, flux_prev):
        batch_size, steps, dim, height, width = x.shape
        basis_dtype = complex_dtype_for(x.dtype)
        basis_w, basis_w_inv = self.prop.get_basis_matrices(basis_dtype)
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()
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
        return forcing, flux_out, basis_w_inv

    def _compute_forcing_step(self, x_curr, dt_step, flux_prev):
        basis_dtype = complex_dtype_for(x_curr.dtype)
        basis_w, basis_w_inv = self.prop.get_basis_matrices(basis_dtype)
        x_eigen = self.prop.basis.encode_with(
            x_curr.permute(0, 2, 3, 1).contiguous(),
            basis_w,
        )
        x_mean = self.flux_pool(x_eigen)
        flux_next, source, gate = self.prop.flux_tracker.forward_step(
            flux_prev,
            x_mean,
            dt_step,
        )
        forcing = self.spatial_gate(gate, source, x_eigen)
        return forcing, flux_next, basis_w_inv

    def forward(self, x, h_prev, dt_seq, flux_prev, noise_seq=None):
        batch_size, steps, dim, height, width = x.shape
        x_flat = x.reshape(batch_size * steps, dim, height, width)
        x_flat = self._apply_spatial(x_flat)
        x = x_flat.reshape(batch_size, steps, dim, height, width)

        forcing, flux_out, basis_w_inv = self._compute_forcing_seq(x, dt_seq, flux_prev)
        forcing_next = self._shift_forcing_next(forcing)

        decay, alpha, beta = self.prop.get_etd2_operators(dt_seq)
        decay = (
            decay.unsqueeze(2)
            .unsqueeze(3)
            .expand(
                batch_size,
                steps,
                height,
                width,
                dim,
            )
        )
        alpha = (
            alpha.unsqueeze(2)
            .unsqueeze(3)
            .expand(
                batch_size,
                steps,
                height,
                width,
                dim,
            )
        )
        beta = (
            beta.unsqueeze(2)
            .unsqueeze(3)
            .expand(
                batch_size,
                steps,
                height,
                width,
                dim,
            )
        )

        noise_seq = self._normalize_block_noise_seq(
            noise_seq,
            batch_size,
            steps,
            dim,
            height,
            width,
        )
        u_t = forcing * alpha + forcing_next * beta
        u_t = u_t + self.prop.generate_stochastic_term(
            u_t.shape,
            dt_seq,
            u_t.dtype,
            noise=noise_seq,
        )

        h_prev_hw = h_prev.reshape(batch_size, height, width, dim)
        h_contrib_t0 = h_prev_hw * decay[:, 0]
        u_t0 = u_t[:, 0] + h_contrib_t0
        u_t = torch.cat([u_t0.unsqueeze(1), u_t[:, 1:]], dim=1)

        a_scan = decay.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            steps,
            dim,
            1,
        )
        x_scan = u_t.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width,
            steps,
            dim,
            1,
        )
        u_out = (
            pscan(a_scan, x_scan)
            .reshape(
                batch_size,
                height,
                width,
                steps,
                dim,
            )
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )

        h_out = u_out[:, -1].reshape(batch_size * height * width, 1, dim)
        decoded = self._decode_sequence(u_out, basis_w_inv)
        decoded_flat = decoded.reshape(batch_size * steps, dim, height, width)
        decoded_with_residual_flat = self._apply_temporal_decode(decoded_flat)
        decoded_with_residual = decoded_with_residual_flat.reshape(
            batch_size,
            steps,
            dim,
            height,
            width,
        )
        combined = decoded_with_residual
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), combined.ndim)
        combined = torch.where(zero_mask, x, combined)
        return combined, h_out, flux_out

    def _decode_h(self, h_4d, basis_w_inv):
        decoded = self._decode_state(h_4d, basis_w_inv)
        return self._apply_temporal_decode(decoded)

    def forward_step(
        self,
        x_curr,
        x_next,
        h_prev,
        dt_step,
        dt_next,
        flux_prev,
        noise_step=None,
        lead_time=None,
    ):
        del lead_time
        if x_next is None or dt_next is None:
            raise ValueError(
                "forward_step requires both x_next (the encoded latent at "
                "t + dt_step) and dt_next (the interval following). Cox-Matthews "
                "ETD2 has piecewise-linear forcing on [t, t + dt_step] anchored "
                "on the two endpoints, and the flux propagator must advance "
                "consistently to the second anchor."
            )
        batch_size, dim, height, width = x_curr.shape
        x_input = x_curr
        x_curr_mixed = self._apply_spatial(x_curr)
        x_next_mixed = self._apply_spatial(x_next)

        forcing_curr, flux_after_curr, basis_w_inv = self._compute_forcing_step(
            x_curr_mixed,
            dt_step,
            flux_prev,
        )
        forcing_next, flux_next, _ = self._compute_forcing_step(
            x_next_mixed,
            dt_next,
            flux_after_curr,
        )

        decay, alpha, beta = self.prop.get_etd2_operators(dt_step)
        decay = (
            decay.unsqueeze(1)
            .unsqueeze(2)
            .expand(
                batch_size,
                height,
                width,
                dim,
            )
        )
        alpha = (
            alpha.unsqueeze(1)
            .unsqueeze(2)
            .expand(
                batch_size,
                height,
                width,
                dim,
            )
        )
        beta = (
            beta.unsqueeze(1)
            .unsqueeze(2)
            .expand(
                batch_size,
                height,
                width,
                dim,
            )
        )

        h_prev_hw = h_prev.reshape(batch_size, height, width, dim)
        noise_step = self._normalize_block_noise_step(
            noise_step,
            batch_size,
            dim,
            height,
            width,
        )
        noise = self.prop.generate_stochastic_term(
            h_prev_hw.shape,
            dt_step,
            h_prev_hw.dtype,
            noise=noise_step,
        )

        h_next = h_prev_hw * decay + forcing_curr * alpha + forcing_next * beta + noise

        decoded_with_residual = self._decode_h(h_next, basis_w_inv)
        z_next = decoded_with_residual
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_step), z_next.ndim)
        z_next = torch.where(zero_mask, x_input, z_next)
        h_next = torch.where(zero_mask, h_prev_hw, h_next)
        flux_after_curr_masked = torch.where(
            _expand_batch_mask(_dt_is_zero(dt_step), flux_after_curr.ndim),
            flux_prev,
            flux_after_curr,
        )
        return (
            z_next,
            h_next.reshape(batch_size * height * width, 1, dim),
            flux_after_curr_masked,
        )


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
        latent_dynamics="real",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.latent_dynamics = latent_dynamics
        ph, pw = patch_size
        self.h_patches = (img_height + (ph - img_height % ph) % ph) // ph
        self.w_patches = (img_width + (pw - img_width % pw) % pw) // pw
        encoder_cls = UniPhyRealEncoder if latent_dynamics == "real" else UniPhyEncoder
        decoder_cls = (
            UniPhyRealDecoder if latent_dynamics == "real" else UniPhyEnsembleDecoder
        )
        self.encoder = encoder_cls(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )
        self.decoder = decoder_cls(
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
        block_cls = RealUniPhyBlock if latent_dynamics == "real" else UniPhyBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=embed_dim,
                    expand=expand,
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
            h0_dtype = (
                self.skip_spatial_proj[0].weight.dtype
                if self.latent_dynamics == "real"
                else dtype
            )
            h0 = block.prop.get_initial_h(batch_size, height, width, device, h0_dtype)
            h0_flat = h0.reshape(batch_size * height * width, 1, dim)
            if hasattr(block.prop, "flux_tracker"):
                f0 = block.prop.flux_tracker.get_initial_state(batch_size, device, dtype)
            else:
                f0 = torch.zeros(batch_size, dim, device=device, dtype=h0_flat.dtype)
            states.append((h0_flat, f0))
        return states

    def _normalize_dt(self, dt, batch_size, steps, device):
        dtype = self.skip_spatial_proj[0].weight.dtype
        dt = dt.detach().to(device=device, dtype=dtype)
        if dt.ndim != 2:
            raise ValueError(f"dt_must_be_2d: got_ndim={dt.ndim}")
        rows, cols = dt.shape
        if (rows, cols) != (batch_size, steps):
            raise ValueError(
                f"dt_shape_mismatch: expected={(batch_size, steps)} got={(rows, cols)}"
            )
        return dt

    def _normalize_step_dt(self, dt, batch_size, device):
        dtype = self.skip_spatial_proj[0].weight.dtype
        dt = dt.detach().to(device=device, dtype=dtype)
        if dt.ndim != 1:
            raise ValueError(f"dt_step_must_be_1d: got_ndim={dt.ndim}")
        (length,) = dt.shape
        if length != batch_size:
            raise ValueError(
                f"dt_step_shape_mismatch: expected={batch_size} got={length}"
            )
        return dt

    def _noise_shape_from_latent(self, latent):
        return (
            latent.shape[0],
            latent.shape[1],
            latent.shape[3],
            latent.shape[4],
            latent.shape[2],
        )

    def _normalize_noise(self, latent, noise):
        expected_shape = self._noise_shape_from_latent(latent)
        if noise is None:
            return torch.zeros(
                expected_shape,
                device=latent.device,
                dtype=latent.dtype,
            )
        return noise.to(device=latent.device, dtype=latent.dtype).view(expected_shape)

    def _normalize_rollout_noise(self, noise, steps, batch_size):
        if noise is None:
            return None
        if isinstance(noise, bool):
            if not noise:
                return None
            return torch.zeros(
                batch_size,
                steps,
                self.h_patches,
                self.w_patches,
                self.embed_dim,
                device=self.skip_spatial_proj[0].weight.device,
                dtype=self.skip_spatial_proj[0].weight.dtype,
            )
        expected_shape = (
            batch_size,
            steps,
            self.h_patches,
            self.w_patches,
            self.embed_dim,
        )
        return noise.to(
            device=self.skip_spatial_proj[0].weight.device,
            dtype=self.skip_spatial_proj[0].weight.dtype,
        ).view(expected_shape)

    def sample_noise(self, x):
        shape = (
            x.shape[0],
            x.shape[1],
            self.h_patches,
            self.w_patches,
            self.embed_dim,
        )
        if self.latent_dynamics == "real":
            return torch.randn(shape, device=x.device, dtype=x.dtype)
        noise_dtype = complex_dtype_for(x.dtype)
        real_dtype = torch.float64 if noise_dtype == torch.complex128 else torch.float32
        real = torch.randn(shape, device=x.device, dtype=real_dtype)
        imag = torch.randn(shape, device=x.device, dtype=real_dtype)
        return torch.complex(real, imag).to(noise_dtype)

    def sample_block_noises(self, x):
        B, T = x.shape[0], x.shape[1]
        if self.latent_dynamics == "real":
            shape_all = (self.depth, B, T, self.h_patches, self.w_patches, self.embed_dim)
            noise_all = torch.randn(shape_all, device=x.device, dtype=x.dtype)
            return [noise_all[i] for i in range(self.depth)]
        noise_dtype = complex_dtype_for(x.dtype)
        real_dtype = torch.float64 if noise_dtype == torch.complex128 else torch.float32
        shape_all = (self.depth, B, T, self.h_patches, self.w_patches, self.embed_dim)
        real_all = torch.randn(shape_all, device=x.device, dtype=real_dtype)
        imag_all = torch.randn(shape_all, device=x.device, dtype=real_dtype)
        noise_all = torch.complex(real_all, imag_all).to(noise_dtype)
        return [noise_all[i] for i in range(self.depth)]

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        shape = (batch_size, steps, self.h_patches, self.w_patches, self.embed_dim)
        if self.latent_dynamics == "real":
            return torch.randn(shape, device=device, dtype=dtype)
        noise_dtype = complex_dtype_for(dtype)
        real_dtype = torch.float64 if noise_dtype == torch.complex128 else torch.float32
        real = torch.randn(shape, device=device, dtype=real_dtype)
        imag = torch.randn(shape, device=device, dtype=real_dtype)
        return torch.complex(real, imag).to(noise_dtype)

    def _noise_step(self, noise_seq, step_idx):
        if noise_seq is None:
            return None
        return noise_seq[:, step_idx]

    def _validate_dt(self, dt):
        if _dt_has_nonfinite(dt):
            raise ValueError("dt_must_be_finite")
        if _dt_has_negative(dt):
            raise ValueError("dt_must_be_nonnegative")

    def _skip_gate_4d(self, z_dec, z_skip):
        def re(t):
            return t.real if torch.is_complex(t) else t

        def im(t):
            return t.imag if torch.is_complex(t) else torch.zeros_like(t)

        dec_ctx = torch.cat(
            [
                re(z_dec).mean(dim=(-2, -1)),
                im(z_dec).mean(dim=(-2, -1)),
                re(z_skip).mean(dim=(-2, -1)),
                im(z_skip).mean(dim=(-2, -1)),
            ],
            dim=1,
        )
        gate = self.skip_context_proj(dec_ctx).to(re(z_dec).dtype)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        delta_mag = (re(z_skip) - re(z_dec)).abs()
        gate = torch.sigmoid(
            gate + self.skip_spatial_proj(delta_mag).to(re(z_dec).dtype)
        )
        return z_dec * (1.0 - gate) + z_skip * gate

    def _apply_decoder_skip(self, z_dec, z_skip):
        if z_skip is None:
            return z_dec
        batch_size, steps, dim, height, width = z_dec.shape
        z_dec_flat = z_dec.contiguous().view(batch_size * steps, dim, height, width)
        z_skip_flat = z_skip.contiguous().view(batch_size * steps, dim, height, width)
        z_out_flat = self._skip_gate_4d(z_dec_flat, z_skip_flat)
        return z_out_flat.view(batch_size, steps, dim, height, width)

    def forward(self, x, dt, z=None, return_latent=False):
        batch_size, steps = x.shape[0], x.shape[1]
        dt_seq = self._normalize_dt(dt, batch_size, steps, x.device)
        self._validate_dt(dt_seq)
        latent = self.encoder(x)
        z_skip = latent
        dtype = (
            latent.dtype
            if self.latent_dynamics == "real"
            else complex_dtype_for(latent.dtype)
        )
        states = self._init_states(batch_size, x.device, dtype)
        use_checkpoint = self.training and torch.is_grad_enabled()
        stochastic = z is not None
        block_noises = (
            self.sample_block_noises(x) if stochastic else [None] * self.depth
        )
        for i, block in enumerate(self.blocks):
            block_noise = (
                self._normalize_noise(latent, block_noises[i]) if stochastic else None
            )
            if use_checkpoint:
                latent, h_next, flux_next = torch.utils.checkpoint.checkpoint(
                    block,
                    latent,
                    states[i][0],
                    dt_seq,
                    states[i][1],
                    block_noise,
                    use_reentrant=False,
                )
            else:
                latent, h_next, flux_next = block(
                    latent,
                    states[i][0],
                    dt_seq,
                    states[i][1],
                    block_noise,
                )
            states[i] = (h_next, flux_next)
        latent = self._apply_decoder_skip(latent, z_skip)
        out = self.decoder(latent)
        zero_mask = _expand_batch_mask(_dt_is_zero(dt_seq), out.ndim)
        out = torch.where(zero_mask, x, out)
        if return_latent:
            return out, latent
        return out

    def _predict_next_z(
        self,
        z_curr,
        dt_step,
        dt_next,
        states,
        step_block_noises,
        lead_time=None,
    ):
        z_running = z_curr
        for i, block in enumerate(self.blocks):
            z_running, _h_pred, _flux_pred = block.forward_step(
                z_running,
                z_running,
                states[i][0],
                dt_step,
                dt_next,
                states[i][1],
                step_block_noises[i],
                lead_time=lead_time,
            )
        return z_running

    def _rollout_chunk_fn(
        self,
        z_curr,
        x_curr,
        chunk_dt_stacked,
        chunk_dt_next_stacked,
        chunk_lead_stacked,
        chunk_noise,
        *flat_states,
    ):
        num_layers = self.depth
        states = [
            (flat_states[i * 2], flat_states[i * 2 + 1]) for i in range(num_layers)
        ]
        chunk_preds = []
        dt_seq = chunk_dt_stacked.unbind(0)
        dt_next_seq = chunk_dt_next_stacked.unbind(0)
        lead_seq = chunk_lead_stacked.unbind(0)
        for step_idx, (dt_step, dt_next, lead_time) in enumerate(
            zip(dt_seq, dt_next_seq, lead_seq)
        ):
            step_skip = z_curr
            if chunk_noise is not None:
                step_block_noises = self.sample_block_noises(z_curr.unsqueeze(1))
                step_block_noises = [n[:, 0] for n in step_block_noises]
            else:
                step_block_noises = [None] * self.depth

            z_pred = self._predict_next_z(
                z_curr,
                dt_step,
                dt_next,
                states,
                step_block_noises,
                lead_time=lead_time,
            )

            new_states = []
            z_running = z_curr
            for i, block in enumerate(self.blocks):
                z_next_for_block = z_pred if i == 0 else z_running
                z_running, h_next, flux_next = block.forward_step(
                    z_running,
                    z_next_for_block,
                    states[i][0],
                    dt_step,
                    dt_next,
                    states[i][1],
                    step_block_noises[i],
                    lead_time=lead_time,
                )
                new_states.append((h_next, flux_next))
            states = new_states
            z_curr = self._apply_decoder_skip(
                z_running.unsqueeze(1),
                step_skip.unsqueeze(1),
            )[:, 0]
            x_pred = self.decoder(z_curr.unsqueeze(1))[:, 0]
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
        dtype = (
            z_ctx_full.dtype
            if self.latent_dynamics == "real"
            else complex_dtype_for(z_ctx_full.dtype)
        )
        states = self._init_states(batch_size, device, dtype)
        use_checkpoint = self.training and torch.is_grad_enabled()

        n_steps = len(dt_list)
        rollout_noise = self._normalize_rollout_noise(
            z_rollout,
            n_steps,
            batch_size,
        )
        dt_steps = [
            self._normalize_step_dt(dt_k, batch_size, device) for dt_k in dt_list
        ]
        for dt_step in dt_steps:
            self._validate_dt(dt_step)

        if n_steps > 0:
            dt_next_steps = dt_steps[1:] + [dt_steps[-1]]
            lead_steps = []
            lead_accum = torch.zeros_like(dt_steps[0])
            for dt_step in dt_steps:
                lead_accum = lead_accum + dt_step
                lead_steps.append(lead_accum)
        else:
            dt_next_steps = []
            lead_steps = []

        if steps_in > 1:
            has_ctx_noise = z_context is not None
            ctx_noise_steps = None
            if has_ctx_noise:
                ctx_src = z_ctx_full[:, :-1]
                ctx_noise_steps = self.sample_block_noises(ctx_src)
            for step_idx in range(steps_in - 1):
                z_curr_for_block = z_ctx_full[:, step_idx]
                z_next_for_block = z_ctx_full[:, step_idx + 1]
                dt_step = dt_ctx_seq_full[:, step_idx + 1]
                if step_idx + 2 < steps_in:
                    dt_next = dt_ctx_seq_full[:, step_idx + 2]
                elif len(dt_steps) > 0:
                    dt_next = dt_steps[0]
                else:
                    dt_next = dt_step
                for i, block in enumerate(self.blocks):
                    h_prev, flux_prev = states[i]
                    noise_i = (
                        ctx_noise_steps[i][:, step_idx]
                        if ctx_noise_steps is not None
                        else None
                    )
                    z_curr_for_block, h_next, flux_next = block.forward_step(
                        z_curr_for_block,
                        z_next_for_block,
                        h_prev,
                        dt_step,
                        dt_next,
                        flux_prev,
                        noise_i,
                        lead_time=dt_step,
                    )
                    z_next_for_block = z_curr_for_block
                    states[i] = (h_next, flux_next)
        z_curr = z_ctx_full[:, -1]
        x_curr = x_context[:, -1]
        output_stride = max(1, int(output_stride))
        output_offset = int(output_offset)
        chunk_size = max(1, int(chunk_size))
        if use_checkpoint:
            chunk_size = 1

        preds = []
        step = 0
        while step < n_steps:
            chunk_end = min(step + chunk_size, n_steps)
            chunk_len = chunk_end - step
            chunk_dt_stacked = torch.stack(dt_steps[step:chunk_end], dim=0)
            chunk_dt_next_stacked = torch.stack(
                dt_next_steps[step:chunk_end],
                dim=0,
            )
            chunk_lead_stacked = torch.stack(lead_steps[step:chunk_end], dim=0)
            chunk_noise = (
                rollout_noise[:, step:chunk_end] if rollout_noise is not None else None
            )

            flat_states = []
            for h_state, flux_state in states:
                flat_states.extend([h_state, flux_state])

            if use_checkpoint:
                outs = torch.utils.checkpoint.checkpoint(
                    self._rollout_chunk_fn,
                    z_curr,
                    x_curr,
                    chunk_dt_stacked,
                    chunk_dt_next_stacked,
                    chunk_lead_stacked,
                    chunk_noise,
                    *flat_states,
                    use_reentrant=False,
                )
            else:
                outs = self._rollout_chunk_fn(
                    z_curr,
                    x_curr,
                    chunk_dt_stacked,
                    chunk_dt_next_stacked,
                    chunk_lead_stacked,
                    chunk_noise,
                    *flat_states,
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
