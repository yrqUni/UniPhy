import torch
import torch.nn as nn
from PScan import pscan
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork


class UniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand,
        num_experts,
        img_height,
        img_width,
        kernel_size=3,
        dt_ref=1.0,
        init_noise_scale=0.01,
        sde_mode="sde",
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = dim
        self.img_height = img_height
        self.img_width = img_width

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
            dt_ref=dt_ref,
            sde_mode=sde_mode,
            init_noise_scale=init_noise_scale,
            max_growth_rate=max_growth_rate,
        )

        self.ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts)
        self.drift_proj = nn.Linear(dim * 2, dim)
        self.flux_tracker = None
        self.last_h_state = None
        self.last_flux_state = None

    def _init_flux_tracker(self, dim, device, dtype):
        from UniPhyOps import GlobalFluxTracker
        self.flux_tracker = GlobalFluxTracker(dim).to(device)
        if dtype == torch.cdouble:
            self.flux_tracker = self.flux_tracker.double()

    def _run_pscan(self, A, X):
        return pscan(A, X)

    def forward(self, x, dt):
        B, T, D, H, W = x.shape

        x_flat = x.reshape(B * T, D, H, W)
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        x_spatial_complex = torch.complex(x_re, x_im)
        x_spatial_5d = x_spatial_complex.reshape(B, T, D, H, W)

        u_t = self.prop.basis.encode(x_spatial_5d.permute(0, 1, 3, 4, 2))

        if self.flux_tracker is None:
            self._init_flux_tracker(D, x.device, x.dtype)

        x_mean = x_spatial_5d.mean(dim=(-2, -1))
        flux_A, flux_X = self.flux_tracker.get_operators(x_mean)
        flux_state = self._run_pscan(
            flux_A.unsqueeze(-1),
            flux_X.unsqueeze(-1)
        ).squeeze(-1)
        flux_output = self.flux_tracker.project(flux_state)

        drift_input = torch.cat([flux_output.real, flux_output.imag], dim=-1)
        drift = self.drift_proj(drift_input.reshape(B * T, -1))
        drift = drift.reshape(B, T, D)
        drift_complex = torch.complex(
            drift[..., : D // 2] if D > 1 else drift,
            drift[..., D // 2 :] if D > 1 else torch.zeros_like(drift),
        )

        dt_tensor = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=x.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.unsqueeze(0).expand(T)
        dt_expanded = dt_tensor.view(1, T, 1, 1, 1).expand(B, T, H, W, 1)

        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)

        op_decay_exp = op_decay.expand(B, T, H, W, D)
        op_forcing_exp = op_forcing.expand(B, T, H, W, D)

        u_t_scaled = u_t * op_forcing_exp

        A_time = op_decay_exp.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)
        X_time = u_t_scaled.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)

        Y_time = self._run_pscan(A_time, X_time)

        u_out = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        self.last_h_state = u_out[:, -1].permute(0, 1, 2, 3).reshape(B * H * W, 1, D).detach()
        self.last_flux_state = flux_state[:, -1].detach()

        noise = self.prop.generate_stochastic_term(
            u_out.shape, dt_expanded, u_out.dtype, h_state=u_out
        )
        u_out = u_out + noise

        x_out = self.prop.basis.decode(u_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)

        x_out_flat = x_out.reshape(B * T, D, H, W)
        x_out_real = torch.cat([x_out_flat.real, x_out_flat.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        delta = self.ffn(x_out_complex)
        delta = delta.reshape(B, T, D, H, W)

        return x + delta

    def forward_step(self, x_t, h_prev, dt, flux_prev=None):
        B, D, H, W = x_t.shape

        x_real = torch.cat([x_t.real, x_t.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
        x_spatial_complex = torch.complex(x_re, x_im)

        forcing = self.prop.basis.encode(x_spatial_complex.permute(0, 2, 3, 1))

        if self.flux_tracker is None:
            self._init_flux_tracker(D, x_t.device, x_t.dtype)

        x_mean = x_spatial_complex.mean(dim=(-2, -1))
        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=x_t.device, dtype=x_t.dtype)
        flux_state, flux_output, flux_gate = self.flux_tracker.forward_step(flux_prev, x_mean)

        dt_val = dt.item() if isinstance(dt, torch.Tensor) and dt.numel() == 1 else dt
        dt_expanded = torch.tensor(
            dt_val,
            device=x_t.device,
            dtype=torch.float64 if x_t.dtype == torch.cdouble else torch.float32,
        )
        dt_expanded = dt_expanded.view(1, 1, 1, 1).expand(B, H, W, 1)

        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)

        h_prev_latent = h_prev.reshape(B, H, W, D)
        forcing_term = forcing

        h_next = h_prev_latent * op_decay + forcing_term * op_forcing

        noise = self.prop.generate_stochastic_term(
            h_next.shape, dt_expanded, h_next.dtype, h_state=h_next
        )
        h_next = h_next + noise

        x_out = self.prop.basis.decode(h_next)
        x_out = x_out.permute(0, 3, 1, 2)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        delta = self.ffn(x_out_complex)

        out = x_t + delta
        h_next_flat = h_next.reshape(B * H * W, 1, D)

        return out, h_next_flat, flux_state


class UniPhyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        expand,
        num_experts,
        depth,
        patch_size,
        img_height,
        img_width,
        dt_ref=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
    ):
        super().__init__()

        self.encoder = UniPhyEncoder(
            in_channels,
            embed_dim,
            patch_size,
            img_height=img_height,
            img_width=img_width,
        )

        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        latent_h = (img_height + pad_h) // patch_size
        latent_w = (img_width + pad_w) // patch_size

        self.blocks = nn.ModuleList(
            [
                UniPhyBlock(
                    embed_dim,
                    expand,
                    num_experts,
                    latent_h,
                    latent_w,
                    dt_ref=dt_ref,
                    sde_mode=sde_mode,
                    init_noise_scale=init_noise_scale,
                    max_growth_rate=max_growth_rate,
                )
                for _ in range(depth)
            ]
        )

        self.decoder = UniPhyEnsembleDecoder(
            out_channels,
            embed_dim,
            patch_size,
            img_height=img_height,
            img_width=img_width,
        )

        self.latent_h = latent_h
        self.latent_w = latent_w
        self.embed_dim = embed_dim

    def forward(self, x, dt):
        z = self.encoder(x)
        for block in self.blocks:
            z = block(z, dt)
        return self.decoder(z)

    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        z = self.encoder(x_cond)

        for block in self.blocks:
            z = block(z, dt_cond)

        states = []
        for block in self.blocks:
            B, T, D, H, W = z.shape
            curr_flux = torch.zeros(B, block.dim, device=device, dtype=torch.cdouble)
            if block.last_flux_state is not None:
                curr_flux = block.last_flux_state
            elif z.shape[1] > 0:
                z_perm = z.permute(0, 1, 3, 4, 2)
                x_encoded = block.prop.basis.encode(z_perm)
                x_eigen_last_seq = x_encoded.mean(dim=(2, 3))
                for t in range(x_eigen_last_seq.shape[1]):
                    curr_flux, _, _ = block.prop.flux_tracker.forward_step(
                        curr_flux, x_eigen_last_seq[:, t]
                    )
            states.append((block.last_h_state.to("cpu"), curr_flux.to("cpu")))
            block.last_h_state = None
            block.last_flux_state = None

        z_curr = z[:, -1].detach()
        del z
        predictions = []

        for k in range(k_steps):
            dt_k = (
                dt_future[:, k]
                if (isinstance(dt_future, torch.Tensor) and dt_future.ndim > 0)
                else dt_future
            )
            z_next = z_curr
            new_states = []

            for i, block in enumerate(self.blocks):
                h_prev_latent = states[i][0].to(device, non_blocking=True)
                flux_prev = states[i][1].to(device, non_blocking=True)

                z_next, h_next_latent, flux_next = block.forward_step(
                    z_next, h_prev_latent, dt_k, flux_prev
                )

                new_states.append(
                    (
                        h_next_latent.to("cpu", non_blocking=True),
                        flux_next.to("cpu", non_blocking=True),
                    )
                )

                del h_prev_latent
                del h_next_latent

            states = new_states
            z_curr = z_next

            pred_pixel = (
                self.decoder(z_curr.unsqueeze(1))
                .squeeze(1)
                .to("cpu", non_blocking=True)
            )
            predictions.append(pred_pixel)

        return torch.stack(predictions, dim=1)
    