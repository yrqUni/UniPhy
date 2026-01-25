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
        self.drift_proj = nn.Linear(dim, dim)
        self.last_h_state = None
        self.last_flux_state = None

    def _run_pscan(self, A, X):
        orig_shape = X.shape
        if X.ndim == 4:
            B, T, C, D = X.shape
            X_5d = X.unsqueeze(-1)
            A_4d = A.squeeze(-1) if A.ndim == 5 else A
            Y_5d = pscan(A_4d, X_5d)
            return Y_5d.squeeze(-1)
        elif X.ndim == 5:
            return pscan(A, X)
        else:
            raise ValueError(f"Unsupported X shape: {X.shape}")

    def forward(self, x, dt):
        B, T, D, H, W = x.shape
        device = x.device
        dtype = x.dtype
        x_flat = x.reshape(B * T, D, H, W)
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_spatial = x_spatial.reshape(B, T, D * 2, H, W)
        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=2)
        x = x + torch.complex(x_s_re, x_s_im)
        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))
        A_flux, X_flux = self.prop.flux_tracker.get_operators(x_mean)
        flux_seq = self._run_pscan(
            A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)
        ).squeeze(-1)
        source_seq = self.prop.flux_tracker.project(flux_seq)
        source_expanded = (
            source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        )
        gate_seq = torch.sigmoid(
            self.prop.flux_tracker.gate_net(
                torch.cat([flux_seq.real, flux_seq.imag], dim=-1)
            )
        )
        gate_expanded = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)
        if dt.ndim == 1:
            dt_expanded = dt.unsqueeze(0).expand(B, T)
        else:
            dt_expanded = dt
        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        op_decay = op_decay.view(B, T, 1, 1, D)
        op_forcing = op_forcing.view(B, T, 1, 1, D)
        u_t = forcing * op_forcing
        A_time = op_decay.expand(B, T, H, W, D).reshape(B * H * W, T, 1, D)
        X_time = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, 1, D)
        Y_time = self._run_pscan(A_time, X_time)
        u_t = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)
        noise = self.prop.generate_stochastic_term(
            u_t.shape, dt_expanded, u_t.dtype, h_state=u_t
        )
        u_t = u_t + noise
        self.last_h_state = u_t[:, -1].detach().reshape(B * H * W, 1, D)
        self.last_flux_state = flux_seq[:, -1].detach()
        x_out = self.prop.basis.decode(u_t)
        x_out = x_out.permute(0, 1, 4, 2, 3)
        x_out_flat = x_out.reshape(B * T, D, H, W)
        x_out_real = torch.cat([x_out_flat.real, x_out_flat.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)
        delta = self.ffn(x_out_complex)
        x_out_complex = x_out_complex + delta
        x_out_complex = x_out_complex.reshape(B, T, D, H, W)
        return x + x_out_complex

    def forward_step(self, x_curr, h_prev, dt, flux_prev):
        B, D, H, W = x_curr.shape
        device = x_curr.device
        x_real = torch.cat([x_curr.real, x_curr.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=1)
        x_curr = x_curr + torch.complex(x_s_re, x_s_im)
        x_perm = x_curr.permute(0, 2, 3, 1)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(1, 2))
        h_tilde_next, flux_next = self.prop.forward_step(
            h_prev, x_eigen.reshape(B * H * W, D), x_mean, dt, flux_prev
        )
        h_out = h_tilde_next.reshape(B, H, W, 1, D)
        x_out = self.prop.basis.decode(h_out.squeeze(3))
        x_out = x_out.permute(0, 3, 1, 2)
        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)
        delta = self.ffn(x_out_complex)
        x_out_complex = x_out_complex + delta
        z_next = x_curr + x_out_complex
        return z_next, h_tilde_next, flux_next


class UniPhyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        expand=4,
        num_experts=8,
        depth=8,
        patch_size=16,
        img_height=64,
        img_width=64,
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
            out_channels, embed_dim, patch_size, img_height=img_height
        )

    def forward(self, x, dt):
        z = self.encoder(x)
        for block in self.blocks:
            z = block(z, dt)
        return self.decoder(z)

    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        z = self.encoder(x_cond)
        states = []
        for block in self.blocks:
            z = block(z, dt_cond)
            curr_flux = torch.zeros(
                x_cond.shape[0], block.dim, device=device, dtype=torch.cdouble
            )
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
    