import torch
import torch.nn as nn
import torch.nn.functional as F
from PScan import pscan
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyOps import TemporalPropagator, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork


class PhysicalState:
    def __init__(self, z, energy, momentum, flux):
        self.z = z
        self.energy = energy
        self.momentum = momentum
        self.flux = flux

    def detach(self):
        return PhysicalState(
            self.z.detach() if self.z is not None else None,
            self.energy.detach() if self.energy is not None else None,
            self.momentum.detach() if self.momentum is not None else None,
            self.flux.detach() if self.flux is not None else None,
        )

    def to(self, device):
        return PhysicalState(
            self.z.to(device) if self.z is not None else None,
            self.energy.to(device) if self.energy is not None else None,
            self.momentum.to(device) if self.momentum is not None else None,
            self.flux.to(device) if self.flux is not None else None,
        )


class EnergyProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.energy_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, target_energy):
        if z.is_complex():
            current_energy = (z.abs() ** 2).mean(dim=(-2, -1))
        else:
            current_energy = (z ** 2).mean(dim=(-2, -1))

        energy_ratio = target_energy / (current_energy + 1e-8)
        energy_ratio = torch.clamp(energy_ratio, 0.5, 2.0)

        if z.ndim == 5:
            scale_factor = energy_ratio.unsqueeze(-1).unsqueeze(-1).sqrt()
        else:
            scale_factor = energy_ratio.unsqueeze(-1).unsqueeze(-1).sqrt()

        z_scaled = z * (1 + self.scale * (scale_factor - 1))

        return z_scaled


class MomentumAdvection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.advection_weight = nn.Parameter(torch.tensor(0.1))
        self.momentum_proj = nn.Linear(dim * 2, dim * 2)

    def forward(self, z, momentum, dt):
        if z.ndim == 5:
            B, T, D, H, W = z.shape
            z_mean = z.mean(dim=(-2, -1))

            if momentum.ndim == 2:
                momentum = momentum.unsqueeze(1).expand(B, T, D)

            mom_cat = torch.cat([momentum.real, momentum.imag], dim=-1)
            mom_update = self.momentum_proj(mom_cat)
            mom_re, mom_im = torch.chunk(mom_update, 2, dim=-1)
            momentum_new = torch.complex(mom_re, mom_im)

            advection_term = momentum_new.unsqueeze(-1).unsqueeze(-1) * self.advection_weight
            z = z + advection_term * dt.view(1, T, 1, 1, 1)

            momentum_out = 0.9 * momentum + 0.1 * z_mean
        else:
            B, D, H, W = z.shape
            z_mean = z.mean(dim=(-2, -1))

            mom_cat = torch.cat([momentum.real, momentum.imag], dim=-1)
            mom_update = self.momentum_proj(mom_cat)
            mom_re, mom_im = torch.chunk(mom_update, 2, dim=-1)
            momentum_new = torch.complex(mom_re, mom_im)

            advection_term = momentum_new.unsqueeze(-1).unsqueeze(-1) * self.advection_weight
            z = z + advection_term * dt

            momentum_out = 0.9 * momentum + 0.1 * z_mean

        return z, momentum_out


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

        self.energy_proj = EnergyProjection(dim)
        self.momentum_advection = MomentumAdvection(dim)

        self.last_h_state = None
        self.last_flux_state = None

    def _run_pscan(self, A, X):
        return pscan(A, X)

    def _compute_energy(self, z):
        if z.is_complex():
            return (z.abs() ** 2).mean(dim=(-2, -1))
        return (z ** 2).mean(dim=(-2, -1))

    def _compute_momentum(self, z):
        return z.mean(dim=(-2, -1))

    def forward(self, state, dt):
        if isinstance(state, PhysicalState):
            x = state.z
            energy = state.energy
            momentum = state.momentum
            flux = state.flux
        else:
            x = state
            energy = self._compute_energy(x)
            momentum = self._compute_momentum(x)
            flux = None

        B, T, D, H, W = x.shape

        x_flat = x.reshape(B * T, D, H, W)
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)

        x_spatial = x_spatial.reshape(B, T, D * 2, H, W)
        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=2)
        x = x + torch.complex(x_s_re, x_s_im)

        if dt.ndim == 1:
            dt_expanded = dt.unsqueeze(0).expand(B, T)
        else:
            dt_expanded = dt

        x, momentum = self.momentum_advection(x, momentum, dt_expanded)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))

        A_flux, X_flux = self.prop.flux_tracker.get_operators(x_mean)
        flux_seq = self._run_pscan(
            A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)
        ).squeeze(-1)

        source_seq = self.prop.flux_tracker.project(flux_seq)
        source_expanded = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        gate_seq = torch.sigmoid(
            self.prop.flux_tracker.gate_net(
                torch.cat([flux_seq.real, flux_seq.imag], dim=-1)
            )
        )
        gate_expanded = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)

        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        op_decay = op_decay.view(B, T, 1, 1, D)
        op_forcing = op_forcing.view(B, T, 1, 1, D)

        u_t = forcing * op_forcing

        A_time = op_decay.expand(B, T, H, W, D).permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)
        X_time = u_t.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)

        Y_time = self._run_pscan(A_time, X_time)

        u_out = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        self.last_h_state = u_out[:, -1].detach().reshape(B * H * W, 1, D)
        self.last_flux_state = flux_seq[:, -1].detach()

        dt_for_noise = dt_expanded.view(B, T, 1, 1, 1).expand(B, T, H, W, 1)
        noise = self.prop.generate_stochastic_term(
            u_out.shape, dt_for_noise, u_out.dtype, h_state=u_out
        )
        u_out = u_out + noise

        x_out = self.prop.basis.decode(u_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)

        x_out = self.energy_proj(x_out, energy)

        x_out_flat = x_out.reshape(B * T, D, H, W)
        x_out_real = torch.cat([x_out_flat.real, x_out_flat.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        delta = self.ffn(x_out_complex)
        x_out_complex = x_out_complex + delta
        x_out_complex = x_out_complex.reshape(B, T, D, H, W)

        z_out = x + x_out_complex

        energy_out = self._compute_energy(z_out)
        momentum_out = momentum

        return PhysicalState(z_out, energy_out, momentum_out, flux_seq[:, -1])

    def forward_step(self, state, h_prev, dt, flux_prev):
        if isinstance(state, PhysicalState):
            x_curr = state.z
            energy = state.energy
            momentum = state.momentum
        else:
            x_curr = state
            energy = self._compute_energy(x_curr.unsqueeze(1)).squeeze(1)
            momentum = self._compute_momentum(x_curr.unsqueeze(1)).squeeze(1)

        B, D, H, W = x_curr.shape
        device = x_curr.device

        x_real = torch.cat([x_curr.real, x_curr.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)

        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=1)
        x_curr = x_curr + torch.complex(x_s_re, x_s_im)

        x_curr, momentum = self.momentum_advection(x_curr, momentum, dt)

        x_perm = x_curr.permute(0, 2, 3, 1)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(1, 2))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=device, dtype=x_curr.dtype)

        flux_next, source, gate = self.prop.flux_tracker.forward_step(flux_prev, x_mean)

        source_expanded = source.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        gate_expanded = gate.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)

        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)

        op_decay, op_forcing = self.prop.get_transition_operators(dt)

        h_prev_reshaped = h_prev.reshape(B, H, W, D)
        h_next = h_prev_reshaped * op_decay + forcing * op_forcing

        dt_for_noise = torch.as_tensor(dt, device=device).view(1, 1, 1, 1)
        noise = self.prop.generate_stochastic_term(
            h_next.shape, dt_for_noise, h_next.dtype, h_state=h_next
        )
        h_next = h_next + noise

        x_out = self.prop.basis.decode(h_next)
        x_out = x_out.permute(0, 3, 1, 2)

        x_out = self.energy_proj(x_out.unsqueeze(1), energy.unsqueeze(1)).squeeze(1)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        delta = self.ffn(x_out_complex)
        x_out_complex = x_out_complex + delta

        z_next = x_curr + x_out_complex
        h_next_flat = h_next.reshape(B * H * W, 1, D)

        energy_out = self._compute_energy(z_next.unsqueeze(1)).squeeze(1)

        state_out = PhysicalState(z_next, energy_out, momentum, flux_next)

        return state_out, h_next_flat, flux_next


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

        B, T, D, H, W = z.shape
        energy = (z.abs() ** 2).mean(dim=(-2, -1)) if z.is_complex() else (z ** 2).mean(dim=(-2, -1))
        momentum = z.mean(dim=(-2, -1))
        flux = torch.zeros(B, D, device=z.device, dtype=z.dtype)

        state = PhysicalState(z, energy, momentum, flux)

        for block in self.blocks:
            state = block(state, dt)

        return self.decoder(state.z)

    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        z = self.encoder(x_cond)

        B, T, D, H, W = z.shape
        energy = (z.abs() ** 2).mean(dim=(-2, -1)) if z.is_complex() else (z ** 2).mean(dim=(-2, -1))
        momentum = z.mean(dim=(-2, -1))
        flux = torch.zeros(B, D, device=device, dtype=z.dtype)

        state = PhysicalState(z, energy, momentum, flux)

        for block in self.blocks:
            state = block(state, dt_cond)

        states = []
        for block in self.blocks:
            h_state = block.last_h_state if block.last_h_state is not None else torch.zeros(
                B * block.img_height * block.img_width, 1, block.dim,
                device=device, dtype=z.dtype
            )
            flux_state = block.last_flux_state if block.last_flux_state is not None else torch.zeros(
                B, block.dim, device=device, dtype=z.dtype
            )
            states.append((h_state.to("cpu"), flux_state.to("cpu")))
            block.last_h_state = None
            block.last_flux_state = None

        z_curr = state.z[:, -1].detach()
        energy_curr = state.energy[:, -1].detach()
        momentum_curr = state.momentum[:, -1].detach() if state.momentum.ndim > 2 else state.momentum.detach()
        flux_curr = state.flux.detach()

        del z, state
        predictions = []

        for k in range(k_steps):
            dt_k = dt_future[:, k] if dt_future.ndim > 1 else dt_future[k] if dt_future.ndim == 1 else dt_future

            current_state = PhysicalState(z_curr, energy_curr, momentum_curr, flux_curr)
            new_states = []

            for i, block in enumerate(self.blocks):
                h_prev = states[i][0].to(device, non_blocking=True)
                flux_prev = states[i][1].to(device, non_blocking=True)

                current_state, h_next, flux_next = block.forward_step(
                    current_state, h_prev, dt_k, flux_prev
                )

                new_states.append((
                    h_next.to("cpu", non_blocking=True),
                    flux_next.to("cpu", non_blocking=True),
                ))

                del h_prev

            states = new_states
            z_curr = current_state.z
            energy_curr = current_state.energy
            momentum_curr = current_state.momentum
            flux_curr = current_state.flux

            pred = self.decoder(z_curr.unsqueeze(1)).squeeze(1).to("cpu", non_blocking=True)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)
    