import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from PScan import pscan
from UniPhyIO import UniPhyEncoder, UniPhyDecoder
from UniPhyOps import LearnableSpectralBasis, GlobalFluxTracker
from UniPhyFFN import UniPhyFeedForwardNetwork


class StochasticPropagator(nn.Module):
    def __init__(
        self,
        dim,
        dt_ref=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.sde_mode = sde_mode
        self.init_noise_scale = init_noise_scale
        self.max_growth_rate = max_growth_rate

        self.basis = LearnableSpectralBasis(dim)
        self.flux_tracker = GlobalFluxTracker(dim)

        self.lambda_re = nn.Parameter(torch.randn(dim) * 0.1 - 0.5)
        self.lambda_im = nn.Parameter(torch.randn(dim) * 0.1)

        self.noise_scale_base = nn.Parameter(torch.ones(dim) * init_noise_scale)
        self.noise_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def get_eigenvalues(self):
        re_part = -F.softplus(self.lambda_re)
        im_part = self.lambda_im
        return torch.complex(re_part, im_part)

    def get_transition_operators(self, dt):
        eigenvalues = self.get_eigenvalues()
        dt_ratio = dt / self.dt_ref

        if dt_ratio.ndim == 0:
            dt_ratio = dt_ratio.unsqueeze(0)
        while dt_ratio.ndim < 2:
            dt_ratio = dt_ratio.unsqueeze(-1)

        dt_ratio = dt_ratio.to(eigenvalues.device)
        exp_term = torch.exp(eigenvalues.unsqueeze(0).unsqueeze(0) * dt_ratio.unsqueeze(-1))

        decay_op = exp_term
        forcing_op = (1.0 - exp_term) / (eigenvalues.unsqueeze(0).unsqueeze(0) + 1e-8)

        return decay_op, forcing_op

    def generate_stochastic_term(self, shape, dt, dtype, h_state=None):
        if self.sde_mode == "ode":
            return torch.zeros(shape, device=self.lambda_re.device, dtype=dtype)

        device = self.lambda_re.device
        dt_val = dt.mean().item() if isinstance(dt, torch.Tensor) else dt

        noise_real = torch.randn(shape, device=device)
        noise_imag = torch.randn(shape, device=device)
        noise = torch.complex(noise_real, noise_imag)

        base_scale = self.noise_scale_base.view(1, 1, 1, -1)
        scaled_noise = noise * base_scale * math.sqrt(dt_val)

        if h_state is not None:
            h_mag = h_state.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            h_real = h_state.real.mean(dim=-1)
            gate = self.noise_gate(h_real).unsqueeze(-1)
            max_noise = h_mag * self.max_growth_rate
            scaled_noise = scaled_noise * gate
            noise_mag = scaled_noise.abs()
            scale_factor = torch.where(
                noise_mag > max_noise,
                max_noise / noise_mag.clamp(min=1e-8),
                torch.ones_like(noise_mag),
            )
            scaled_noise = scaled_noise * scale_factor

        return scaled_noise

    def forward(self, x_input, h_prev, flux_prev, dt, source=None):
        B, H, W, D = x_input.shape
        device = x_input.device
        spatial_size = H * W

        x_tilde = self.basis.encode(x_input)

        h_prev_latent = h_prev.view(B, spatial_size, 1, D)
        x_tilde_flat = x_tilde.view(B, spatial_size, 1, D)

        if source is not None:
            source_tilde = self.basis.encode(source)
            source_flat = source_tilde.view(B, spatial_size, 1, D)
        else:
            source_flat = torch.zeros_like(x_tilde_flat)

        flux_next, gate = self.flux_tracker(x_tilde_flat, flux_prev.unsqueeze(1))
        flux_next = flux_next.squeeze(1)

        gate_expanded = gate.view(B, 1, 1, D).expand(B, spatial_size, 1, D)
        forcing_term = x_tilde_flat * gate_expanded + source_flat * (1.0 - gate_expanded)

        op_decay, op_forcing = self.get_transition_operators(dt)
        op_decay = op_decay.view(B, 1, 1, D)
        op_forcing = op_forcing.view(B, 1, 1, D)

        h_next_latent = h_prev_latent * op_decay + forcing_term * op_forcing

        return h_next_latent.view(B, H, W, D), flux_next


class UniPhyBlock(nn.Module):
    def __init__(
        self,
        dim,
        expand=4,
        num_experts=8,
        dt_ref=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref

        self.prop = StochasticPropagator(
            dim=dim,
            dt_ref=dt_ref,
            sde_mode=sde_mode,
            init_noise_scale=init_noise_scale,
            max_growth_rate=max_growth_rate,
        )

        self.norm_spatial = nn.LayerNorm(dim * 2)
        self.norm_temporal = nn.LayerNorm(dim * 2)
        self.ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts)

    def _run_pscan(self, A, X):
        return pscan(A, X)

    def forward(self, x, h_prev, dt, flux_prev):
        B, T, D, H, W = x.shape
        device = x.device

        x_seq = x.permute(0, 1, 3, 4, 2).contiguous()

        x_eigen = self.prop.basis.encode(x_seq)

        source_seq = x_eigen.clone()
        flux_seq, gate_seq = self.prop.flux_tracker(
            x_eigen.view(B * T, H * W, 1, D),
            flux_prev.unsqueeze(1).expand(B, T, -1).reshape(B * T, D),
        )
        flux_seq = flux_seq.view(B, T, D)
        gate_seq = gate_seq.view(B, T, D)

        gate_expanded = gate_seq.view(B, T, 1, 1, D).expand(B, T, H, W, D)
        source_expanded = source_seq

        forcing = x_eigen * gate_expanded + source_expanded * (1.0 - gate_expanded)

        if dt.ndim == 1:
            dt_expanded = dt.unsqueeze(0).expand(B, T)
        else:
            dt_expanded = dt

        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        op_decay = op_decay.view(B, T, 1, 1, D)
        op_forcing = op_forcing.view(B, T, 1, 1, D)

        u_t = forcing * op_forcing

        A_time = (
            op_decay.expand(B, T, H, W, D)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(B * H * W, T, D, 1)
        )
        X_time = (
            u_t.permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(B * H * W, T, D, 1)
        )

        Y_time = self._run_pscan(A_time, X_time)

        u_out = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        self.last_h_state = u_out[:, -1].detach().clone().reshape(B * H * W, 1, D)
        self.last_flux_state = flux_seq[:, -1].detach().clone()

        x_out = self.prop.basis.decode(u_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=2)
        x_out_flat = x_out_real.reshape(B * T, D * 2, H, W)
        x_out_norm = self.norm_spatial(x_out_flat.permute(0, 2, 3, 1))
        x_out_norm = x_out_norm.permute(0, 3, 1, 2).reshape(B, T, D * 2, H, W)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=2)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        x_out_ffn = x_out_complex.reshape(B * T, D, H, W)
        delta = self.ffn(x_out_ffn)
        delta = delta.reshape(B, T, D, H, W)
        x_out_complex = x_out_complex + delta

        z_out = x + x_out_complex
        h_next = self.last_h_state
        flux_next = self.last_flux_state

        return z_out, h_next, flux_next

    def forward_step(self, x_curr, h_prev, dt, flux_prev):
        B, D, H, W = x_curr.shape
        device = x_curr.device

        x_spatial = x_curr.permute(0, 2, 3, 1).contiguous()

        x_eigen = self.prop.basis.encode(x_spatial)

        h_prev_reshaped = h_prev.view(B, H, W, D)

        flux_next, gate = self.prop.flux_tracker(
            x_eigen.view(B, H * W, 1, D),
            flux_prev,
        )
        flux_next = flux_next.squeeze(1)
        gate = gate.squeeze(1)

        source = x_eigen.clone()
        gate_expanded = gate.view(B, 1, 1, D).expand(B, H, W, D)
        forcing = x_eigen * gate_expanded + source * (1.0 - gate_expanded)

        dt_tensor = torch.as_tensor(dt, device=device, dtype=x_curr.dtype)
        if dt_tensor.is_complex():
            dt_tensor = dt_tensor.real

        op_decay, op_forcing = self.prop.get_transition_operators(dt_tensor)
        op_decay = op_decay.view(1, 1, 1, D).expand(B, H, W, D)
        op_forcing = op_forcing.view(1, 1, 1, D).expand(B, H, W, D)

        h_next = h_prev_reshaped * op_decay + forcing * op_forcing

        dt_for_noise = dt_tensor.view(1, 1, 1, 1)
        noise = self.prop.generate_stochastic_term(
            h_next.shape, dt_for_noise, h_next.dtype, h_state=h_next
        )
        h_next = h_next + noise

        x_out = self.prop.basis.decode(h_next)
        x_out = x_out.permute(0, 3, 1, 2)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1))
        x_out_norm = x_out_norm.permute(0, 3, 1, 2)
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)

        delta = self.ffn(x_out_complex)
        x_out_complex = x_out_complex + delta

        z_next = x_curr + x_out_complex
        h_next_flat = h_next.reshape(B * H * W, 1, D)

        return z_next, h_next_flat, flux_next


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.dt_ref = dt_ref

        self.h_patches = (img_height + patch_size - 1) // patch_size
        self.w_patches = (img_width + patch_size - 1) // patch_size

        self.encoder = UniPhyEncoder(
            in_ch=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )

        self.decoder = UniPhyDecoder(
            out_ch=out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_height=img_height,
            img_width=img_width,
        )

        self.blocks = nn.ModuleList([
            UniPhyBlock(
                dim=embed_dim,
                expand=expand,
                num_experts=num_experts,
                dt_ref=dt_ref,
                sde_mode=sde_mode,
                init_noise_scale=init_noise_scale,
                max_growth_rate=max_growth_rate,
            )
            for _ in range(depth)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_states(self, batch_size, device, dtype):
        H_p = self.h_patches
        W_p = self.w_patches
        D = self.embed_dim

        states = []
        for _ in range(self.depth):
            h_init = torch.zeros(batch_size * H_p * W_p, 1, D, device=device, dtype=dtype)
            flux_init = torch.zeros(batch_size, D, device=device, dtype=dtype)
            states.append((h_init, flux_init))

        return states

    def forward(self, x, dt):
        B, T, C, H, W = x.shape
        device = x.device

        z = self.encoder(x)

        if z.dtype.is_complex:
            base_dtype = torch.complex64
        else:
            base_dtype = z.dtype

        h_prev = None
        flux_prev = None

        for i, block in enumerate(self.blocks):
            if i == 0:
                h_init = torch.zeros(
                    B * self.h_patches * self.w_patches, 1, self.embed_dim,
                    device=device, dtype=base_dtype
                )
                flux_init = torch.zeros(B, self.embed_dim, device=device, dtype=base_dtype)
                z, h_prev, flux_prev = block(z, h_init, dt, flux_init)
            else:
                z, h_prev, flux_prev = block(z, h_prev, dt, flux_prev)

        out = self.decoder(z)

        return out

    def forward_rollout(self, x_init, dt_list, num_steps=None):
        if num_steps is None:
            num_steps = len(dt_list)

        B, C, H, W = x_init.shape
        device = x_init.device

        z_curr = self.encoder(x_init.unsqueeze(1)).squeeze(1)

        if z_curr.dtype.is_complex:
            base_dtype = torch.complex64
        else:
            base_dtype = z_curr.dtype

        states = self._init_states(B, "cpu", base_dtype)
        predictions = []

        for k in range(num_steps):
            dt_k = dt_list[k] if k < len(dt_list) else dt_list[-1]

            z_next = z_curr.clone()
            new_states = []

            for i, block in enumerate(self.blocks):
                h_prev = states[i][0].to(device, non_blocking=True)
                flux_prev = states[i][1].to(device, non_blocking=True)

                z_next, h_next, flux_next = block.forward_step(
                    z_next, h_prev, dt_k, flux_prev
                )

                new_states.append((
                    h_next.detach().clone().to("cpu", non_blocking=True),
                    flux_next.detach().clone().to("cpu", non_blocking=True),
                ))

                del h_prev, flux_prev

            states = new_states
            z_curr = z_next.clone()

            pred = self.decoder(z_curr.unsqueeze(1)).squeeze(1)
            pred_cpu = pred.detach().clone().to("cpu", non_blocking=True)
            predictions.append(pred_cpu)

            del pred

            if k % 10 == 0:
                torch.cuda.empty_cache()

        result = torch.stack(predictions, dim=1)

        return result

    def get_initial_states(self, batch_size, device):
        return self._init_states(batch_size, device, torch.complex64)

    def reset_states(self):
        for block in self.blocks:
            if hasattr(block, "last_h_state"):
                block.last_h_state = None
            if hasattr(block, "last_flux_state"):
                block.last_flux_state = None
                