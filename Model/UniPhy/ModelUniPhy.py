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
        self.last_h_state = None
        self.last_flux_state = None

    def _spatial_process(self, x_curr):
        if x_curr.ndim == 5:
            B, T, D, H, W = x_curr.shape
            x_flat = x_curr.reshape(B * T, D, H, W)
        else:
            x_flat = x_curr
            B, D, H, W = x_flat.shape
            T = None

        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)

        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=1)
        x_out = x_flat + torch.complex(x_s_re, x_s_im)

        if T is not None:
            x_out = x_out.reshape(B, T, D, H, W)

        return x_out

    def _temporal_decode(self, x_out):
        if x_out.ndim == 5:
            B, T, D, H, W = x_out.shape
            x_flat = x_out.reshape(B * T, D, H, W)
        else:
            x_flat = x_out
            T = None

        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_temporal(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)

        delta = self.ffn(x_complex)
        x_complex = x_complex + delta

        if T is not None:
            B = x_out.shape[0]
            x_complex = x_complex.reshape(B, T, -1, x_complex.shape[2], x_complex.shape[3])

        return x_complex

    def forward(self, x, h_prev, dt, flux_prev):
        B, T, D, H, W = x.shape
        device = x.device

        x = self._spatial_process(x)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=device, dtype=x.dtype)

        flux_list = []
        source_list = []
        gate_list = []
        current_flux = flux_prev

        for t in range(T):
            x_mean_t = x_mean[:, t]
            new_flux, source, gate = self.prop.flux_tracker.forward_step(current_flux, x_mean_t)
            flux_list.append(new_flux)
            source_list.append(source)
            gate_list.append(gate)
            current_flux = new_flux

        flux_seq = torch.stack(flux_list, dim=1)
        source_seq = torch.stack(source_list, dim=1)
        gate_seq = torch.stack(gate_list, dim=1)

        source_expanded = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        gate_expanded = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)

        if dt.ndim == 1:
            dt_expanded = dt.unsqueeze(0).expand(B, T)
        else:
            dt_expanded = dt

        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        op_decay_expanded = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing_expanded = op_forcing.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        
        u_t = forcing * op_forcing_expanded
        A_time = op_decay_expanded.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X_time = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        
        Y_time = pscan(A_time, X_time)
        u_out = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        if h_prev is not None:
            h_prev_reshaped = h_prev.reshape(B, H, W, D)
            h_init_contrib = h_prev_reshaped.unsqueeze(1) * op_decay_expanded
            u_out = u_out + h_init_contrib

        self.last_h_state = u_out[:, -1].detach().reshape(B * H * W, 1, D)
        self.last_flux_state = flux_seq[:, -1].detach()

        x_out = self.prop.basis.decode(u_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)

        x_out = self._temporal_decode(x_out)

        z_out = x + x_out

        return z_out, self.last_h_state, self.last_flux_state

    def forward_step(self, x_curr, h_prev, dt, flux_prev):
        B, D, H, W = x_curr.shape
        device = x_curr.device

        x_curr = self._spatial_process(x_curr)

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

        x_out = self._temporal_decode(x_out)

        z_next = x_curr + x_out
        h_next_flat = h_next.reshape(B * H * W, 1, D)

        self.last_h_state = h_next_flat.detach()
        self.last_flux_state = flux_next.detach()

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

        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        self.h_patches = (img_height + pad_h) // patch_size
        self.w_patches = (img_width + pad_w) // patch_size

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
            ensemble_size=1,
            img_height=img_height,
            img_width=img_width,
        )

        self.blocks = nn.ModuleList([
            UniPhyBlock(
                dim=embed_dim,
                expand=expand,
                num_experts=num_experts,
                img_height=self.h_patches,
                img_width=self.w_patches,
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
            base_dtype = z.dtype
        else:
            base_dtype = torch.complex64

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
            base_dtype = z_curr.dtype
        else:
            base_dtype = torch.complex64

        states = self._init_states(B, device, base_dtype)
        predictions = []

        for k in range(num_steps):
            dt_k = dt_list[k] if k < len(dt_list) else dt_list[-1]

            z_next = z_curr
            new_states = []

            for i, block in enumerate(self.blocks):
                h_prev, flux_prev = states[i]

                z_next, h_next, flux_next = block.forward_step(z_next, h_prev, dt_k, flux_prev)

                new_states.append((h_next.detach(), flux_next.detach()))

            states = new_states
            z_curr = z_next

            pred = self.decoder(z_curr.unsqueeze(1)).squeeze(1)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def forecast(self, x_cond, dt_cond, k_steps, dt_future):
        device = next(self.parameters()).device
        B = x_cond.shape[0]

        z = self.encoder(x_cond)

        if z.dtype.is_complex:
            base_dtype = z.dtype
        else:
            base_dtype = torch.complex64

        h_prev = None
        flux_prev = None

        for i, block in enumerate(self.blocks):
            if i == 0:
                h_init = torch.zeros(
                    B * self.h_patches * self.w_patches, 1, self.embed_dim,
                    device=device, dtype=base_dtype
                )
                flux_init = torch.zeros(B, self.embed_dim, device=device, dtype=base_dtype)
                z, h_prev, flux_prev = block(z, h_init, dt_cond, flux_init)
            else:
                z, h_prev, flux_prev = block(z, h_prev, dt_cond, flux_prev)

        states = []
        for block in self.blocks:
            h_state = block.last_h_state if block.last_h_state is not None else torch.zeros(
                B * self.h_patches * self.w_patches, 1, self.embed_dim,
                device=device, dtype=base_dtype
            )
            flux_state = block.last_flux_state if block.last_flux_state is not None else torch.zeros(
                B, self.embed_dim, device=device, dtype=base_dtype
            )
            states.append((h_state, flux_state))

        z_curr = z[:, -1].detach()
        del z

        predictions = []

        for k in range(k_steps):
            if dt_future.ndim > 1:
                dt_k = dt_future[:, k]
            elif dt_future.ndim == 1:
                dt_k = dt_future[k]
            else:
                dt_k = dt_future

            z_next = z_curr
            new_states = []

            for i, block in enumerate(self.blocks):
                h_prev, flux_prev = states[i]

                z_next, h_next, flux_next = block.forward_step(z_next, h_prev, dt_k, flux_prev)

                new_states.append((h_next.detach(), flux_next.detach()))

            states = new_states
            z_curr = z_next

            pred = self.decoder(z_curr.unsqueeze(1)).squeeze(1)
            predictions.append(pred)

            if k % 10 == 0:
                torch.cuda.empty_cache()

        return torch.stack(predictions, dim=1)

    def get_initial_states(self, batch_size, device):
        return self._init_states(batch_size, device, torch.complex64)

    def reset_states(self):
        for block in self.blocks:
            block.last_h_state = None
            block.last_flux_state = None
            