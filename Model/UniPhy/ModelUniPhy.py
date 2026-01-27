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

        if is_5d:
            x_out = x_out.reshape(B, T, D, H, W)

        return x_out

    def _temporal_decode(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, D, H, W = x.shape
            x_flat = x.reshape(B * T, D, H, W)
        else:
            x_flat = x

        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_temporal(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_re, x_im = torch.chunk(x_norm, 2, dim=1)
        x_complex = torch.complex(x_re, x_im)

        delta = self.ffn(x_complex)
        x_out = x_complex + delta

        if is_5d:
            x_out = x_out.reshape(B, T, D, H, W)

        return x_out

    def forward(self, x, h_prev, dt, flux_prev):
        B, T, D, H, W = x.shape
        device = x.device

        x = self._spatial_process(x)

        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        x_mean = x_eigen.mean(dim=(2, 3))

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=device, dtype=x.dtype)

        A_flux, X_flux = self.prop.flux_tracker.get_scan_operators(x_mean)
        flux_seq = pscan(A_flux, X_flux)
        flux_seq = flux_seq.squeeze(-1)

        decay_val = self.prop.flux_tracker._get_decay() 
        decay_steps = decay_val.view(1, 1, D).expand(B, T, D)
        decay_cum = torch.cumprod(decay_steps, dim=1)
        prev_contribution = flux_prev.unsqueeze(1) * decay_cum
        flux_seq = flux_seq + prev_contribution

        source_seq, gate_seq = self.prop.flux_tracker.compute_output(flux_seq)
        flux_out = flux_seq[:, -1]

        source_exp = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        gate_exp = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        forcing = x_eigen * gate_exp + source_exp * (1 - gate_exp)

        if dt.ndim == 1:
            dt_exp = dt.unsqueeze(0).expand(B, T)
        else:
            dt_exp = dt

        op_decay, op_forcing = self.prop.get_transition_operators(dt_exp)
        op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

        u_t = forcing * op_forcing

        if self.prop.sde_mode == "sde":
            dt_noise = dt_exp.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            noise = self.prop.generate_stochastic_term(
                shape=u_t.shape,
                dt=dt_noise,
                dtype=u_t.dtype,
                h_state=x_eigen 
            )
            u_t = u_t + noise

        if h_prev is not None:
            h_reshaped = h_prev.reshape(B, H, W, D)
            h_contrib_t0 = h_reshaped * op_decay[:, 0]
            u_t_list = list(torch.unbind(u_t, dim=1))
            u_t_list[0] = u_t_list[0] + h_contrib_t0
            u_t = torch.stack(u_t_list, dim=1)

        A = op_decay.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)

        Y = pscan(A, X)

        u_out = Y.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        h_out = u_out[:, -1].reshape(B * H * W, 1, D)
        
        x_out = self.prop.basis.decode(u_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)
        x_out = self._temporal_decode(x_out)

        return x + x_out, h_out, flux_out

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

        if flux_prev is None:
            flux_prev = torch.zeros(B, D, device=device, dtype=x_curr.dtype)

        decay_val = self.prop.flux_tracker._get_decay()
        
        x_mean_flat = x_mean
        x_cat = torch.cat([x_mean_flat.real, x_mean_flat.imag], dim=-1)
        x_in = self.prop.flux_tracker.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        x_mixed = torch.complex(x_re, x_im)
        
        flux_next = flux_prev * decay_val + x_mixed
        source, gate = self.prop.flux_tracker.compute_output(flux_next)

        source_expanded = source.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        gate_expanded = gate.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)

        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)

        op_decay, op_forcing = self.prop.get_transition_operators(dt)
        
        if op_decay.ndim == 2:
            op_decay = op_decay.unsqueeze(1).unsqueeze(1)
        if op_forcing.ndim == 2:
            op_forcing = op_forcing.unsqueeze(1).unsqueeze(1)

        h_prev_reshaped = h_prev.reshape(B, H, W, D)
        
        noise = 0
        if self.prop.sde_mode == "sde":
            if dt.ndim == 1:
                dt_noise = dt.view(B, 1, 1, 1)
            elif dt.ndim == 0:
                dt_noise = dt
            else:
                dt_noise = dt.view(B, 1, 1, 1)

            noise = self.prop.generate_stochastic_term(
                shape=h_prev_reshaped.shape,
                dt=dt_noise,
                dtype=h_prev_reshaped.dtype,
                h_state=x_eigen 
            )

        h_next = h_prev_reshaped * op_decay + forcing * op_forcing + noise

        x_out = self.prop.basis.decode(h_next)
        x_out = x_out.permute(0, 3, 1, 2)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
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
        self.embed_dim = embed_dim
        self.depth = depth
        self.img_height = img_height
        self.img_width = img_width

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
            ensemble_size=10,
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

    def _init_states(self, B, device, dtype):
        H, W, D = self.h_patches, self.w_patches, self.embed_dim
        states = []
        for _ in range(self.depth):
            h = torch.zeros(B * H * W, 1, D, device=device, dtype=dtype)
            f = torch.zeros(B, D, device=device, dtype=dtype)
            states.append((h, f))
        return states

    def forward(self, x, dt, member_idx=None):
        B, T, C, H, W = x.shape
        device = x.device

        z = self.encoder(x)
        dtype = z.dtype if z.dtype.is_complex else torch.complex64

        states = self._init_states(B, device, dtype)

        for i, block in enumerate(self.blocks):
            h_prev, flux_prev = states[i]
            z, h_next, flux_next = block(z, h_prev, dt, flux_prev)
            states[i] = (h_next, flux_next)

        out = self.decoder(z, member_idx=member_idx)

        if out.shape[-2] != H or out.shape[-1] != W:
            out = out[..., :H, :W]

        return out

    @torch.no_grad()
    def forward_rollout(self, x_init, dt_list, num_steps=None):
        if num_steps is None:
            num_steps = len(dt_list)

        B = x_init.shape[0]
        device = x_init.device
        target_h, target_w = x_init.shape[-2:]

        z = self.encoder(x_init.unsqueeze(1)).squeeze(1)
        dtype = z.dtype if z.dtype.is_complex else torch.complex64

        states = self._init_states(B, device, dtype)
        preds = []

        for k in range(num_steps):
            dt_k = dt_list[k] if k < len(dt_list) else dt_list[-1]
            new_states = []
            
            z_curr_layer = z

            for i, block in enumerate(self.blocks):
                h_prev, flux_prev = states[i]
                z_curr_layer, h_next, flux_next = block.forward_step(z_curr_layer, h_prev, dt_k, flux_prev)
                new_states.append((h_next, flux_next))

            states = new_states
            
            pred = self.decoder(z_curr_layer.unsqueeze(1)).squeeze(1)

            if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
                pred = pred[..., :target_h, :target_w]

            preds.append(pred)

            if k < num_steps - 1:
                z = self.encoder(pred.unsqueeze(1)).squeeze(1)

        return torch.stack(preds, dim=1)
    