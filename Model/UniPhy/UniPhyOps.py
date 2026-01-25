import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianCliffordConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        img_height,
        img_width
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = in_channels

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )

        self.log_metric_param = nn.Parameter(torch.zeros(1, 1, img_height, img_width))

        self.metric_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Tanh()
        )

        self.viscosity_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        laplacian_init = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        self.register_buffer(
            "laplacian_kernel",
            laplacian_init.reshape(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        )

        self.metric_scale = nn.Parameter(torch.tensor(0.1))
        self.viscosity_scale = nn.Parameter(torch.tensor(0.01))
        self.diffusion_scale = nn.Parameter(torch.tensor(0.01))
        self.dispersion_scale = nn.Parameter(torch.tensor(0.01))

        self.anti_diffusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        metric = torch.exp(self.log_metric_param * self.metric_scale)
        if metric.shape[-2:] != (H, W):
            metric = F.interpolate(metric, size=(H, W), mode="bilinear", align_corners=False)

        refine = self.metric_refiner(x)
        metric = metric * (1.0 + refine)

        scale = torch.sqrt(metric + 1e-8)
        inv_scale = 1.0 / scale

        x_scaled = x * scale

        diffusion_term = F.conv2d(
            x_scaled, self.laplacian_kernel, padding=1, groups=self.groups
        )

        local_viscosity = self.viscosity_gate(x) * inv_scale

        diffusion = diffusion_term * local_viscosity * self.viscosity_scale * self.diffusion_scale

        anti_diff_weight = self.anti_diffusion_gate(x)
        dispersion = -diffusion_term * self.dispersion_scale

        x_evolved = x_scaled + diffusion - anti_diff_weight * diffusion * 0.5 + dispersion

        x_final = x_evolved * inv_scale
        out = self.conv(x_final)

        return out

class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        q_re, _ = torch.linalg.qr(torch.randn(dim, dim))
        q_im, _ = torch.linalg.qr(torch.randn(dim, dim))

        self.u_raw_re = nn.Parameter(q_re * 0.1)
        self.u_raw_im = nn.Parameter(q_im * 0.1)
        self.v_raw_re = nn.Parameter(q_re.T * 0.1)
        self.v_raw_im = nn.Parameter(q_im.T * 0.1)
        self.log_sigma = nn.Parameter(torch.zeros(dim))

        n = torch.arange(dim)
        k = torch.arange(dim).reshape(-1, 1)
        dft_matrix = torch.exp(-2j * torch.pi * n * k / dim) / (dim ** 0.5)
        self.register_buffer("dft_basis", dft_matrix)
        self.dft_weight = nn.Parameter(torch.tensor(0.2))

    def _cayley_orthogonalize(self, raw_re, raw_im):
        A = torch.complex(raw_re, raw_im)
        A_skew = A - A.T.conj()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        Q = torch.linalg.solve(I + A_skew, I - A_skew)
        return Q

    def _get_basis(self):
        U = self._cayley_orthogonalize(self.u_raw_re, self.u_raw_im)
        V = self._cayley_orthogonalize(self.v_raw_re, self.v_raw_im)
        S = torch.diag(torch.exp(self.log_sigma).to(U.dtype))
        return U, S, V

    def encode(self, x):
        U, S_mat, V = self._get_basis()
        S_diag = torch.diag(S_mat)

        learned_path = torch.einsum("...d, de -> ...e", x, V) * S_diag

        x_complex = x if x.is_complex() else torch.complex(x, torch.zeros_like(x))
        dft_path = torch.einsum("...d, de -> ...e", x_complex, self.dft_basis)

        alpha = torch.sigmoid(self.dft_weight)
        return learned_path * (1 - alpha) + dft_path * alpha

    def decode(self, h):
        U, S_mat, V = self._get_basis()
        S_diag = torch.diag(S_mat)
        alpha = torch.sigmoid(self.dft_weight)
        
        W_learned = V * S_diag.unsqueeze(0) * (1 - alpha)
        W_dft = self.dft_basis * alpha
        W_total = W_learned + W_dft
        
        if h.is_complex() and not W_total.is_complex():
            W_total = W_total.to(h.dtype)
            
        x_rec = torch.linalg.solve(W_total.T, h.unsqueeze(-1)).squeeze(-1)
        return x_rec

class GlobalFluxTracker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.decay_re = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.decay_im = nn.Parameter(torch.randn(dim) * 0.1)

        self.input_mix = nn.Linear(dim * 2, dim * 2)
        self.output_proj = nn.Linear(dim * 2, dim * 2)

        nn.init.xavier_uniform_(self.input_mix.weight)
        nn.init.zeros_(self.input_mix.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def _get_decay(self):
        return torch.complex(torch.sigmoid(self.decay_re), self.decay_im)

    def get_operators(self, x_mean_seq):
        B, T, D = x_mean_seq.shape
        decay = self._get_decay()

        x_flat = x_mean_seq.reshape(B * T, D)
        x_cat = torch.cat([x_flat.real, x_flat.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)

        X = torch.complex(x_re, x_im).reshape(B, T, D).contiguous()
        A = decay.unsqueeze(0).unsqueeze(0).expand(B, T, D).contiguous()

        return A, X

    def project(self, state):
        s_cat = torch.cat([state.real, state.imag], dim=-1)
        out = self.output_proj(s_cat)
        out_re, out_im = torch.chunk(out, 2, dim=-1)
        return torch.complex(out_re, out_im)

    def forward_step(self, prev_state, x_t):
        decay = self._get_decay()

        x_cat = torch.cat([x_t.real, x_t.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        update = torch.complex(x_re, x_im)

        new_state = prev_state * decay + update
        source = self.project(new_state)

        s_cat = torch.cat([new_state.real, new_state.imag], dim=-1)
        gate = self.gate_net(s_cat)

        return new_state, source, gate

class TemporalPropagator(nn.Module):
    def __init__(
        self,
        dim,
        dt_ref=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3
    ):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.sde_mode = sde_mode
        self.max_growth_rate = max_growth_rate

        if sde_mode == "sde":
            self.base_noise = nn.Parameter(torch.tensor(float(init_noise_scale)))
            self.uncertainty_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
                nn.Softplus()
            )
        else:
            self.register_buffer("base_noise", torch.tensor(0.0))
            self.uncertainty_net = None

        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim)

        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.randn(dim) * 1.0)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)

    def _get_effective_lambda(self):
        l_phys = torch.complex(-torch.exp(self.ld), self.lf)
        l_law = torch.complex(self.law_re, self.law_im)
        l_total = l_phys + l_law

        bounded_re = torch.tanh(l_total.real) * self.max_growth_rate
        return torch.complex(bounded_re, l_total.imag)

    def get_transition_operators(self, dt):
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        lam = self._get_effective_lambda()

        if dt.ndim == 0:
            dt = dt.unsqueeze(0)

        dt_ratio = dt / self.dt_ref
        exp_arg = lam.unsqueeze(0) * dt_ratio.unsqueeze(-1)
        decay = torch.exp(exp_arg)
        forcing = (decay - 1) / (lam.unsqueeze(0) + 1e-8)

        return decay, forcing

    def generate_stochastic_term(self, target_shape, dt, dtype, h_state=None):
        if self.sde_mode != "sde":
            return torch.zeros(target_shape, device=self.ld.device, dtype=dtype)

        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        l_re = self._get_effective_lambda().real

        base_var = (
            (self.base_noise ** 2)
            * torch.expm1(2 * l_re * (dt / self.dt_ref).unsqueeze(-1))
            / (2 * l_re + 1e-8)
        )
        base_var = torch.clamp(base_var, min=0.0)

        if h_state is not None and self.uncertainty_net is not None:
            h_mean = h_state.mean(dim=1) if h_state.ndim > 2 else h_state
            h_input = h_mean.real if h_mean.is_complex() else h_mean
            local_uncertainty = self.uncertainty_net(h_input)

            while local_uncertainty.ndim < base_var.ndim:
                local_uncertainty = local_uncertainty.unsqueeze(1)

            dynamic_var = base_var * local_uncertainty
        else:
            dynamic_var = base_var

        std = torch.sqrt(dynamic_var).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std

    def forward_step(self, h_prev_latent, x_input, x_global_mean_encoded, dt, flux_state):
        x_tilde = self.basis.encode(x_input)

        if x_tilde.ndim == 2:
            x_tilde = x_tilde.unsqueeze(1)

        flux_next, source, gate = self.flux_tracker.forward_step(
            flux_state, x_global_mean_encoded
        )

        op_decay, op_forcing = self.get_transition_operators(dt)

        B = source.shape[0]
        total_batch = x_tilde.shape[0]
        D = x_tilde.shape[-1]
        spatial_size = total_batch // B

        source_expanded = (
            source.view(B, 1, 1, D)
            .expand(B, spatial_size, 1, D)
            .reshape(total_batch, 1, D)
        )

        gate_expanded = (
            gate.view(B, 1, 1, D)
            .expand(B, spatial_size, 1, D)
            .reshape(total_batch, 1, D)
        )

        forcing_term = x_tilde * gate_expanded + source_expanded * (1 - gate_expanded)
        h_tilde_next = h_prev_latent * op_decay + forcing_term * op_forcing

        h_tilde_next = h_tilde_next + self.generate_stochastic_term(
            h_tilde_next.shape, dt, h_tilde_next.dtype, h_state=h_tilde_next
        )

        return h_tilde_next, flux_next
    