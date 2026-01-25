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

        self.dispersion_kernel = nn.Parameter(
            torch.randn(in_channels, 1, 3, 3) * 0.01
        )
        self.dispersion_scale = nn.Parameter(torch.tensor(0.01))

        self.anti_diffusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_out = self.conv(x)

        metric_field = self.log_metric_param.expand(x.size(0), -1, -1, -1)
        metric_refine = self.metric_refiner(x)
        metric_field = metric_field * (1 + metric_refine)

        laplacian = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.groups)
        viscosity_weight = self.viscosity_gate(x)
        diffusion = laplacian * viscosity_weight * self.viscosity_scale

        dispersion = F.conv2d(x, self.dispersion_kernel, padding=1, groups=self.groups)
        dispersion = dispersion * self.dispersion_scale

        anti_diff_weight = self.anti_diffusion_gate(x)
        dispersion = dispersion * anti_diff_weight

        metric_correction = metric_field * base_out * self.metric_scale

        return base_out + metric_correction + diffusion + dispersion


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.U_re = nn.Parameter(torch.eye(dim))
        self.U_im = nn.Parameter(torch.zeros(dim, dim))
        self.V_re = nn.Parameter(torch.eye(dim))
        self.V_im = nn.Parameter(torch.zeros(dim, dim))

        self.dft_weight = nn.Parameter(torch.tensor(0.2))

        self.register_buffer("dft_matrix_re", self._build_dft_matrix(dim).real)
        self.register_buffer("dft_matrix_im", self._build_dft_matrix(dim).imag)

    def _build_dft_matrix(self, n):
        idx = torch.arange(n, dtype=torch.float64)
        phase = -2 * torch.pi * idx.unsqueeze(0) * idx.unsqueeze(1) / n
        return torch.complex(torch.cos(phase), torch.sin(phase)) / (n ** 0.5)

    def _get_U(self):
        return torch.complex(self.U_re, self.U_im)

    def _get_V(self):
        return torch.complex(self.V_re, self.V_im)

    def _get_dft(self):
        return torch.complex(self.dft_matrix_re, self.dft_matrix_im)

    def encode(self, x):
        U = self._get_U().to(x.dtype)
        dft = self._get_dft().to(x.dtype)
        w = torch.sigmoid(self.dft_weight)

        x_svd = torch.einsum("...d,de->...e", x, U)
        x_dft = torch.einsum("...d,de->...e", x, dft)

        return (1 - w) * x_svd + w * x_dft

    def decode(self, z):
        U = self._get_U().to(z.dtype)
        dft = self._get_dft().to(z.dtype)
        w = torch.sigmoid(self.dft_weight)

        z_svd = torch.einsum("...d,ed->...e", z, U.conj())
        z_dft = torch.einsum("...d,ed->...e", z, dft.conj().T)

        return (1 - w) * z_svd + w * z_dft


class GlobalFluxTracker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.decay_re = nn.Parameter(torch.zeros(dim))
        self.decay_im = nn.Parameter(torch.zeros(dim))

        self.input_mix = nn.Linear(dim * 2, dim * 2)
        self.output_proj = nn.Linear(dim * 2, dim * 2)

        nn.init.eye_(self.input_mix.weight[:dim, :dim])
        nn.init.eye_(self.input_mix.weight[dim:, dim:])
        nn.init.zeros_(self.input_mix.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.gate_min = 0.01
        self.gate_max = 0.99

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
        gate = torch.clamp(gate, min=self.gate_min, max=self.gate_max)

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

        self.lam_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im = nn.Parameter(torch.randn(dim) * 0.1)

    def _get_effective_lambda(self):
        lam_re_bounded = self.lam_re.clamp(-self.max_growth_rate, self.max_growth_rate)
        return torch.complex(lam_re_bounded, self.lam_im)

    def get_transition_operators(self, dt):
        lam = self._get_effective_lambda()
        dt_tensor = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=lam.device)

        if dt_tensor.ndim == 0:
            dt_ratio = dt_tensor / self.dt_ref
        else:
            dt_ratio = dt_tensor.unsqueeze(-1) / self.dt_ref

        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)

        lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
        forcing = torch.expm1(exp_arg) / lam_safe

        return decay, forcing

    def get_transition_operators_batch(self, dt_seq):
        lam = self._get_effective_lambda()

        if dt_seq.ndim == 1:
            dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        elif dt_seq.ndim == 2:
            dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        else:
            dt_ratio = dt_seq / self.dt_ref

        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)

        lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
        forcing = torch.expm1(exp_arg) / lam_safe

        return decay, forcing

    def generate_stochastic_term(self, shape, dt, dtype, h_state=None):
        if self.sde_mode != "sde":
            return torch.zeros(shape, dtype=dtype, device=self.base_noise.device)

        device = self.base_noise.device
        dt_tensor = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=device)

        base_scale = self.base_noise.abs() * torch.sqrt(dt_tensor.abs().float() + 1e-8)

        if h_state is not None and self.uncertainty_net is not None:
            h_flat = h_state.reshape(-1, self.dim)
            h_real = torch.cat([h_flat.real, h_flat.imag], dim=-1) if h_flat.is_complex() else h_flat
            h_mag = h_real.abs().mean(dim=-1, keepdim=True)
            uncertainty = self.uncertainty_net(h_mag.expand(-1, self.dim))
            uncertainty = uncertainty.reshape(shape)
            scale = base_scale * (1 + uncertainty)
        else:
            scale = base_scale

        noise_re = torch.randn(shape, device=device, dtype=torch.float64 if dtype == torch.cdouble else torch.float32)
        noise_im = torch.randn(shape, device=device, dtype=torch.float64 if dtype == torch.cdouble else torch.float32)

        return torch.complex(noise_re, noise_im) * scale

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

        return h_tilde_next, flux_next
    