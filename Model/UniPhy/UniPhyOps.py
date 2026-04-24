import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_dtype_for(dtype):
    if dtype in {torch.float64, torch.complex128}:
        return torch.complex128
    return torch.complex64


class MultiScaleSpatialMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch_local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.branch_regional = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim, bias=False),
            nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        self.branch_large = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim, bias=False),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1),
        )
        mid = max(1, dim // 4)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(dim, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, 3, 1),
        )
        nn.init.zeros_(self.mix_gate[-1].weight)
        nn.init.zeros_(self.mix_gate[-1].bias)
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def _forward_real(self, x):
        y1 = self.branch_local(x)
        y2 = self.branch_regional(x)
        y3 = self.branch_large(x)
        weights = torch.softmax(self.mix_gate(y1 + y2 + y3), dim=1)
        return y1 * weights[:, 0:1] + y2 * weights[:, 1:2] + y3 * weights[:, 2:3]

    def forward(self, x_complex):
        real = self._forward_real(x_complex.real)
        imag = self._forward_real(x_complex.imag)
        delta = torch.complex(real, imag) * self.output_scale
        return x_complex + delta


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim, neumann_terms=8, rho_max=0.5):
        super().__init__()
        self.dim = dim
        self.neumann_terms = neumann_terms
        self.rho_max = rho_max
        n = torch.arange(dim, dtype=torch.float64)
        k = torch.arange(dim, dtype=torch.float64).reshape(-1, 1)
        dft = torch.exp(-2j * torch.pi * n * k / dim) / (dim**0.5)
        dft_inv = dft.conj().transpose(-1, -2).contiguous()
        self.register_buffer("dft_re", dft.real.clone())
        self.register_buffer("dft_im", dft.imag.clone())
        self.register_buffer("dft_inv_re", dft_inv.real.clone())
        self.register_buffer("dft_inv_im", dft_inv.imag.clone())
        self.w_re = nn.Parameter(torch.zeros(dim, dim))
        self.w_im = nn.Parameter(torch.zeros(dim, dim))
        self.alpha_logit = nn.Parameter(torch.tensor(2.0))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _get_bounded_perturbation(self, dtype, beta):
        raw = torch.complex(torch.tanh(self.w_re), torch.tanh(self.w_im)).to(dtype)
        raw = raw * (self.dim**-0.5)
        norm = torch.linalg.matrix_norm(raw, ord=2)
        safe_norm = torch.clamp(norm, min=torch.finfo(raw.real.dtype).eps)
        scale = torch.clamp(self.rho_max / safe_norm, max=1.0)
        return beta.to(raw.real.dtype) * raw * scale

    def get_matrix(self, dtype):
        eye = torch.eye(self.dim, device=self.w_re.device, dtype=dtype)
        dft_w = torch.complex(self.dft_re, self.dft_im).to(dtype)
        dft_inv = torch.complex(self.dft_inv_re, self.dft_inv_im).to(dtype)
        alpha = self.get_alpha().to(dft_w.real.dtype)
        beta = 1.0 - alpha
        alpha_scale = (1.0 + alpha * 1e-3).to(dtype)
        perturb = self._get_bounded_perturbation(dtype, beta)
        learned = eye + perturb
        learned_inv = eye
        series_term = eye
        for _ in range(self.neumann_terms):
            series_term = -series_term @ perturb
            learned_inv = learned_inv + series_term
        w = alpha_scale * (dft_w @ learned)
        w_inv = (learned_inv @ dft_inv) / alpha_scale
        return w, w_inv


    def encode_with(self, x, matrix):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        work_dtype = (
            torch.complex128 if matrix.dtype == torch.complex64 else matrix.dtype
        )
        out = torch.einsum(
            "...d,de->...e", x.to(work_dtype), matrix.to(work_dtype)
        )
        return out.to(matrix.dtype)

    def decode_with(self, h, matrix_inv):
        work_dtype = (
            torch.complex128
            if matrix_inv.dtype == torch.complex64
            else matrix_inv.dtype
        )
        out = torch.einsum(
            "...d,de->...e", h.to(work_dtype), matrix_inv.to(work_dtype)
        )
        return out.to(matrix_inv.dtype)


def _safe_forcing(exp_arg, dt_ratio, eps=1e-7):
    safe_mask = exp_arg.abs() > eps
    safe_exp_arg = torch.where(safe_mask, exp_arg, torch.ones_like(exp_arg))
    phi1 = torch.expm1(exp_arg) / safe_exp_arg
    phi1_taylor = (
        torch.ones_like(exp_arg) + exp_arg / 2.0 + exp_arg * exp_arg / 6.0
    )
    return dt_ratio.to(exp_arg.dtype) * torch.where(safe_mask, phi1, phi1_taylor)


class GlobalFluxTracker(nn.Module):
    def __init__(self, dim, dt_ref):
        super().__init__()
        if dt_ref <= 0:
            raise ValueError("dt_ref must be positive")
        self.dim = dim
        self.dt_ref = dt_ref
        self.memory_slots = 1
        self.state_dim = dim * self.memory_slots
        self.decay_re = nn.Parameter(torch.randn(self.state_dim) * 0.1 - 1.0)
        self.decay_im = nn.Parameter(torch.randn(self.state_dim) * 0.1)
        self.input_mix = nn.Linear(dim * 2, self.state_dim * 2)
        self.output_proj = nn.Linear(self.state_dim * 2, dim * 2)
        self.gate_context = nn.Linear(self.state_dim * 2 + dim * 2, dim * 2)
        self.gate_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.constant_(self.gate_net[0].bias, 0.0)
        self.gate_min = 0.01
        self.gate_max = 0.99
        self.h0_re = nn.Parameter(torch.zeros(self.state_dim))
        self.h0_im = nn.Parameter(torch.zeros(self.state_dim))

    def get_initial_state(self, batch_size, device, dtype):
        h0 = torch.complex(self.h0_re.to(device), self.h0_im.to(device)).to(dtype)
        return h0.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _get_continuous_params(self):
        lam_re = -F.softplus(self.decay_re)
        lam_im = self.decay_im
        return torch.complex(lam_re, lam_im)

    def _compute_exp_operators(self, dt_ratio):
        lam = self._get_continuous_params()
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)
        forcing = _safe_forcing(exp_arg, dt_ratio)
        return decay, forcing

    def _mix_input(self, x_flat):
        x_cat = torch.cat([x_flat.real, x_flat.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        return torch.complex(x_re, x_im)

    def _compute_output(self, flux):
        state_cat = torch.cat([flux.real, flux.imag], dim=-1)
        source_out = self.output_proj(state_cat)
        src_re, src_im = torch.chunk(source_out, 2, dim=-1)
        source = torch.complex(src_re, src_im)
        gate_input = torch.cat([state_cat, source.real, source.imag], dim=-1)
        gate_input = self.gate_context(gate_input)
        gate = self.gate_net(gate_input)
        gate = gate * (self.gate_max - self.gate_min) + self.gate_min
        return source, gate

    def get_scan_operators(self, x_mean_seq, dt_seq):
        batch_size, steps, dim = x_mean_seq.shape
        dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        decay, forcing_op = self._compute_exp_operators(dt_ratio)
        decay = decay.unsqueeze(-1)
        forcing_op = forcing_op.unsqueeze(-1)
        x_flat = x_mean_seq.reshape(batch_size * steps, dim)
        x_mixed = self._mix_input(x_flat).reshape(batch_size, steps, self.state_dim, 1)
        x_scan = x_mixed * forcing_op
        return decay, x_scan

    def compute_output_seq(self, flux_seq):
        return self._compute_output(flux_seq)

    def forward_step(self, prev_state, x_t, dt_step):
        dt_ratio = dt_step.unsqueeze(-1) / self.dt_ref
        decay, forcing_op = self._compute_exp_operators(dt_ratio)
        x_mixed = self._mix_input(x_t)
        new_state = prev_state * decay + x_mixed * forcing_op
        source, gate = self._compute_output(new_state)
        return new_state, source, gate


def _compute_sde_scale(lam_re, dt_ratio, base_noise, eps=1e-7):
    scaled = 2 * lam_re * dt_ratio
    safe_mask = scaled.abs() > eps
    safe_scaled = torch.where(safe_mask, scaled, torch.ones_like(scaled))
    phi1 = torch.expm1(scaled) / safe_scaled
    phi1_taylor = torch.ones_like(scaled) + scaled / 2.0 + scaled * scaled / 6.0
    variance_factor = dt_ratio * torch.where(safe_mask, phi1, phi1_taylor)
    std_scale = torch.sqrt(torch.clamp(variance_factor, min=0.0))
    return base_noise * std_scale


class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref, init_noise_scale):
        super().__init__()
        if dt_ref <= 0:
            raise ValueError("dt_ref must be positive")
        self.dim = dim
        self.dt_ref = dt_ref
        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim, dt_ref)
        self.lam_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im = nn.Parameter(torch.randn(dim) * 0.1)
        self.h0_re = nn.Parameter(torch.zeros(dim))
        self.h0_im = nn.Parameter(torch.zeros(dim))
        self.base_noise = nn.Parameter(torch.ones(dim) * init_noise_scale)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def get_initial_h(self, batch_size, height, width, device, dtype):
        h0 = torch.complex(self.h0_re.to(device), self.h0_im.to(device)).to(dtype)
        return h0.reshape(1, 1, 1, self.dim).expand(batch_size, height, width, -1)

    def _get_effective_lambda(self):
        lam_re_bounded = -F.softplus(self.lam_re)
        return torch.complex(lam_re_bounded, self.lam_im)

    def get_basis_matrices(self, dtype):
        return self.basis.get_matrix(dtype)

    def _compute_exp_operators(self, dt_ratio):
        lam = self._get_effective_lambda()
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)
        forcing = _safe_forcing(exp_arg, dt_ratio)
        return decay, forcing

    def get_transition_operators_seq(self, dt_seq):
        dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        return self._compute_exp_operators(dt_ratio)

    def get_transition_operators_step(self, dt_step):
        dt_ratio = dt_step.unsqueeze(-1) / self.dt_ref
        return self._compute_exp_operators(dt_ratio)

    def _normalize_explicit_noise(self, noise, dtype):
        if noise is None:
            raise ValueError("explicit noise is required")
        if noise.shape[-1] != self.dim:
            raise ValueError(
                "noise latent dimension mismatch: expected "
                f"{self.dim}, got {noise.shape[-1]}"
            )
        if noise.is_complex():
            noise_complex = noise.to(dtype)
        else:
            real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
            noise_real = noise.to(real_dtype)
            noise_complex = torch.complex(
                noise_real, torch.zeros_like(noise_real)
            ).to(dtype)
        reduce_dims = tuple(range(1, noise_complex.ndim))
        if reduce_dims:
            mag_sq = noise_complex.real.square() + noise_complex.imag.square()
            rms = torch.sqrt(mag_sq.mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6))
            noise_complex = noise_complex / rms
        return noise_complex

    def generate_stochastic_term_seq(
        self, shape, dt_seq, dtype, h_state, noise_seq=None
    ):
        if noise_seq is None:
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)
        if tuple(noise_seq.shape) != tuple(shape):
            raise ValueError(
                "noise shape mismatch for sequence term: expected "
                f"{tuple(shape)}, got {tuple(noise_seq.shape)}"
            )
        _, _, _, _, dim = shape
        lam_re = self._get_effective_lambda().real
        dt_real = dt_seq.real if dt_seq.is_complex() else dt_seq
        dt_ratio = dt_real.reshape(shape[0], shape[1], 1, 1, 1) / self.dt_ref
        lam_re_exp = lam_re.reshape(1, 1, 1, 1, dim)
        base_exp = self.base_noise.abs().reshape(1, 1, 1, 1, dim)
        final_scale = _compute_sde_scale(lam_re_exp, dt_ratio, base_exp)
        if h_state is not None and self.uncertainty_net is not None:
            factor = self.uncertainty_net(h_state.abs()) * 2.0
            final_scale = final_scale * factor
        return self._normalize_explicit_noise(noise_seq, dtype) * final_scale

    def generate_stochastic_term_step(
        self,
        shape,
        dt_step,
        dtype,
        h_state,
        noise_step=None,
    ):
        if noise_step is None:
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)
        if tuple(noise_step.shape) != tuple(shape):
            raise ValueError(
                "noise shape mismatch for step term: expected "
                f"{tuple(shape)}, got {tuple(noise_step.shape)}"
            )
        _, _, _, dim = shape
        lam_re = self._get_effective_lambda().real
        dt_real = dt_step.real if dt_step.is_complex() else dt_step
        dt_ratio = dt_real.reshape(shape[0], 1, 1, 1) / self.dt_ref
        lam_re_exp = lam_re.reshape(1, 1, 1, dim)
        base_exp = self.base_noise.abs().reshape(1, 1, 1, dim)
        final_scale = _compute_sde_scale(lam_re_exp, dt_ratio, base_exp)
        if h_state is not None and self.uncertainty_net is not None:
            factor = self.uncertainty_net(h_state.abs()) * 2.0
            final_scale = final_scale * factor
        return self._normalize_explicit_noise(noise_step, dtype) * final_scale
