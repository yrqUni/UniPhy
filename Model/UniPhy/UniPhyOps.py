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


class RealMultiScaleSpatialMixer(nn.Module):
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

    def forward(self, x):
        y1 = self.branch_local(x)
        y2 = self.branch_regional(x)
        y3 = self.branch_large(x)
        weights = torch.softmax(self.mix_gate(y1 + y2 + y3), dim=1)
        delta = y1 * weights[:, 0:1] + y2 * weights[:, 1:2] + y3 * weights[:, 2:3]
        return x + delta * self.output_scale


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
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
        scale = torch.clamp(1.0 / safe_norm, max=1.0)
        return beta.to(raw.real.dtype) * raw * scale

    def get_matrix(self, dtype):
        cache_key = str(dtype)
        if (
            not self.training
            and hasattr(self, "_matrix_cache")
            and self._matrix_cache.get("key") == cache_key
        ):
            return self._matrix_cache["w"], self._matrix_cache["w_inv"]

        eye = torch.eye(self.dim, device=self.w_re.device, dtype=dtype)
        dft_w = torch.complex(self.dft_re, self.dft_im).to(dtype)
        dft_inv = torch.complex(self.dft_inv_re, self.dft_inv_im).to(dtype)
        alpha = self.get_alpha().to(dft_w.real.dtype)
        beta = 1.0 - alpha
        alpha_scale = (1.0 + alpha * 1e-3).to(dtype)
        perturb = self._get_bounded_perturbation(dtype, beta)
        learned = eye + perturb
        learned_inv = torch.linalg.inv(learned)
        w = alpha_scale * (dft_w @ learned)
        w_inv = (learned_inv @ dft_inv) / alpha_scale

        if not self.training:
            if not hasattr(self, "_matrix_cache"):
                object.__setattr__(self, "_matrix_cache", {})
            self._matrix_cache["key"] = cache_key
            self._matrix_cache["w"] = w
            self._matrix_cache["w_inv"] = w_inv
        return w, w_inv

    def encode_with(self, x, matrix):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        x = x.to(matrix.dtype)
        return torch.einsum("...d,de->...e", x, matrix)

    def decode_with(self, h, matrix_inv):
        h = h.to(matrix_inv.dtype)
        return torch.einsum("...d,de->...e", h, matrix_inv)


def _safe_phi1(exp_arg, dt_phys, eps=1e-7):
    safe_mask = exp_arg.abs() > eps
    safe_exp_arg = torch.where(safe_mask, exp_arg, torch.ones_like(exp_arg))
    phi1 = torch.expm1(exp_arg) / safe_exp_arg
    phi1_taylor = torch.ones_like(exp_arg) + exp_arg / 2.0 + exp_arg * exp_arg / 6.0
    return dt_phys.to(exp_arg.dtype) * torch.where(safe_mask, phi1, phi1_taylor)


def _safe_phi2(exp_arg, dt_phys, eps=1e-4):
    safe_mask = exp_arg.abs() > eps
    safe_exp_arg = torch.where(safe_mask, exp_arg, torch.ones_like(exp_arg))
    phi2 = (torch.expm1(exp_arg) - exp_arg) / (safe_exp_arg * safe_exp_arg)
    half = torch.full_like(exp_arg, 0.5)
    phi2_taylor = half + exp_arg / 6.0 + exp_arg * exp_arg / 24.0
    return dt_phys.to(exp_arg.dtype) * torch.where(safe_mask, phi2, phi2_taylor)


def _etd2_coefficients(lam_phys, dt_phys):
    exp_arg = lam_phys * dt_phys
    decay = torch.exp(exp_arg)
    phi1_dt = _safe_phi1(exp_arg, dt_phys)
    phi2_dt = _safe_phi2(exp_arg, dt_phys)
    alpha = phi1_dt - phi2_dt
    beta = phi2_dt
    return decay, alpha, beta


class GlobalFluxTracker(nn.Module):

    def __init__(self, dim, dt_ref):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.decay_re_raw = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.decay_im_raw = nn.Parameter(torch.randn(dim) * 0.1)
        self.input_mix = nn.Linear(dim * 2, dim * 2)
        self.output_proj = nn.Linear(dim * 2, dim * 2)
        self.gate_context = nn.Linear(dim * 4, dim * 2)
        self.gate_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.constant_(self.gate_net[0].bias, 0.0)
        self.gate_min = 0.01
        self.gate_max = 0.99
        self.h0_re = nn.Parameter(torch.zeros(dim))
        self.h0_im = nn.Parameter(torch.zeros(dim))

    def get_initial_state(self, batch_size, device, dtype):
        h0 = torch.complex(self.h0_re.to(device), self.h0_im.to(device)).to(dtype)
        return h0.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _get_continuous_params(self):
        lam_re_phys = -F.softplus(self.decay_re_raw) / self.dt_ref
        lam_im_phys = self.decay_im_raw / self.dt_ref
        return torch.complex(lam_re_phys, lam_im_phys)

    def _compute_etd1_operators(self, dt_phys):
        lam_phys = self._get_continuous_params()
        exp_arg = lam_phys * dt_phys
        decay = torch.exp(exp_arg)
        forcing = _safe_phi1(exp_arg, dt_phys) / self.dt_ref
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
        dt_phys = dt_seq.unsqueeze(-1)
        decay, forcing_op = self._compute_etd1_operators(dt_phys)
        decay = decay.unsqueeze(-1)
        forcing_op = forcing_op.unsqueeze(-1)
        x_flat = x_mean_seq.reshape(batch_size * steps, dim)
        x_mixed = self._mix_input(x_flat).reshape(batch_size, steps, self.dim, 1)
        x_scan = x_mixed * forcing_op
        return decay, x_scan

    def compute_output_seq(self, flux_seq):
        return self._compute_output(flux_seq)

    def forward_step(self, prev_state, x_t, dt_step):
        dt_phys = dt_step.unsqueeze(-1)
        decay, forcing_op = self._compute_etd1_operators(dt_phys)
        x_mixed = self._mix_input(x_t)
        new_state = prev_state * decay + x_mixed * forcing_op
        source, gate = self._compute_output(new_state)
        return new_state, source, gate


def _compute_sde_covariance(lam_phys, dt_phys, base_re, base_im):
    real_arg = 2.0 * lam_phys.real * dt_phys
    R = _safe_phi1(real_arg, dt_phys)
    complex_arg = 2.0 * lam_phys * dt_phys
    P_complex = _safe_phi1(complex_arg, dt_phys)
    P_re = P_complex.real
    P_im = P_complex.imag
    base_re_sq = base_re * base_re
    base_im_sq = base_im * base_im
    sum_sigma_sq = base_re_sq + base_im_sq
    diff_sigma_sq = base_re_sq - base_im_sq
    var_re = 0.5 * sum_sigma_sq * R + 0.5 * diff_sigma_sq * P_re
    var_im = 0.5 * sum_sigma_sq * R - 0.5 * diff_sigma_sq * P_re
    cov = 0.5 * diff_sigma_sq * P_im
    var_re = torch.clamp(var_re, min=0.0)
    var_im = torch.clamp(var_im, min=0.0)
    return var_re, var_im, cov


def _cholesky_components(var_re, var_im, cov, eps=1e-30):
    sigma_re = torch.sqrt(var_re + eps)
    cross_factor = cov / sigma_re
    sigma_im_cond_sq = torch.clamp(var_im - cross_factor * cross_factor, min=0.0)
    sigma_im_cond = torch.sqrt(sigma_im_cond_sq + eps)
    return sigma_re, cross_factor, sigma_im_cond


class TemporalPropagator(nn.Module):

    def __init__(self, dim, dt_ref, init_noise_scale):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim, dt_ref)

        self.lam_re_raw = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im_raw = nn.Parameter(torch.randn(dim) * 0.1)
        self.h0_re = nn.Parameter(torch.zeros(dim))
        self.h0_im = nn.Parameter(torch.zeros(dim))
        self.base_noise_re = nn.Parameter(torch.ones(dim) * init_noise_scale)
        self.base_noise_im = nn.Parameter(torch.ones(dim) * init_noise_scale)

    def get_initial_h(self, batch_size, height, width, device, dtype):
        h0 = torch.complex(self.h0_re.to(device), self.h0_im.to(device)).to(dtype)
        return h0.reshape(1, 1, 1, self.dim).expand(batch_size, height, width, -1)

    def _get_effective_lambda(self):

        lam_re_phys = -F.softplus(self.lam_re_raw) / self.dt_ref
        lam_im_phys = self.lam_im_raw / self.dt_ref
        return torch.complex(lam_re_phys, lam_im_phys)

    def get_basis_matrices(self, dtype):
        return self.basis.get_matrix(dtype)

    def get_etd2_operators(self, dt):
        lam_phys = self._get_effective_lambda()
        dt_phys = dt.unsqueeze(-1)
        decay, alpha_phys, beta_phys = _etd2_coefficients(lam_phys, dt_phys)
        scale = 1.0 / self.dt_ref
        return decay, alpha_phys * scale, beta_phys * scale

    def _normalize_explicit_noise(self, noise, dtype):
        if not noise.is_complex():
            raise TypeError(
                "explicit noise must be complex (xi_re + i*xi_im); "
                f"got real dtype={noise.dtype}"
            )
        return noise.to(dtype)

    def _compute_noise_scales(self, dt_phys):
        lam_phys = self._get_effective_lambda()
        broadcast_shape = [1] * dt_phys.ndim
        broadcast_shape[-1] = self.dim
        lam_view = lam_phys.reshape(broadcast_shape)
        dt_phys_cast = dt_phys.to(lam_phys.real.dtype)
        scale = (1.0 / self.dt_ref) ** 0.5
        base_re = self.base_noise_re.abs().reshape(broadcast_shape) * scale
        base_im = self.base_noise_im.abs().reshape(broadcast_shape) * scale
        var_re, var_im, cov = _compute_sde_covariance(
            lam_view,
            dt_phys_cast,
            base_re,
            base_im,
        )
        sigma_re, cross_factor, sigma_im_cond = _cholesky_components(
            var_re, var_im, cov
        )
        return sigma_re, cross_factor, sigma_im_cond

    def _apply_cholesky_noise(
        self, xi_re, xi_im, sigma_re, cross_factor, sigma_im_cond
    ):
        eps_re = sigma_re * xi_re
        eps_im = cross_factor * xi_re + sigma_im_cond * xi_im
        return eps_re, eps_im

    def generate_stochastic_term(self, shape, dt, dtype, noise=None):
        if noise is None:
            return torch.zeros(shape, dtype=dtype, device=self.lam_re_raw.device)
        noise = noise.view(shape)
        expand_shape = tuple(dt.shape) + (1,) * (len(shape) - dt.ndim)
        dt_phys = dt.reshape(expand_shape)
        sigma_re, cross_factor, sigma_im_cond = self._compute_noise_scales(dt_phys)
        noise_c = self._normalize_explicit_noise(noise, dtype)
        real_dtype = noise_c.real.dtype
        eps_re, eps_im = self._apply_cholesky_noise(
            noise_c.real,
            noise_c.imag,
            sigma_re.to(real_dtype),
            cross_factor.to(real_dtype),
            sigma_im_cond.to(real_dtype),
        )
        return torch.complex(eps_re, eps_im)
