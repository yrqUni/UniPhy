import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.u_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.u_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(dim))
        n = torch.arange(dim).float()
        k = torch.arange(dim).float().reshape(-1, 1)
        dft_matrix = torch.exp(-2j * torch.pi * n * k / dim) / (dim ** 0.5)
        self.register_buffer("dft_basis", dft_matrix)
        self.register_buffer(
            "dft_basis_inv", dft_matrix.conj().T.contiguous(),
        )
        self.dft_weight = nn.Parameter(torch.tensor(0.0))

    def _unitary_from_skew(self, raw_re, raw_im):
        A = torch.complex(raw_re, raw_im)
        A_skew = A - A.T.conj()
        return torch.matrix_exp(A_skew)

    def get_matrix(self, dtype):
        U = self._unitary_from_skew(self.u_raw_re, self.u_raw_im)
        V = self._unitary_from_skew(self.v_raw_re, self.v_raw_im)
        S = torch.exp(self.log_sigma)
        S_inv = torch.exp(-self.log_sigma)
        alpha = torch.sigmoid(self.dft_weight)
        S_c = S.to(U.dtype)
        S_inv_c = S_inv.to(U.dtype)
        W_learned = U @ torch.diag(S_c) @ V.conj().T
        W_learned_inv = V @ torch.diag(S_inv_c) @ U.conj().T
        dft = self.dft_basis.to(dtype=dtype)
        dft_inv = self.dft_basis_inv.to(dtype=dtype)
        W = W_learned * (1 - alpha) + dft * alpha
        W_inv = W_learned_inv * (1 - alpha) + dft_inv * alpha
        return W, W_inv

    def encode_with(self, x, W):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        return torch.einsum("...d,de->...e", x, W)

    def decode_with(self, h, W_inv):
        return torch.einsum("...d,de->...e", h, W_inv)

    def encode(self, x):
        W, _ = self.get_matrix(
            x.dtype if x.is_complex() else torch.complex64,
        )
        return self.encode_with(x, W)

    def decode(self, h):
        _, W_inv = self.get_matrix(h.dtype)
        return self.decode_with(h, W_inv)


def _safe_forcing(exp_arg, eps=1e-7):
    safe_denom = exp_arg + eps * torch.sign(
        exp_arg.real + eps
    ).to(exp_arg.dtype)
    return torch.expm1(exp_arg) / safe_denom


class GlobalFluxTracker(nn.Module):
    def __init__(self, dim, dt_ref):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.decay_re = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.decay_im = nn.Parameter(torch.randn(dim) * 0.1)
        self.input_mix = nn.Linear(dim * 2, dim * 2)
        self.output_proj = nn.Linear(dim * 2, dim * 2)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.gate_min = 0.01
        self.gate_max = 0.99

    def _get_continuous_params(self):
        lam_re = -F.softplus(self.decay_re)
        lam_im = self.decay_im
        return torch.complex(lam_re, lam_im)

    def _compute_exp_operators(self, dt_ratio):
        lam = self._get_continuous_params()
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)
        forcing = _safe_forcing(exp_arg)
        return decay, forcing

    def _mix_input(self, x_flat):
        x_cat = torch.cat([x_flat.real, x_flat.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        return torch.complex(x_re, x_im)

    def _compute_output(self, flux):
        s_cat = torch.cat([flux.real, flux.imag], dim=-1)
        source_out = self.output_proj(s_cat)
        src_re, src_im = torch.chunk(source_out, 2, dim=-1)
        source = torch.complex(src_re, src_im)
        gate = self.gate_net(s_cat)
        gate = gate * (self.gate_max - self.gate_min) + self.gate_min
        return source, gate

    def get_scan_operators(self, x_mean_seq, dt_seq):
        B, T, D = x_mean_seq.shape
        dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        decay, forcing_op = self._compute_exp_operators(dt_ratio)
        decay = decay.unsqueeze(-1)
        forcing_op = forcing_op.unsqueeze(-1)
        x_flat = x_mean_seq.reshape(B * T, D)
        x_mixed = self._mix_input(x_flat).reshape(B, T, D, 1)
        X_scan = x_mixed * forcing_op
        return decay, X_scan

    def compute_output_seq(self, flux_seq):
        return self._compute_output(flux_seq)

    def forward_step(self, prev_state, x_t, dt_step):
        B, D = prev_state.shape
        dt_ratio = dt_step.unsqueeze(-1) / self.dt_ref
        decay, forcing_op = self._compute_exp_operators(dt_ratio)
        x_mixed = self._mix_input(x_t)
        new_state = prev_state * decay + x_mixed * forcing_op
        source, gate = self._compute_output(new_state)
        return new_state, source, gate


class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref, sde_mode, init_noise_scale):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.sde_mode = sde_mode
        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim, dt_ref)
        self.lam_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im = nn.Parameter(torch.randn(dim) * 0.1)
        if sde_mode == "sde":
            self.base_noise = nn.Parameter(
                torch.ones(dim) * init_noise_scale
            )
            self.uncertainty_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid(),
            )
        else:
            self.register_buffer("base_noise", torch.tensor(0.0))
            self.uncertainty_net = None

    def _get_effective_lambda(self):
        lam_re_bounded = -F.softplus(self.lam_re)
        return torch.complex(lam_re_bounded, self.lam_im)

    def _compute_exp_operators(self, dt_ratio):
        lam = self._get_effective_lambda()
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)
        forcing = _safe_forcing(exp_arg)
        return decay, forcing

    def get_transition_operators_seq(self, dt_seq):
        dt_ratio = dt_seq.unsqueeze(-1) / self.dt_ref
        return self._compute_exp_operators(dt_ratio)

    def get_transition_operators_step(self, dt_step):
        dt_ratio = dt_step.unsqueeze(-1) / self.dt_ref
        return self._compute_exp_operators(dt_ratio)

    def generate_stochastic_term_seq(self, shape, dt_seq, dtype, h_state):
        if self.sde_mode != "sde":
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)
        B, T, H, W, D = shape
        device = self.lam_re.device
        lam_re = self._get_effective_lambda().real
        dt_real = dt_seq.real if dt_seq.is_complex() else dt_seq
        dt_ratio = dt_real.reshape(B, T, 1, 1, 1) / self.dt_ref
        lam_re_exp = lam_re.reshape(1, 1, 1, 1, D)
        exp_term = torch.exp(2 * lam_re_exp * dt_ratio)
        denom_safe = 2 * lam_re_exp - 1e-8
        variance_factor = (exp_term - 1.0) / denom_safe
        std_scale = torch.sqrt(torch.clamp(variance_factor, min=1e-8))
        base_exp = self.base_noise.abs().reshape(1, 1, 1, 1, D)
        final_scale = base_exp * std_scale
        noise_re = torch.randn(shape, device=device, dtype=torch.float32)
        noise_im = torch.randn(shape, device=device, dtype=torch.float32)
        if h_state is not None and self.uncertainty_net is not None:
            h_mag = h_state.abs()
            factor = self.uncertainty_net(h_mag) * 2.0
            final_scale = final_scale * factor
        if dtype.is_complex:
            return torch.complex(
                noise_re * final_scale, noise_im * final_scale,
            )
        return noise_re * final_scale

    def generate_stochastic_term_step(self, shape, dt_step, dtype, h_state):
        if self.sde_mode != "sde":
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)
        B, H, W, D = shape
        device = self.lam_re.device
        lam_re = self._get_effective_lambda().real
        dt_real = dt_step.real if dt_step.is_complex() else dt_step
        dt_ratio = dt_real.reshape(B, 1, 1, 1) / self.dt_ref
        lam_re_exp = lam_re.reshape(1, 1, 1, D)
        exp_term = torch.exp(2 * lam_re_exp * dt_ratio)
        denom_safe = 2 * lam_re_exp - 1e-8
        variance_factor = (exp_term - 1.0) / denom_safe
        std_scale = torch.sqrt(torch.clamp(variance_factor, min=1e-8))
        base_exp = self.base_noise.abs().reshape(1, 1, 1, D)
        final_scale = base_exp * std_scale
        noise_re = torch.randn(shape, device=device, dtype=torch.float32)
        noise_im = torch.randn(shape, device=device, dtype=torch.float32)
        if h_state is not None and self.uncertainty_net is not None:
            h_mag = h_state.abs()
            factor = self.uncertainty_net(h_mag) * 2.0
            final_scale = final_scale * factor
        if dtype.is_complex:
            return torch.complex(
                noise_re * final_scale, noise_im * final_scale,
            )
        return noise_re * final_scale


class RiemannianCliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 img_height, img_width):
        super().__init__()
        self.conv_e0 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding,
            bias=False,
        )
        self.conv_e1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding,
            bias=False,
        )
        self.conv_e2 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding,
            bias=False,
        )
        self.conv_e12 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding,
            bias=False,
        )
        self.smooth_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        lat = torch.linspace(-math.pi / 2, math.pi / 2, img_height)
        self.register_buffer(
            "cos_lat", torch.cos(lat).view(1, 1, img_height, 1),
        )
        self.metric_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, C, H, W = x.shape
        y_e0 = self.conv_e0(x)
        y_e1 = self.conv_e1(x)
        y_e2 = self.conv_e2(x)
        y_e12 = self.conv_e12(x)
        cos_lat = self.cos_lat.expand(B, 1, H, W)
        cos_lat_safe = torch.clamp(cos_lat, min=1e-6)
        y_e1_scaled = y_e1 * cos_lat_safe
        y_e12_scaled = y_e12 * cos_lat_safe
        out = y_e0 + y_e1_scaled + y_e2 + y_e12_scaled
        out = out * self.metric_scale
        out = out + self.bias.view(1, -1, 1, 1)
        out = self.smooth_conv(out)
        return out
    