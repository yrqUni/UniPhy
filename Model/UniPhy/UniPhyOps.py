import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_continuous_operators_seq(lam, dt, tau_ref_hours):
    if dt.is_complex():
        dt = dt.real
    dt_ratio = dt / float(tau_ref_hours)
    exp_arg = dt_ratio.unsqueeze(-1) * lam.view(1, 1, -1)
    decay = torch.exp(exp_arg)
    lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
    forcing = torch.expm1(exp_arg) / lam_safe.view(1, 1, -1)
    return decay, forcing


def compute_continuous_operators_step(lam, dt, tau_ref_hours):
    if dt.is_complex():
        dt = dt.real
    dt_ratio = dt / float(tau_ref_hours)
    exp_arg = dt_ratio.unsqueeze(-1) * lam.view(1, -1)
    decay = torch.exp(exp_arg)
    lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
    forcing = torch.expm1(exp_arg) / lam_safe.view(1, -1)
    return decay, forcing


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.u_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.u_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)

    @staticmethod
    def _unitary_from_raw(re_raw, im_raw):
        raw = torch.complex(re_raw, im_raw)
        q, _ = torch.linalg.qr(raw)
        return q

    def get_basis(self):
        u = self._unitary_from_raw(self.u_raw_re, self.u_raw_im)
        v = self._unitary_from_raw(self.v_raw_re, self.v_raw_im)
        return u, v

    def encode(self, x):
        u, _ = self.get_basis()
        return torch.matmul(x, u.conj())

    def decode(self, x):
        u, _ = self.get_basis()
        return torch.matmul(x, u)


class GlobalFluxTracker(nn.Module):
    def __init__(self, dim, tau_ref_hours=1.0):
        super().__init__()
        self.dim = int(dim)
        self.tau_ref_hours = float(tau_ref_hours)
        self.decay_re = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.decay_im = nn.Parameter(torch.randn(dim) * 0.1)
        self.gate_raw = nn.Parameter(torch.randn(dim) * 0.02)
        self.source_raw_re = nn.Parameter(torch.randn(dim) * 0.02)
        self.source_raw_im = nn.Parameter(torch.randn(dim) * 0.02)

    def _get_continuous_params(self):
        return torch.complex(self.decay_re, self.decay_im)

    def get_scan_operators(self, x_mean_seq, dt):
        bsz, t_len, dim = x_mean_seq.shape
        lam = self._get_continuous_params()
        decay, forcing = compute_continuous_operators_seq(lam, dt, self.tau_ref_hours)
        u = x_mean_seq * forcing
        return decay.unsqueeze(-1), u.unsqueeze(-1)

    def get_transition_operators_step(self, dt):
        lam = self._get_continuous_params()
        return compute_continuous_operators_step(lam, dt, self.tau_ref_hours)

    def compute_output_seq(self, flux_seq):
        bsz, t_len, _ = flux_seq.shape
        gate = torch.sigmoid(self.gate_raw).view(1, 1, -1).expand(bsz, t_len, -1)
        source = torch.complex(self.source_raw_re, self.source_raw_im).view(1, 1, -1).expand(bsz, t_len, -1)
        return source * flux_seq, gate

    def compute_output_step(self, flux_next):
        bsz = flux_next.shape[0]
        gate = torch.sigmoid(self.gate_raw).view(1, -1).expand(bsz, -1)
        source = torch.complex(self.source_raw_re, self.source_raw_im).view(1, -1).expand(bsz, -1)
        return source * flux_next, gate


class TemporalPropagator(nn.Module):
    def __init__(
        self,
        dim,
        tau_ref_hours=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.tau_ref_hours = float(tau_ref_hours)
        self.sde_mode = str(sde_mode)
        self.init_noise_scale = float(init_noise_scale)
        self.max_growth_rate = float(max_growth_rate)
        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim, tau_ref_hours=self.tau_ref_hours)
        self.lam_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im = nn.Parameter(torch.randn(dim) * 0.1)
        self.noise_raw = nn.Parameter(torch.randn(dim) * 0.02)

    def _get_effective_lambda(self):
        lam_re = -F.softplus(-self.lam_re)
        lam_re = torch.clamp(lam_re, min=-self.max_growth_rate)
        return torch.complex(lam_re, self.lam_im)

    def get_transition_operators_seq(self, dt):
        lam = self._get_effective_lambda()
        return compute_continuous_operators_seq(lam, dt, self.tau_ref_hours)

    def get_transition_operators_step(self, dt):
        lam = self._get_effective_lambda()
        return compute_continuous_operators_step(lam, dt, self.tau_ref_hours)

    def generate_stochastic_term_seq(self, shape, dt, dtype):
        if self.sde_mode != "sde":
            return torch.zeros(shape, device=self.noise_raw.device, dtype=dtype)
        dt_scale = torch.sqrt(torch.clamp(dt, min=0.0)).to(dtype=torch.float32)
        dt_scale = dt_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise_scale = F.softplus(self.noise_raw).to(dtype=torch.float32)
        noise_scale = noise_scale.view(1, 1, 1, 1, -1)
        eps = torch.randn(shape, device=self.noise_raw.device, dtype=dtype)
        if not eps.is_complex():
            eps = torch.complex(eps, torch.zeros_like(eps))
        return eps * dt_scale.to(device=eps.device) * noise_scale.to(device=eps.device) * float(self.init_noise_scale)

    def generate_stochastic_term_step(self, shape, dt, dtype):
        if self.sde_mode != "sde":
            return torch.zeros(shape, device=self.noise_raw.device, dtype=dtype)
        dt_scale = torch.sqrt(torch.clamp(dt, min=0.0)).to(dtype=torch.float32)
        dt_scale = dt_scale.view(dt_scale.shape[0], 1, 1, 1)
        noise_scale = F.softplus(self.noise_raw).to(dtype=torch.float32)
        noise_scale = noise_scale.view(1, 1, 1, -1)
        eps = torch.randn(shape, device=self.noise_raw.device, dtype=dtype)
        if not eps.is_complex():
            eps = torch.complex(eps, torch.zeros_like(eps))
        return eps * dt_scale.to(device=eps.device) * noise_scale.to(device=eps.device) * float(self.init_noise_scale)


class RiemannianCliffordConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.metric_raw = nn.Parameter(torch.randn(out_channels) * 0.02)

    def forward(self, x):
        metric = 1.0 + torch.tanh(self.metric_raw).view(1, -1, 1, 1)
        y = F.conv2d(x, self.weight, bias=self.bias, padding=self.padding)
        return y * metric
