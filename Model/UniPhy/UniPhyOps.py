import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianCliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, img_height, img_width):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
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
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.groups = in_channels

    def forward(self, x):
        dtype = self.conv.weight.dtype
        x = x.to(dtype)
        base_log = self.log_metric_param.to(dtype)
        dynamic_log = self.metric_refiner(x) * 0.1
        effective_log_metric = base_log + dynamic_log
        scale = torch.exp(effective_log_metric)
        inv_scale = torch.exp(-effective_log_metric)
        kernel = self.laplacian_kernel.to(dtype)
        diffusion_term = F.conv2d(x, kernel, padding=1, groups=self.groups)
        local_viscosity = self.viscosity_gate(x) * inv_scale
        x_diffused = x + diffusion_term * local_viscosity * 0.01
        x_scaled = x_diffused * scale
        out = self.conv(x_scaled)
        out = out * inv_scale
        return out

class ComplexPLUTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        perm_idx = torch.randperm(dim)
        self.register_buffer('perm_idx', perm_idx)
        self.register_buffer('inv_perm_idx', torch.argsort(perm_idx))
        self.l_re = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.l_im = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_re = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_im = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_s = nn.Parameter(torch.zeros(dim))
        self.u_p = nn.Parameter(torch.zeros(dim))
        self.register_buffer('mask_l', torch.tril(torch.ones(dim, dim), diagonal=-1))
        self.register_buffer('mask_u', torch.triu(torch.ones(dim, dim), diagonal=1))

    def _mat(self):
        device = self.l_re.device
        dtype = self.l_re.dtype
        L = torch.complex(self.l_re * self.mask_l + torch.eye(self.dim, device=device, dtype=dtype), self.l_im * self.mask_l)
        diag = torch.complex(torch.exp(self.u_s) * torch.cos(self.u_p), torch.exp(self.u_s) * torch.sin(self.u_p))
        U = torch.complex(self.u_re, self.u_im) * self.mask_u + torch.diag_embed(diag)
        return L, U

    def encode(self, x):
        L, U = self._mat()
        L_inv = torch.linalg.solve_triangular(L, torch.eye(self.dim, device=x.device, dtype=L.dtype), upper=False)
        U_inv = torch.linalg.solve_triangular(U, torch.eye(self.dim, device=x.device, dtype=U.dtype), upper=True)
        mat_inv = U_inv @ L_inv
        return torch.matmul(x[..., self.inv_perm_idx].to(mat_inv.dtype), mat_inv.T)

    def decode(self, x):
        L, U = self._mat()
        return torch.matmul(torch.matmul(x, U.T), L.T)[..., self.perm_idx]

class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.noise_scale = noise_scale
        self.basis = ComplexPLUTransform(dim)
        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.randn(dim) * 1.0)
        self.src_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.src_im = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)

    def _get_spectrum(self):
        return torch.complex(-torch.exp(self.ld), self.lf)

    def _get_source_law(self, h):
        bias = torch.complex(self.src_re, self.src_im)
        weight = torch.complex(self.law_re, self.law_im)
        return h * weight + bias

    def get_transition_operators(self, dt):
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        dt_eff = dt / self.dt_ref
        Lambda = self._get_spectrum()
        Z = Lambda * dt_eff.unsqueeze(-1)
        mask = torch.abs(Z) < 1e-4
        phi1 = torch.where(mask, 1.0 + 0.5 * Z, torch.expm1(Z) / torch.where(mask, torch.ones_like(Z), Z))
        return torch.exp(Z), phi1 * (dt_eff.unsqueeze(-1) * self.dt_ref)

    def generate_stochastic_term(self, target_shape, dt, dtype):
        if not self.training or self.noise_scale <= 0:
            return torch.zeros(target_shape, device=self.ld.device, dtype=dtype)
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        l_re = -torch.exp(self.ld)
        var = (self.noise_scale ** 2) * torch.expm1(2 * l_re * (dt / self.dt_ref).unsqueeze(-1)) / (2 * l_re)
        std = torch.sqrt(torch.abs(var)).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std

    def forward(self, h_prev, x_input, dt):
        h_tilde = self.basis.encode(h_prev)
        x_tilde = self.basis.encode(x_input)
        op_decay, op_forcing = self.get_transition_operators(dt)
        source = self._get_source_law(h_tilde)
        h_tilde_next = h_tilde * op_decay + (x_tilde + source) * op_forcing
        h_tilde_next = h_tilde_next + self.generate_stochastic_term(h_tilde_next.shape, dt, h_tilde_next.dtype)
        return self.basis.decode(h_tilde_next)
    