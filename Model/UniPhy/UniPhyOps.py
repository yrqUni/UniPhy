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
        laplacian_init = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplacian_kernel = nn.Parameter(laplacian_init.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        
        self.metric_scale = nn.Parameter(torch.tensor(0.1))
        self.viscosity_scale = nn.Parameter(torch.tensor(0.01))
        self.diffusion_scale = nn.Parameter(torch.tensor(0.01))
        
        self.groups = in_channels

    def forward(self, x):
        dtype = self.conv.weight.dtype
        x = x.to(dtype)
        base_log = torch.clamp(self.log_metric_param.to(dtype), -5, 5)
        dynamic_log = self.metric_refiner(x) * self.metric_scale
        effective_log_metric = base_log + dynamic_log
        
        scale = torch.exp(effective_log_metric)
        inv_scale = torch.exp(-effective_log_metric)
        
        kernel = self.laplacian_kernel.to(dtype)
        diffusion_term = F.conv2d(x, kernel, padding=1, groups=self.groups)
        local_viscosity = self.viscosity_gate(x) * inv_scale
        
        x_diffused = x + diffusion_term * local_viscosity * self.viscosity_scale * self.diffusion_scale
        x_scaled = x_diffused * scale
        out = self.conv(x_scaled)
        out = out * inv_scale
        return out

class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.u_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.v_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.v_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def _cayley_transform(self, raw_re, raw_im):        
        raw_re = torch.clamp(raw_re, -50.0, 50.0)
        raw_im = torch.clamp(raw_im, -50.0, 50.0)
        
        A_re = (raw_re - raw_re.T) * 0.5
        A_im = (raw_im + raw_im.T) * 0.5
        A = torch.complex(A_re, A_im)
        
        A_64 = A.to(torch.complex128)
        I_64 = torch.eye(self.dim, device=raw_re.device, dtype=torch.complex128)
        
        eps = 1e-9
        U_64 = torch.linalg.solve(I_64 * (1.0 + eps) - A_64, I_64 + A_64)
    
        return U_64.to(A.dtype)

    def _get_basis(self):
        U = self._cayley_transform(self.u_raw_re, self.u_raw_im)
        V = self._cayley_transform(self.v_raw_re, self.v_raw_im)
        S = torch.exp(torch.clamp(self.log_sigma, -10, 10)).type_as(U)
        S_mat = torch.diag_embed(S + 0j) 
        return U, S_mat, V

    def encode(self, x):
        U, S_mat, V = self._get_basis()
        S_inv_val = torch.exp(-torch.clamp(self.log_sigma, -10, 10)).type_as(U)
        S_inv = torch.diag_embed(S_inv_val + 0j)
        M_inv = V @ S_inv @ U.conj().T
        return torch.matmul(x.to(M_inv.dtype), M_inv.T)

    def decode(self, x):
        U, S_mat, V = self._get_basis()
        M = U @ S_mat @ V.conj().T
        return torch.matmul(x.to(M.dtype), M.T)
    
class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01, selective_dynamics=True):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.noise_scale = noise_scale
        self.selective_dynamics = selective_dynamics
        self.basis = ComplexSVDTransform(dim)
        
        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.randn(dim) * 1.0)
        self.src_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.src_im = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)

    def _get_effective_lambda(self):
        l_phys = torch.complex(-torch.exp(torch.clamp(self.ld, max=10.0)), self.lf)
        l_law = torch.complex(torch.clamp(self.law_re, -5, 5), self.law_im)
        return l_phys + l_law

    def _get_source_bias(self):
        return torch.complex(self.src_re, self.src_im)

    def get_transition_operators(self, dt):
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        dt_eff = (dt / self.dt_ref).unsqueeze(-1)
        Lambda = self._get_effective_lambda()
        Z = Lambda * dt_eff
        
        mask = torch.abs(Z) < 1e-4
        phi1 = torch.where(mask, 1.0 + 0.5 * Z + (Z**2)/6.0, torch.expm1(Z) / torch.where(mask, torch.ones_like(Z), Z))
        return torch.exp(Z), phi1 * (dt_eff * self.dt_ref)

    def generate_stochastic_term(self, target_shape, dt, dtype):
        if not self.training or self.noise_scale <= 0:
            return torch.zeros(target_shape, device=self.ld.device, dtype=dtype)
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        l_re = self._get_effective_lambda().real
        var = (self.noise_scale ** 2) * torch.expm1(2 * l_re * (dt / self.dt_ref).unsqueeze(-1)) / (2 * l_re - 1e-6)
        std = torch.sqrt(torch.abs(var)).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std

    def forward(self, h_prev, x_input, dt):
        h_tilde = self.basis.encode(h_prev)
        if h_tilde.ndim == 2: h_tilde = h_tilde.unsqueeze(1)
        x_tilde = self.basis.encode(x_input)
        if x_tilde.ndim == 2: x_tilde = x_tilde.unsqueeze(1)
        
        op_decay, op_forcing = self.get_transition_operators(dt)
        bias = self._get_source_bias()
        
        h_tilde_next = h_tilde * op_decay + (x_tilde + bias) * op_forcing
        h_tilde_next = h_tilde_next + self.generate_stochastic_term(h_tilde_next.shape, dt, h_tilde_next.dtype)
        
        return self.basis.decode(h_tilde_next)
    