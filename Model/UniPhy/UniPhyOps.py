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
        base_log = self.log_metric_param.to(dtype)
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
        A_re = (raw_re - raw_re.T) * 0.5
        A_im = (raw_im + raw_im.T) * 0.5
        A = torch.complex(A_re, A_im)
        I = torch.eye(self.dim, device=raw_re.device, dtype=A.dtype)
        U = torch.linalg.solve(I - A, I + A) 
        return U
    
    def _get_basis(self):
        U = self._cayley_transform(self.u_raw_re, self.u_raw_im)
        V = self._cayley_transform(self.v_raw_re, self.v_raw_im)
        S = torch.exp(self.log_sigma).type_as(U)
        S_mat = torch.diag_embed(S + 0j) 
        return U, S_mat, V
    
    def encode(self, x):
        U, S_mat, V = self._get_basis()
        S_inv_val = torch.exp(-self.log_sigma).type_as(U)
        S_inv = torch.diag_embed(S_inv_val + 0j)
        M_inv = V @ S_inv @ U.conj().T
        return torch.matmul(x.to(M_inv.dtype), M_inv.T)
    
    def decode(self, x):
        U, S_mat, V = self._get_basis()
        M = U @ S_mat @ V.conj().T
        return torch.matmul(x.to(M.dtype), M.T)
    
class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01, selective_dynamics=True, n_groups=2):
        super().__init__()
        self.dim = dim
        self.dt_ref_base = dt_ref
        self.noise_scale = noise_scale
        self.selective_dynamics = selective_dynamics
        self.n_groups = n_groups
        self.basis = ComplexSVDTransform(dim)
        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.zeros(dim))
        dt_scalers = torch.ones(dim)
        group_size = dim // n_groups
        for i in range(n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < n_groups - 1 else dim
            if i == 0:
                dt_scalers[start:end] = 0.1
                nn.init.normal_(self.lf[start:end], std=2.0)
            else:
                dt_scalers[start:end] = 10.0
                nn.init.normal_(self.lf[start:end], std=0.1)
        self.register_buffer('dt_scaler', dt_scalers)
        self.src_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.src_im = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)
        if self.selective_dynamics:
            self.selector = nn.Linear(dim * 2, dim * 2)
            nn.init.zeros_(self.selector.weight)
            nn.init.zeros_(self.selector.bias)

    def _get_effective_lambda(self, x_state=None):
        l_phys = torch.complex(-torch.exp(self.ld), self.lf)
        l_law = torch.complex(self.law_re, self.law_im)
        lambda_static = l_phys + l_law
        if self.selective_dynamics and x_state is not None:
            if x_state.ndim == 2:
                x_cat = torch.cat([x_state.real, x_state.imag], dim=-1)
                delta = self.selector(x_cat)
                d_re, d_im = torch.chunk(delta, 2, dim=-1)
                return lambda_static + torch.complex(d_re, d_im)
            elif x_state.ndim == 3:
                x_cat = torch.cat([x_state.real, x_state.imag], dim=-1)
                delta = self.selector(x_cat)
                d_re, d_im = torch.chunk(delta, 2, dim=-1)
                return lambda_static.unsqueeze(0).unsqueeze(0) + torch.complex(d_re, d_im)
        return lambda_static
    
    def _get_source_bias(self):
        return torch.complex(self.src_re, self.src_im)
    
    def get_transition_operators(self, dt, x_state=None):
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        dt_ref_eff = self.dt_ref_base * self.dt_scaler
        if dt.ndim == x_state.ndim - 1:
             dt = dt.unsqueeze(-1)
        dt_normalized = dt / dt_ref_eff
        Lambda = self._get_effective_lambda(x_state)
        Z = Lambda * dt_normalized
        mask = torch.abs(Z) < 1e-4
        phi1 = torch.where(mask, 1.0 + 0.5 * Z + (Z**2)/6.0, torch.expm1(Z) / torch.where(mask, torch.ones_like(Z), Z))
        op_decay = torch.exp(Z)
        op_forcing = phi1 * (dt_normalized * dt_ref_eff)
        return op_decay, op_forcing
    
    def generate_stochastic_term(self, target_shape, dt, dtype, x_state=None):
        if not self.training or self.noise_scale <= 0:
            return torch.zeros(target_shape, device=self.ld.device, dtype=dtype)
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        dt_ref_eff = self.dt_ref_base * self.dt_scaler
        l_re = self._get_effective_lambda(x_state).real
        if dt.ndim < l_re.ndim:
             dt = dt.unsqueeze(-1)
        ratio = dt / dt_ref_eff
        var = (self.noise_scale ** 2) * torch.expm1(2 * l_re * ratio) / (2 * l_re)
        std = torch.sqrt(torch.abs(var)).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std
    
    def forward(self, h_prev, x_input, dt):
        h_tilde = self.basis.encode(h_prev)
        if h_tilde.ndim == 2: h_tilde = h_tilde.unsqueeze(1)
        x_tilde = self.basis.encode(x_input)
        if x_tilde.ndim == 2: x_tilde = x_tilde.unsqueeze(1)
        op_decay, op_forcing = self.get_transition_operators(dt, x_state=x_tilde.squeeze(1))
        bias = self._get_source_bias()
        if op_decay.ndim == 2 and h_tilde.ndim == 3: op_decay = op_decay.unsqueeze(1)
        if op_forcing.ndim == 2 and x_tilde.ndim == 3: op_forcing = op_forcing.unsqueeze(1)
        h_tilde_next = h_tilde * op_decay + (x_tilde + bias) * op_forcing
        h_tilde_next = h_tilde_next + self.generate_stochastic_term(h_tilde_next.shape, dt, h_tilde_next.dtype, x_state=x_tilde.squeeze(1))
        return self.basis.decode(h_tilde_next)
    