import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.noise_scale = noise_scale
        self.basis = ComplexSVDTransform(dim)
        
        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.randn(dim) * 1.0)
        
        self.lambda_net = nn.Linear(dim, dim * 2) 
        nn.init.zeros_(self.lambda_net.weight)
        nn.init.zeros_(self.lambda_net.bias)
        
        self.src_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.src_im = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)

    def _get_effective_lambda(self, x=None):
        l_phys = torch.complex(-torch.exp(self.ld), self.lf)
        l_law = torch.complex(self.law_re, self.law_im)
        l_base = l_phys + l_law
        
        if x is not None:
            dyn_params = self.lambda_net(x.real) 
            dyn_re, dyn_im = torch.chunk(dyn_params, 2, dim=-1)
            l_dyn = torch.complex(torch.tanh(dyn_re), torch.tanh(dyn_im)) 
            return l_base + l_dyn
        return l_base

    def _get_source_bias(self):
        return torch.complex(self.src_re, self.src_im)

    def get_transition_operators(self, dt, x=None):
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        dt_eff = (dt / self.dt_ref).unsqueeze(-1)
        
        Lambda = self._get_effective_lambda(x)
        Z = Lambda * dt_eff
        
        mask = torch.abs(Z) < 1e-4
        
        Z_safe = torch.where(mask, torch.ones_like(Z), Z)
        
        phi1 = torch.where(mask, 
                           1.0 + 0.5 * Z, 
                           torch.expm1(Z) / Z_safe)
        
        phi2 = torch.where(mask,
                           0.5 + Z / 6.0,
                           (torch.expm1(Z) - Z) / (Z_safe ** 2))
                           
        return torch.exp(Z), phi1 * (dt_eff * self.dt_ref), phi2 * (dt_eff * self.dt_ref)

    def generate_stochastic_term(self, target_shape, dt, dtype):
        if not self.training or self.noise_scale <= 0:
            return torch.zeros(target_shape, device=self.ld.device, dtype=dtype)
        dt = torch.as_tensor(dt, device=self.ld.device, dtype=self.ld.dtype)
        l_re = self._get_effective_lambda(None).real
        var = (self.noise_scale ** 2) * torch.expm1(2 * l_re * (dt / self.dt_ref).unsqueeze(-1)) / (2 * l_re)
        std = torch.sqrt(torch.abs(var)).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std

    def forward(self, h_prev, x_input, dt):
        h_tilde = self.basis.encode(h_prev)
        if h_tilde.ndim == 2: h_tilde = h_tilde.unsqueeze(1)
        
        x_tilde = self.basis.encode(x_input)
        if x_tilde.ndim == 2: x_tilde = x_tilde.unsqueeze(1)
        
        op_decay, op_phi1, op_phi2 = self.get_transition_operators(dt, x_tilde)
        bias = self._get_source_bias()
        
        x_forcing = x_tilde + bias
        
        if x_tilde.shape[1] > 1:
            x_curr = x_forcing
            x_next = torch.cat([x_forcing[:, 1:], x_forcing[:, -1:]], dim=1)
            
            coeff_curr = op_phi1 - op_phi2
            coeff_next = op_phi2
            
            u_t = x_curr * coeff_curr + x_next * coeff_next
            
            noise = self.generate_stochastic_term(u_t.shape, dt, u_t.dtype)
            u_t = u_t + noise
            
            return self.basis.decode(h_tilde), op_decay, u_t
            
        else:
            u_t = x_forcing * op_phi1
            noise = self.generate_stochastic_term(u_t.shape, dt, u_t.dtype)
            h_tilde_next = h_tilde * op_decay + u_t + noise
            return self.basis.decode(h_tilde_next)
        