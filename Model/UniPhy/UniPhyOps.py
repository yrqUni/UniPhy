import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

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
        base_log = self.log_metric_param
        dynamic_log = self.metric_refiner(x) * 0.1
        effective_log_metric = base_log + dynamic_log
        scale = torch.exp(effective_log_metric)
        inv_scale = torch.exp(-effective_log_metric)
        
        kernel = self.laplacian_kernel.to(x.dtype)
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
        self.l_lower_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.l_lower_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_upper_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_upper_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.u_log_s = nn.Parameter(torch.zeros(dim)) 
        self.u_phase = nn.Parameter(torch.zeros(dim))
        self.register_buffer('mask_lower', torch.tril(torch.ones(dim, dim), diagonal=-1))
        self.register_buffer('mask_upper', torch.triu(torch.ones(dim, dim), diagonal=1))

    def _assemble_matrices(self):
        dtype = self.l_lower_real.dtype
        device = self.l_lower_real.device
        L_real = self.l_lower_real * self.mask_lower + torch.eye(self.dim, device=device, dtype=dtype)
        L_imag = self.l_lower_imag * self.mask_lower
        L = torch.complex(L_real, L_imag)
        diag_mod = torch.exp(self.u_log_s)
        diag = torch.complex(diag_mod * torch.cos(self.u_phase), diag_mod * torch.sin(self.u_phase))
        U_rest = torch.complex(self.u_upper_real, self.u_upper_imag) * self.mask_upper
        U = U_rest + torch.diag_embed(diag)
        return L, U

    def encode(self, x):
        L, U = self._assemble_matrices()
        L_inv = torch.linalg.solve_triangular(L, torch.eye(self.dim, device=x.device, dtype=L.dtype), upper=False)
        U_inv = torch.linalg.solve_triangular(U, torch.eye(self.dim, device=x.device, dtype=U.dtype), upper=True)
        x_perm = x[..., self.inv_perm_idx]
        mat_inv = U_inv @ L_inv
        x_eigen = torch.matmul(x_perm.to(mat_inv.dtype), mat_inv.T)
        return x_eigen

    def decode(self, x_eigen):
        L, U = self._assemble_matrices()
        x_step1 = torch.matmul(x_eigen, U.T)
        x_step2 = torch.matmul(x_step1, L.T)
        x_out = x_step2[..., self.perm_idx]
        return x_out

class AnalyticSpectralPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.noise_scale = noise_scale
        self.basis = ComplexPLUTransform(dim)
        self.lambda_log_decay = nn.Parameter(torch.randn(dim) * 0.5 - 2.0) 
        self.lambda_freq = nn.Parameter(torch.randn(dim) * 1.0)

    def _get_spectrum(self):
        real = -torch.exp(self.lambda_log_decay)
        imag = self.lambda_freq
        return torch.complex(real, imag)

    def _safe_phi_1(self, z):
        small_mask = torch.abs(z) < 1e-4
        z_safe = z.clone()
        z_safe[small_mask] = 1.0 
        res_normal = torch.expm1(z_safe) / z_safe
        res_small = 1.0 + z * 0.5 + (z * z) / 6.0
        return torch.where(small_mask, res_small, res_normal)

    def get_transition_operators(self, dt):
        Lambda = self._get_spectrum()
        if isinstance(dt, torch.Tensor):
            if dt.ndim == 0: dt = dt.view(1)
            dt_eff = dt.view(-1, 1) / self.dt_ref
        elif isinstance(dt, float):
            dt_eff = torch.tensor(dt, device=Lambda.device, dtype=Lambda.real.dtype).view(1, 1) / self.dt_ref
        else:
             dt_eff = torch.tensor(dt, device=Lambda.device, dtype=Lambda.real.dtype).view(-1, 1) / self.dt_ref
        Z = Lambda.unsqueeze(0) * dt_eff
        op_decay = torch.exp(Z)
        op_forcing = self._safe_phi_1(Z) * dt_eff * self.dt_ref
        return op_decay, op_forcing

    def generate_stochastic_term(self, shape, dt, device):
        if not self.training or self.noise_scale <= 0:
            return torch.zeros(shape, device=device, dtype=torch.cfloat)
        
        if isinstance(dt, torch.Tensor):
            std = torch.sqrt(dt.abs() / self.dt_ref) * self.noise_scale
            
            if std.ndim == 2:
                B, T = std.shape
                N = shape[0]
                if N % B == 0:
                    HW = N // B
                    std = std.unsqueeze(1).expand(B, HW, T).reshape(N, T, 1)
                else:
                    std = std.view(-1, 1, 1)
            
            elif std.ndim == 1:
                std = std.view(-1, 1, 1)
                if std.shape[0] != shape[0]:
                     if shape[0] % std.shape[0] == 0:
                         factor = shape[0] // std.shape[0]
                         std = std.repeat_interleave(factor, dim=0)

        else:
            std = (dt / self.dt_ref)**0.5 * self.noise_scale
            
        noise = torch.randn(shape, device=device, dtype=torch.cfloat) * std
        return noise

    def forward(self, h_prev, x_input, dt):
        h_tilde = self.basis.encode(h_prev)
        x_tilde = self.basis.encode(x_input)
        
        op_decay, op_forcing = self.get_transition_operators(dt)
        
        h_tilde_next = h_tilde * op_decay + x_tilde * op_forcing
        
        noise = self.generate_stochastic_term(h_tilde_next.shape, dt, h_prev.device)
        h_tilde_next = h_tilde_next + noise
        
        h_next = self.basis.decode(h_tilde_next)
        return h_next

class SpectralStep(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim, h, w, dtype=torch.cfloat) * 0.02)
        self.viscosity_x = nn.Parameter(torch.tensor(1e-4))
        self.viscosity_y = nn.Parameter(torch.tensor(1e-4))
        self.alpha = nn.Parameter(torch.tensor(2.0))
        kx = torch.fft.fftfreq(w, d=1.0)
        ky = torch.fft.fftfreq(h, d=1.0)
        k_x, k_y = torch.meshgrid(kx, ky, indexing='xy')
        self.register_buffer('k_x_abs', k_x.abs())
        self.register_buffer('k_y_abs', k_y.abs())

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')
        original_dc = x_fft[..., 0, 0].clone()
        x_fft = x_fft * self.weight
        alpha_clamp = self.alpha.clamp(1.0, 3.0)
        vx = self.viscosity_x.abs()
        vy = self.viscosity_y.abs()
        k_term = vx * torch.pow(self.k_x_abs, alpha_clamp) + vy * torch.pow(self.k_y_abs, alpha_clamp)
        dissipation = torch.exp(-k_term * (H * W))
        x_fft = x_fft * dissipation.unsqueeze(0).unsqueeze(0)
        x_fft = x_fft.clone()
        x_fft[..., 0, 0] = original_dc
        out = torch.fft.ifft2(x_fft, s=(H, W), norm='ortho')
        return out

