import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

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
        
        diffusion_term = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.groups)
        local_viscosity = self.viscosity_gate(x) * inv_scale
        x_diffused = x + diffusion_term * local_viscosity * 0.01
        
        x_scaled = x_diffused * scale
        out = self.conv(x_scaled)
        out = out * inv_scale
        return out

class SpectralNoiseInjector(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        
        ky = torch.fft.fftfreq(h, d=1.0)
        kx = torch.fft.fftfreq(w, d=1.0)
        k_x, k_y = torch.meshgrid(kx, ky, indexing='xy')
        k_sq = k_x**2 + k_y**2
        initial_scale = 1.0 / (k_sq + 1.0)
        self.register_buffer('spectral_decay', initial_scale.unsqueeze(0).repeat(dim, 1, 1))
        
        self.controller = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid() 
        )

    def forward(self, x, dt):
        B, T, D, H, W = x.shape
        x_energy = x.abs().mean(dim=(3, 4))
        noise_gain = self.controller(x_energy).view(B, T, D, 1, 1)
        
        noise_fft = torch.randn(B, T, D, H, W, dtype=torch.cfloat, device=x.device)
        scale = self.spectral_decay.unsqueeze(0).unsqueeze(0)
        noise_colored_fft = noise_fft * scale
        noise_colored = torch.fft.ifft2(noise_colored_fft, s=(H, W), norm='ortho')
        
        dt_shape = dt.view(B, T, 1, 1, 1)
        smooth_dt = F.softplus(dt_shape) + 1e-6
        diffusion = noise_colored * noise_gain * torch.sqrt(smooth_dt)
        return diffusion

class LyapunovController(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        target_dim = out_dim if out_dim is not None else dim

        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, target_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x_mag = x.abs()
        global_energy = x_mag.mean(dim=(3, 4))
        control_signal = self.net(global_energy)
        return control_signal

class TimeWarper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Tanh() 
        )

    def forward(self, x, dt):
        x_state = x.abs().mean(dim=(3, 4)) 
        warp_log = self.net(x_state).squeeze(-1) 
        warp_factor = torch.exp(warp_log * 0.5)
        return dt * warp_factor

class MetriplecticPropagator(nn.Module):
    def __init__(self, dim, img_height, img_width, dt_ref=1.0, stochastic=True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dt_ref = dt_ref

        self.L_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.M_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)

        self.stability_control = LyapunovController(dim, out_dim=dim * 2)
        self.time_warper = TimeWarper(dim)

        self.stochastic = stochastic
        if stochastic:
            self.noise_injector = SpectralNoiseInjector(dim, img_height, img_width)
        else:
            self.noise_injector = None

    def get_operators(self, dt, x_context=None):
        omega = self.L_generator.diagonal()
        gamma_static = self.M_generator.diagonal()

        if x_context is not None:
            control = self.stability_control(x_context)
            d_omega = control[..., :self.dim] * 0.1
            d_gamma = control[..., self.dim:] * 0.1
            
            omega_t = omega + d_omega
            gamma_t = gamma_static + d_gamma
            dt_eff = self.time_warper(x_context, dt)
        else:
            omega_t = omega
            gamma_t = gamma_static
            dt_eff = dt

        L_eigen = torch.complex(gamma_t, omega_t)

        if dt_eff.ndim == 2:
            dt_cast = dt_eff.unsqueeze(-1)
        elif dt_eff.ndim == 1:
            dt_cast = dt_eff.view(-1, 1, 1)
        else:
            dt_cast = dt_eff.unsqueeze(-1)

        evo_diag = torch.exp(L_eigen * dt_cast)

        Q = torch.eye(self.dim, device=dt.device, dtype=torch.cfloat)
        
        return Q, Q, evo_diag, dt_eff

    def inject_noise(self, x, dt_eff):
        if self.stochastic and self.noise_injector is not None:
            return self.noise_injector(x, dt_eff)
        return torch.zeros_like(x)
    
class StablePropagator(nn.Module):
    def __init__(self, dim, img_height, img_width, dt_ref=1.0, stochastic=True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dt_ref = dt_ref
        
        self.frequencies = nn.Parameter(torch.randn(dim) * 1.0)
        self.static_growth = nn.Parameter(torch.randn(dim) * 0.001 - 0.005)
        
        self.stability_control = LyapunovController(dim)
        self.time_warper = TimeWarper(dim)
        self.skew_generator = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
        self.stochastic = stochastic
        if stochastic:
            self.noise_injector = SpectralNoiseInjector(dim, img_height, img_width)
        else:
            self.noise_injector = None

    def get_orthogonal_basis(self):
        S = self.skew_generator.triu(1)
        S = S - S.t()
        Q = torch.matrix_exp(S)
        return Q

    def get_operators(self, dt, x_context=None):
        Q = self.get_orthogonal_basis()
        V = Q.to(dtype=torch.cfloat)
        V_inv = V.T 
        
        sigma_static = self.static_growth
        
        if x_context is not None:
            control_signal = self.stability_control(x_context)
            sigma_dynamic = control_signal * 0.01 
            sigma_total = sigma_static.view(1, 1, -1) + sigma_dynamic
            dt_eff = self.time_warper(x_context, dt)
        else:
            sigma_total = sigma_static.view(1, 1, -1)
            dt_eff = dt
            
        sigma_final = -F.softplus(-sigma_total)
        omega = self.frequencies.view(1, 1, -1)
        L = torch.complex(sigma_final, omega)
        
        if dt_eff.ndim == 2:
            dt_cast = dt_eff.unsqueeze(-1)
        elif dt_eff.ndim == 1:
            dt_cast = dt_eff.view(-1, 1, 1)
        else:
            dt_cast = dt_eff.unsqueeze(-1)
            
        evo_diag = torch.exp(L * dt_cast)
        return V, V_inv, evo_diag, dt_eff

    def inject_noise(self, x, dt_eff):
        if self.stochastic and self.noise_injector is not None:
            return self.noise_injector(x, dt_eff)
        return torch.zeros_like(x)

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

