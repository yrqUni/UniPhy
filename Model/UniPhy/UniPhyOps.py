import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class GatedChannelMixer(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expand)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = torch.chunk(self.fc1(x), 2, dim=-1)
        hidden = self.act(x1) * x2
        hidden = self.drop(hidden)
        out = self.fc2(hidden)
        out = self.drop(out)
        return out

class RiemannianCliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, img_height, img_width):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.log_metric_param = nn.Parameter(torch.zeros(1, 1, img_height, img_width))
        self.metric_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
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
        
        if isinstance(dt, torch.Tensor) and dt.ndim == 4:
            dt_cast = dt.unsqueeze(1)
        else:
            dt_cast = dt

        smooth_dt = F.softplus(dt_cast) + 1e-6
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
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x, dt):
        local_energy = x.abs().mean(dim=(1, 2), keepdim=True).squeeze(1)
        
        warp_log = self.net(local_energy) 
        warp_factor = torch.exp(warp_log * 2.0)
        
        if isinstance(dt, torch.Tensor):
             if dt.ndim == 1:
                 dt = dt.view(-1, 1, 1, 1)
             elif dt.ndim == 0:
                 dt = dt.view(1, 1, 1, 1)
        return dt * warp_factor
    
class MetriplecticPropagator(nn.Module):
    def __init__(self, dim, img_height, img_width, dt_ref=1.0, stochastic=True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.dt_ref = dt_ref
        self.L_generator = nn.Parameter(torch.randn(dim) * 0.01)
        self.dissipation_base = nn.Parameter(torch.randn(dim) * 0.0)
        self.stability_control = LyapunovController(dim, out_dim=dim * 3)
        self.time_warper = TimeWarper(dim)
        self.stochastic = stochastic
        if stochastic:
            self.noise_injector = SpectralNoiseInjector(dim, img_height, img_width)
        else:
            self.noise_injector = None

    def get_operators(self, dt, x_context=None):
        omega_base = self.L_generator
        gamma_base = -F.softplus(self.dissipation_base)

        if x_context is not None:
            control = self.stability_control(x_context)
            d_omega = control[..., :self.dim] * 0.1
            d_gamma = control[..., self.dim : 2*self.dim]
            valve_raw = control[..., 2*self.dim:]
            
            omega_t = omega_base + d_omega
            gamma_t = -F.softplus(self.dissipation_base + d_gamma)
            B_valve = torch.sigmoid(valve_raw) * 2.0
            
            dt_eff = self.time_warper(x_context, dt)
        else:
            omega_t = omega_base
            gamma_t = gamma_base
            B_valve = torch.ones_like(omega_base)
            dt_eff = dt

        L_eigen = torch.complex(gamma_t, omega_t)
        
        if dt_eff.ndim == 2: 
            dt_cast = dt_eff.unsqueeze(-1)
        elif dt_eff.ndim == 1: 
            dt_cast = dt_eff.view(-1, 1, 1)
        elif dt_eff.ndim == 4: 
            dt_cast = dt_eff.permute(0, 2, 3, 1).unsqueeze(-1)
            if L_eigen.ndim == 3:
                L_eigen = L_eigen.unsqueeze(1).unsqueeze(1)
        else: 
            dt_cast = dt_eff.unsqueeze(-1)

        A_evo = torch.exp(L_eigen * dt_cast)
        return None, B_valve, A_evo, dt_eff

    def inject_noise(self, x, dt_eff):
        if self.stochastic and self.noise_injector is not None:
            return self.noise_injector(x, dt_eff)
        return torch.zeros_like(x)

class SpectralStep(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.pad_h = h // 4 
        self.total_h = h + 2 * self.pad_h
        self.freq_h = self.total_h 
        self.freq_w = w // 2 + 1 
        self.weight = nn.Parameter(torch.randn(dim, self.freq_h, self.freq_w, dtype=torch.cfloat) * 0.02)
        self.viscosity_x = nn.Parameter(torch.tensor(1e-4))
        self.viscosity_y = nn.Parameter(torch.tensor(1e-4))
        self.alpha = nn.Parameter(torch.tensor(2.0))
        kx = torch.fft.rfftfreq(w, d=1.0)
        ky = torch.fft.fftfreq(self.total_h, d=1.0)
        k_x, k_y = torch.meshgrid(kx, ky, indexing='xy')
        self.register_buffer('k_x_abs', k_x.abs())
        self.register_buffer('k_y_abs', k_y.abs())

    def forward(self, x):
        is_complex = x.is_complex()
        if is_complex:
            x_in = torch.cat([x.real, x.imag], dim=0)
        else:
            x_in = x

        B_in, C, H, W = x_in.shape
        x_pad = F.pad(x_in, (0, 0, self.pad_h, self.pad_h), mode='reflect')
        x_fft = torch.fft.rfft2(x_pad, norm='ortho')
        
        original_dc = x_fft[..., 0, 0].clone()
        x_fft = x_fft * self.weight
        
        alpha_clamp = self.alpha.clamp(1.0, 3.0)
        vx = self.viscosity_x.abs()
        vy = self.viscosity_y.abs()
        
        k_term = vx * torch.pow(self.k_x_abs, alpha_clamp) + vy * torch.pow(self.k_y_abs, alpha_clamp)
        dissipation = torch.exp(-k_term * (self.total_h * W))
        
        x_fft = x_fft * dissipation.unsqueeze(0).unsqueeze(0)
        x_fft[..., 0, 0] = original_dc
        
        out_pad = torch.fft.irfft2(x_fft, s=(self.total_h, W), norm='ortho')
        
        if self.pad_h > 0:
            out_real = out_pad[..., self.pad_h : -self.pad_h, :]
        else:
            out_real = out_pad
            
        if is_complex:
            out_r, out_i = torch.chunk(out_real, 2, dim=0)
            return torch.complex(out_r, out_i)
        else:
            return out_real

