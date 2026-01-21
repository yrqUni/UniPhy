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
        is_5d = (x.ndim == 5)
        if is_5d:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        if x.is_complex():
            x_real = x.real
            x_imag = x.imag
            x_mag = x.abs()
        else:
            x_real = x
            x_imag = None
            x_mag = x

        base_log = self.log_metric_param
        dynamic_log = self.metric_refiner(x_mag) * 0.1
        effective_log_metric = base_log + dynamic_log
        scale = torch.exp(effective_log_metric)
        inv_scale = torch.exp(-effective_log_metric)
        
        local_viscosity = self.viscosity_gate(x_mag) * inv_scale

        def apply_physics(feat):
            diffusion_term = F.conv2d(feat, self.laplacian_kernel, padding=1, groups=self.groups)
            x_diffused = feat + diffusion_term * local_viscosity * 0.01
            x_scaled = x_diffused * scale
            out = self.conv(x_scaled)
            out = out * inv_scale
            return out

        out_real = apply_physics(x_real)
        if x_imag is not None:
            out_imag = apply_physics(x_imag)
            out = torch.complex(out_real, out_imag)
        else:
            out = out_real
            
        if is_5d:
            out = out.view(B, T, *out.shape[1:])
            
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
        elif isinstance(dt, torch.Tensor) and dt.ndim == 5:
            dt_cast = dt
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
        global_energy = x_mag.mean(dim=(-2, -1))
        control_signal = self.net(global_energy)
        return control_signal

class CFLTimeWarper(nn.Module):
    def __init__(self, dim, dx=1.0, max_velocity=10.0):
        super().__init__()
        self.dx = dx
        self.max_velocity = max_velocity
        self.cfl_estimator = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, 3, 1, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, dt_ref):
        if x.is_complex():
            x_mag = x.abs()
        else:
            x_mag = torch.norm(x, dim=2)

        v_proxy = x_mag.mean(dim=(3, 4), keepdim=True) + 1e-6
        dt_max = self.dx / (v_proxy * self.max_velocity)
        
        B, T, C, H, W = x.shape
        x_in = x.reshape(B * T, C, H, W)
        if x_in.is_complex():
            x_in = torch.cat([x_in.real, x_in.imag], dim=1)
        
        alpha = self.cfl_estimator(x_in).view(B, T, 1, 1, 1)
        
        if isinstance(dt_ref, torch.Tensor):
            if dt_ref.ndim == 2 and dt_ref.shape[:2] == (B, T):
                 dt_ref = dt_ref.view(B, T, 1, 1, 1)
            elif dt_ref.ndim == 0:
                 dt_ref = dt_ref.view(1, 1, 1, 1, 1)

        dt_constrained = alpha * torch.minimum(dt_ref, dt_max)
        
        return dt_constrained
    
class MetriplecticPropagator(nn.Module):
    def __init__(self, dim, img_height, img_width, dt_ref=1.0, stochastic=True):
        super().__init__()
        self.dim = dim
        self.H = img_height
        self.W = img_width
        self.time_warper = CFLTimeWarper(dim)
        self.L_generator = nn.Parameter(torch.randn(dim, 1, 1) * 0.01)
        self.dissipation_base = nn.Parameter(torch.randn(dim, 1, 1) * 0.0)
        self.stability_control = LyapunovController(dim, out_dim=dim * 3)
        self.viscosity = nn.Parameter(torch.tensor(1e-3))
        
        ky = torch.fft.fftfreq(self.H, d=1.0)
        kx = torch.fft.rfftfreq(self.W, d=1.0)
        k_x, k_y = torch.meshgrid(kx, ky, indexing='xy')
        self.register_buffer('k_squared', (k_x**2 + k_y**2).unsqueeze(0))

        self.stochastic = stochastic
        if stochastic:
            self.noise_injector = SpectralNoiseInjector(dim, img_height, img_width)
        else:
            self.noise_injector = None

    def get_operators(self, dt, x_context):
        dt_eff = self.time_warper(x_context, dt)
        
        control = self.stability_control(x_context)
        d_omega = control[..., :self.dim] * 0.1
        valve_raw = control[..., 2*self.dim:]
        
        d_omega_expanded = d_omega.unsqueeze(-1).unsqueeze(-1)
        omega_t = self.L_generator + d_omega_expanded
        
        B_valve = torch.sigmoid(valve_raw) * 2.0
        
        gamma_base = -F.softplus(self.dissipation_base)
        nu_eff = F.softplus(self.viscosity)
        
        gamma_k = gamma_base - nu_eff * self.k_squared
        
        gamma_term = gamma_k.unsqueeze(0).unsqueeze(0) * dt_eff
        omega_term = omega_t * dt_eff
        
        A_decay = torch.exp(gamma_term)
        
        half_angle = 1j * omega_term * 0.5
        A_rot = (1 + half_angle) / (1 - half_angle)
        
        A_evo_spectral = A_decay * A_rot
        
        return None, B_valve, A_evo_spectral, dt_eff

    def inject_noise(self, x, dt_eff):
        if self.stochastic and self.noise_injector is not None:
            return self.noise_injector(x, dt_eff)
        return torch.zeros_like(x)

    def forward_spectral(self, z_in, dt, pscan_module=None):
        _, B_op, A_op, dt_eff = self.get_operators(dt, z_in)
        
        if z_in.is_complex():
            u_in = torch.cat([z_in.real, z_in.imag], dim=0)
            
            if A_op.shape[0] == z_in.shape[0]:
                A_op = torch.cat([A_op, A_op], dim=0)
            
            if B_op is not None and B_op.shape[0] == z_in.shape[0]:
                B_op = torch.cat([B_op, B_op], dim=0)
        else:
            u_in = z_in

        z_fft = torch.fft.rfft2(u_in, norm='ortho')

        if B_op is not None and B_op.ndim == 3:
             u_fft = z_fft * B_op.unsqueeze(-1).unsqueeze(-1)
        else:
             u_fft = z_fft

        B, T, C, H, W_f = z_fft.shape
        u_flat = u_fft.permute(0, 3, 4, 1, 2).reshape(B*H*W_f, T, C)
        A_flat = A_op.permute(0, 3, 4, 1, 2).reshape(B*H*W_f, T, C)
        
        return u_flat, A_flat, dt_eff, (B, T, C, H, W_f)

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
        is_5d = (x.ndim == 5)
        if is_5d:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            
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
            out = torch.complex(out_r, out_i)
        else:
            out = out_real

        if is_5d:
            out = out.view(B, T, *out.shape[1:])
            
        return out

