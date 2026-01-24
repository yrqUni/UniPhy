import torch
import torch.nn as nn
import torch.nn.functional as F
from PScan import PScanTriton

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
        self.laplacian_kernel = nn.Parameter(laplacian_init.reshape(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        
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
    
class GlobalFluxTracker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.decay_re = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)
        self.decay_im = nn.Parameter(torch.randn(dim) * 0.1)
        self.input_mix = nn.Linear(dim * 2, dim * 2)
        self.output_proj = nn.Linear(dim * 2, dim * 2)
        
        nn.init.xavier_uniform_(self.input_mix.weight)
        nn.init.zeros_(self.input_mix.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _get_decay(self):
        return torch.complex(torch.sigmoid(self.decay_re), self.decay_im)

    def get_operators(self, x_mean_seq):
        B, T, D = x_mean_seq.shape
        decay = self._get_decay() 
        
        x_flat = x_mean_seq.reshape(B * T, D)
        x_cat = torch.cat([x_flat.real, x_flat.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        u_t = torch.complex(x_re, x_im).reshape(B, T, D)
        
        A = decay.view(1, D, 1).expand(B, D, T).contiguous()
        X = u_t.permute(0, 2, 1).contiguous()
        
        return A, X

    def project(self, flux_states):
        B, D, T = flux_states.shape
        h_state = flux_states.permute(0, 2, 1) 
        
        h_flat = h_state.reshape(B * T, D)
        out_cat = self.output_proj(torch.cat([h_flat.real, h_flat.imag], dim=-1))
        out_re, out_im = torch.chunk(out_cat, 2, dim=-1)
        source_seq = torch.complex(out_re, out_im).reshape(B, T, D)
        
        return source_seq

    def forward_step(self, flux_state, x_mean):
        x_cat = torch.cat([x_mean.real, x_mean.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        x_complex = torch.complex(x_re, x_im)
        
        decay = self._get_decay()
        new_state = flux_state * decay + x_complex
        
        out_cat = self.output_proj(torch.cat([new_state.real, new_state.imag], dim=-1))
        out_re, out_im = torch.chunk(out_cat, 2, dim=-1)
        source = torch.complex(out_re, out_im)
        
        return new_state, source

class TemporalPropagator(nn.Module):
    def __init__(self, dim, dt_ref=1.0, noise_scale=0.01):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.noise_scale = noise_scale
        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim)
        
        self.ld = nn.Parameter(torch.randn(dim) * 0.5 - 2.0)
        self.lf = nn.Parameter(torch.randn(dim) * 1.0)
        self.law_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.law_im = nn.Parameter(torch.randn(dim) * 0.01)

    def _get_effective_lambda(self):
        l_phys = torch.complex(-torch.exp(self.ld), self.lf)
        l_law = torch.complex(self.law_re, self.law_im)
        l_total = l_phys + l_law
        stable_re = -F.softplus(-l_total.real)
        return torch.complex(stable_re, l_total.imag)

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
        var = (self.noise_scale ** 2) * torch.expm1(2 * l_re * (dt / self.dt_ref).unsqueeze(-1)) / (2 * l_re)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var).to(dtype)
        noise = torch.randn(target_shape, device=self.ld.device, dtype=dtype)
        return noise * std

    def forward_step(self, h_prev, x_input, x_global_mean_encoded, dt, flux_state):
        h_tilde = self.basis.encode(h_prev)
        if h_tilde.ndim == 2: h_tilde = h_tilde.unsqueeze(1)
        x_tilde = self.basis.encode(x_input)
        if x_tilde.ndim == 2: x_tilde = x_tilde.unsqueeze(1)

        flux_next, source = self.flux_tracker.forward_step(flux_state, x_global_mean_encoded)
        
        op_decay, op_forcing = self.get_transition_operators(dt)
        
        B = source.shape[0]
        total_batch = x_tilde.shape[0]
        D = x_tilde.shape[-1]
        
        if total_batch % B != 0:
             raise ValueError(f"Total batch size {total_batch} is not divisible by flux batch size {B}")
        
        spatial_size = total_batch // B
        
        source_expanded = source.view(B, 1, 1, D).expand(B, spatial_size, 1, D).reshape(total_batch, 1, D)
        
        forcing_term = x_tilde + source_expanded
        
        h_tilde_next = h_tilde * op_decay + forcing_term * op_forcing
        h_tilde_next = h_tilde_next + self.generate_stochastic_term(h_tilde_next.shape, dt, h_tilde_next.dtype)
        
        return self.basis.decode(h_tilde_next), flux_next
    