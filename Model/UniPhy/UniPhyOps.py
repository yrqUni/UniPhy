import torch
import torch.nn as nn
import math
from PScan import PScanTriton
from CliffordConv2d import CliffordConv2d
from SpectralStep import SpectralStep

class HamiltonianPropagator(nn.Module):
    def __init__(self, d_model, conserve_energy=True):
        super().__init__()
        self.d_model = d_model
        self.pscan = PScanTriton.apply
        freq_init = torch.exp(torch.linspace(math.log(1.0), math.log(100.0), d_model))
        self.log_freq = nn.Parameter(torch.log(freq_init))
        self.conserve_energy = conserve_energy
        if not conserve_energy:
            self.log_decay = nn.Parameter(torch.log(0.01 * torch.ones(d_model)))
        self.input_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def _get_evolution_op(self, dt):
        freq = torch.exp(self.log_freq).to(dt.device)
        if self.conserve_energy:
            decay = 0.0
        else:
            decay = torch.exp(self.log_decay).to(dt.device)
        if dt.dim() < freq.dim():
            dt = dt.unsqueeze(-1)
        mag = torch.exp(-decay * dt)
        phase = freq * dt
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.complex(real, imag)

    def forward_parallel(self, x, dt, initial_state=None):
        B, T, H, W, C = x.shape
        x_flat = x.view(B, T, -1, C).permute(0, 2, 1, 3).reshape(-1, T, C)
        dt_flat = dt.view(B, T, 1, 1).expand(B, T, H * W, 1).permute(0, 2, 1, 3).reshape(-1, T, 1)
        A = self._get_evolution_op(dt_flat)
        x_proj = self.input_proj(x_flat)
        x_r, x_i = torch.chunk(x_proj, 2, dim=-1)
        X_val = torch.complex(x_r, x_i) * dt_flat
        if initial_state is not None:
            h_prev_flat = initial_state.view(-1, C)
            X_val[:, 0, :] = X_val[:, 0, :] + A[:, 0, :] * h_prev_flat
        h_states = self.pscan(A, X_val)
        h_final = h_states[:, -1, :].view(B, H, W, C)
        h_cat = torch.cat([h_states.real, h_states.imag], dim=-1)
        out = self.output_proj(h_cat)
        out = out.view(B, H * W, T, C).permute(0, 2, 1, 3).view(B, T, H, W, C)
        return out, h_final

    def step_serial(self, x_step, dt_step, h_prev):
        B, H, W, C = x_step.shape
        x_flat = x_step.view(-1, C)
        h_prev_flat = h_prev.view(-1, C)
        dt_flat = dt_step.view(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
        
        A = self._get_evolution_op(dt_flat)
        
        x_proj = self.input_proj(x_flat)
        x_r, x_i = torch.chunk(x_proj, 2, dim=-1)
        X_val = torch.complex(x_r, x_i) * dt_flat
        
        h_next_flat = A * h_prev_flat + X_val
        h_cat = torch.cat([h_next_flat.real, h_next_flat.imag], dim=-1)
        out = self.output_proj(h_cat)
        out = out.view(B, H, W, C)
        h_next = h_next_flat.view(B, H, W, C)
        return out, h_next

    def get_initial_state(self, x):
        B, H, W, C = x.shape
        return torch.zeros(B, H, W, self.d_model, dtype=torch.complex64, device=x.device)

class UniPhyBlock(nn.Module):
    def __init__(self, dim, input_shape):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.propagator = HamiltonianPropagator(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.space_conv = CliffordConv2d(dim, kernel_size=3, padding=1)
        self.space_spectral = SpectralStep(dim, rank=32, w_freq=input_shape[1] // 2 + 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward_parallel(self, x, dt, initial_state=None):
        residual = x
        x = self.norm1(x)
        x, h_final = self.propagator.forward_parallel(x, dt, initial_state)
        x = x + residual
        residual = x
        x = self.norm2(x)
        B, T, H, W, C = x.shape
        x_reshaped = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x_conv = self.space_conv(x_reshaped)
        dt_expanded = dt.view(-1)
        x_spec = self.space_spectral(x_reshaped, dt_expanded)
        x_spatial = x_conv + x_spec
        x_spatial = x_spatial.permute(0, 2, 3, 1).view(B, T, H, W, C)
        x = self.mlp(x_spatial)
        x = x + residual
        return x, h_final

    def step_serial(self, x, dt, h_prev):
        residual = x
        x = self.norm1(x)
        x, h_next = self.propagator.step_serial(x, dt, h_prev)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x_in = x.permute(0, 3, 1, 2)
        x_conv = self.space_conv(x_in)
        x_spec = self.space_spectral(x_in, dt)
        x_spatial = x_conv + x_spec
        x_spatial = x_spatial.permute(0, 2, 3, 1)
        x = self.mlp(x_spatial)
        x = x + residual
        return x, h_next

class UniPhyNet(nn.Module):
    def __init__(self, in_ch, dim, input_shape, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(in_ch, dim)
        self.layers = nn.ModuleList([
            UniPhyBlock(dim, input_shape) for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim, in_ch)

    def forward_parallel(self, x, dt):
        x = x.permute(0, 1, 3, 4, 2)
        x = self.embedding(x)
        final_states = []
        for layer in self.layers:
            x, h_final = layer.forward_parallel(x, dt)
            final_states.append(h_final)
        out = self.head(x)
        return out.permute(0, 1, 4, 2, 3), final_states

    def step_serial(self, x, dt, prev_states):
        x = x.permute(0, 2, 3, 1)
        x = self.embedding(x)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, h_next = layer.step_serial(x, dt, prev_states[i])
            next_states.append(h_next)
        out = self.head(x)
        return out.permute(0, 3, 1, 2), next_states

class UniPhySolver(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def inference(self, context_x, context_dt, future_steps, future_dt):
        B, T_ctx, C, H, W = context_x.shape
        _, current_states = self.model.forward_parallel(context_x, context_dt)
        curr_x = context_x[:, -1]
        predictions = []
        if isinstance(future_dt, float):
            future_dt = torch.full((B,), future_dt, device=context_x.device)
        for _ in range(future_steps):
            curr_x, current_states = self.model.step_serial(curr_x, future_dt, current_states)
            predictions.append(curr_x)
        return torch.stack(predictions, dim=1)

