import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from TritonOps import FusedHamiltonian, fused_curl_2d

class CliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_s = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.weight_x = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.weight_y = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.weight_b = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.padding = padding

    def forward(self, x):
        s_in, x_in, y_in, b_in = torch.chunk(x, 4, dim=1)
        s_out = (F.conv2d(s_in, self.weight_s, padding=self.padding) + 
                 F.conv2d(x_in, self.weight_x, padding=self.padding) + 
                 F.conv2d(y_in, self.weight_y, padding=self.padding) - 
                 F.conv2d(b_in, self.weight_b, padding=self.padding))
        x_out = (F.conv2d(s_in, self.weight_x, padding=self.padding) + 
                 F.conv2d(x_in, self.weight_s, padding=self.padding) - 
                 F.conv2d(y_in, self.weight_b, padding=self.padding) + 
                 F.conv2d(b_in, self.weight_y, padding=self.padding))
        y_out = (F.conv2d(s_in, self.weight_y, padding=self.padding) + 
                 F.conv2d(x_in, self.weight_b, padding=self.padding) + 
                 F.conv2d(y_in, self.weight_s, padding=self.padding) - 
                 F.conv2d(b_in, self.weight_x, padding=self.padding))
        b_out = (F.conv2d(s_in, self.weight_b, padding=self.padding) + 
                 F.conv2d(x_in, self.weight_y, padding=self.padding) - 
                 F.conv2d(y_in, self.weight_x, padding=self.padding) + 
                 F.conv2d(b_in, self.weight_s, padding=self.padding))
        return torch.cat([s_out, x_out, y_out, b_out], dim=1)

class StochasticHamiltonianPropagator(nn.Module):
    def __init__(self, hidden_dim, h, w):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_h = h
        self.freq_w = w 
        self.hamiltonian_real = nn.Parameter(torch.randn(hidden_dim, self.freq_h, self.freq_w) * 0.01)
        self.hamiltonian_imag = nn.Parameter(torch.randn(hidden_dim, self.freq_h, self.freq_w) * 0.01)
        self.noise_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z, dt):
        z_spec = torch.fft.fft2(z, norm='ortho')
        sigma = F.softplus(self.noise_scale)
        z_next_r, z_next_i = FusedHamiltonian.apply(
            z_spec.real, z_spec.imag, 
            self.hamiltonian_real, self.hamiltonian_imag, 
            dt, sigma
        )
        z_shifted_spec = torch.complex(z_next_r, z_next_i)
        z_shifted = torch.fft.ifft2(z_shifted_spec, s=(self.freq_h, self.freq_w), norm='ortho')
        return z_shifted

class StreamFunctionConstraint(nn.Module):
    def __init__(self, in_ch, h, w):
        super().__init__()
        self.psi_net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        self.register_buffer('grid_base', torch.stack((xx, yy), dim=-1))

    def forward(self, z, dt):
        B, C, H, W = z.shape
        psi = self.psi_net(z.real) * dt
        if not self.training and z.is_cuda:
            u, v = fused_curl_2d(psi)
        else:
            u, v = self._curl_pytorch(psi)
        flow_norm = torch.cat([u / (W/2), v / (H/2)], dim=1).permute(0, 2, 3, 1)
        grid = self.grid_base.unsqueeze(0).expand(B, -1, -1, -1)
        sampling_grid = grid - flow_norm
        z_real = F.grid_sample(z.real, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        z_imag = F.grid_sample(z.imag, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        return torch.complex(z_real, z_imag)

    def _curl_pytorch(self, psi):
        psi_pad = F.pad(psi, (1, 1, 1, 1), mode='replicate')
        u = (psi_pad[:, :, 2:, 1:-1] - psi_pad[:, :, :-2, 1:-1]) / 2.0
        v = -(psi_pad[:, :, 1:-1, 2:] - psi_pad[:, :, 1:-1, :-2]) / 2.0
        return u, v

class GeoSymSDE(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim=8, h=64, w=128):
        super().__init__()
        self.clifford_encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim * 4, 1),
            CliffordConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU()
        )
        self.to_complex_real = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)
        self.to_complex_imag = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)
        self.stochastic_op = StochasticHamiltonianPropagator(hidden_dim, h, w)
        self.stream_op = StreamFunctionConstraint(hidden_dim, h, w)
        self.from_complex = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 1)
        self.clifford_decoder = nn.Sequential(
            CliffordConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, out_ch, 1)
        )

    def forward(self, x, dt=1.0):
        geo_feat = self.clifford_encoder(x)
        z_real = self.to_complex_real(geo_feat)
        z_imag = self.to_complex_imag(geo_feat)
        z = torch.complex(z_real, z_imag)
        z_half = self.stochastic_op(z, dt / 2.0)
        z_adv = self.stream_op(z_half, dt)
        z_final = self.stochastic_op(z_adv, dt / 2.0)
        z_cat = torch.cat([z_final.real, z_final.imag], dim=1)
        rec_feat = self.from_complex(z_cat)
        out = self.clifford_decoder(rec_feat + geo_feat)
        return x + out

