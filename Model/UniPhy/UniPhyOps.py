import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexSVDTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.u_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.u_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_re = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_raw_im = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(dim))

        n = torch.arange(dim).unsqueeze(1).float()
        k = torch.arange(dim).unsqueeze(0).float()
        dft_matrix = torch.exp(-2j * torch.pi * n * k / dim) / (dim ** 0.5)
        self.register_buffer("dft_basis", dft_matrix)
        self.dft_weight = nn.Parameter(torch.tensor(0.2))

    def _cayley_orthogonalize(self, raw_re, raw_im):
        A = torch.complex(raw_re, raw_im)
        A_skew = A - A.T.conj()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        I_reg = I * 1e-6
        Q = torch.linalg.solve(I + A_skew + I_reg, I - A_skew)
        return Q

    def _get_basis(self):
        U = self._cayley_orthogonalize(self.u_raw_re, self.u_raw_im)
        V = self._cayley_orthogonalize(self.v_raw_re, self.v_raw_im)
        S = torch.diag(torch.exp(self.log_sigma).to(U.dtype))
        return U, S, V

    def encode(self, x):
        U, S_mat, V = self._get_basis()
        S_diag = torch.diag(S_mat)

        learned_path = torch.einsum("...d, de -> ...e", x, V) * S_diag

        if x.is_complex():
            x_complex = x
        else:
            x_complex = torch.complex(x, torch.zeros_like(x))

        dft_basis = self.dft_basis.to(dtype=x_complex.dtype)
        dft_path = torch.einsum("...d, de -> ...e", x_complex, dft_basis)

        alpha = torch.sigmoid(self.dft_weight)
        combined = alpha * dft_path + (1 - alpha) * learned_path

        return combined

    def decode(self, z):
        U, S_mat, V = self._get_basis()
        S_diag = torch.diag(S_mat)

        S_inv = 1.0 / (S_diag + 1e-8)
        V_H = V.conj().T

        learned_path = torch.einsum("...d, de -> ...e", z * S_inv, V_H)

        dft_basis_H = self.dft_basis.conj().T.to(dtype=z.dtype)
        dft_path = torch.einsum("...d, de -> ...e", z, dft_basis_H)

        alpha = torch.sigmoid(self.dft_weight)
        combined = alpha * dft_path + (1 - alpha) * learned_path

        return combined


class LearnableSpectralBasis(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform = ComplexSVDTransform(dim)

    def encode(self, x):
        return self.transform.encode(x)

    def decode(self, z):
        return self.transform.decode(z)


class GlobalFluxTracker(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim * 2

        self.flux_update = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.source_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

        self.decay = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, flux_prev):
        if x.is_complex():
            x_real = torch.cat([x.real, x.imag], dim=-1)
        else:
            x_real = torch.cat([x, torch.zeros_like(x)], dim=-1)

        if flux_prev.is_complex():
            flux_real = torch.cat([flux_prev.real, flux_prev.imag], dim=-1)
        else:
            flux_real = torch.cat([flux_prev, torch.zeros_like(flux_prev)], dim=-1)

        B = x.shape[0]
        if x.ndim == 4:
            x_mean = x_real.mean(dim=(1, 2))
        else:
            x_mean = x_real.mean(dim=1) if x_real.ndim > 2 else x_real

        if flux_real.ndim == 1:
            flux_real = flux_real.unsqueeze(0).expand(B, -1)

        flux_input = torch.cat([x_mean, flux_real], dim=-1)
        flux_delta = self.flux_update(flux_input)

        decay_factor = torch.sigmoid(self.decay)
        if flux_prev.is_complex():
            flux_next_real = flux_prev.real * decay_factor + flux_delta
            flux_next = torch.complex(flux_next_real, flux_prev.imag * decay_factor)
        else:
            flux_next = flux_prev * decay_factor + flux_delta

        gate_input = torch.cat([x_mean, flux_real], dim=-1)
        gate = self.gate_net(gate_input)

        return flux_next, gate

    def forward_step(self, flux_prev, x_mean):
        if x_mean.is_complex():
            x_real = torch.cat([x_mean.real, x_mean.imag], dim=-1)
        else:
            x_real = torch.cat([x_mean, torch.zeros_like(x_mean)], dim=-1)

        if flux_prev.is_complex():
            flux_real = torch.cat([flux_prev.real, flux_prev.imag], dim=-1)
        else:
            flux_real = torch.cat([flux_prev, torch.zeros_like(flux_prev)], dim=-1)

        flux_input = torch.cat([x_real, flux_real], dim=-1)
        flux_delta = self.flux_update(flux_input)

        decay_factor = torch.sigmoid(self.decay)
        if flux_prev.is_complex():
            flux_next_real = flux_prev.real * decay_factor + flux_delta
            flux_next = torch.complex(flux_next_real, flux_prev.imag * decay_factor)
        else:
            flux_next = flux_prev * decay_factor + flux_delta

        source_input = torch.cat([x_real, flux_real], dim=-1)
        source = self.source_net(source_input)

        gate_input = torch.cat([x_real, flux_real], dim=-1)
        gate = self.gate_net(gate_input)

        if x_mean.is_complex():
            source = torch.complex(source, torch.zeros_like(source))

        return flux_next, source, gate


class TemporalPropagator(nn.Module):
    def __init__(
        self,
        dim,
        dt_ref=1.0,
        sde_mode="sde",
        init_noise_scale=0.01,
        max_growth_rate=0.3,
    ):
        super().__init__()
        self.dim = dim
        self.dt_ref = dt_ref
        self.sde_mode = sde_mode
        self.init_noise_scale = init_noise_scale
        self.max_growth_rate = max_growth_rate

        self.basis = ComplexSVDTransform(dim)
        self.flux_tracker = GlobalFluxTracker(dim)

        self.lam_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.lam_im = nn.Parameter(torch.randn(dim) * 0.1)

        if sde_mode == "sde":
            self.base_noise = nn.Parameter(torch.ones(dim) * init_noise_scale)
            self.uncertainty_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
                nn.Softplus(),
            )
        else:
            self.register_buffer("base_noise", torch.tensor(0.0))
            self.uncertainty_net = None

    def _get_effective_lambda(self):
        lam_re_bounded = self.lam_re.clamp(-self.max_growth_rate, self.max_growth_rate)
        return torch.complex(lam_re_bounded, self.lam_im)

    def get_transition_operators(self, dt):
        lam = self._get_effective_lambda()

        if isinstance(dt, torch.Tensor):
            dt_tensor = dt
        else:
            dt_tensor = torch.tensor(dt, device=lam.device, dtype=torch.float32)

        if dt_tensor.is_complex():
            dt_tensor = dt_tensor.real

        if dt_tensor.ndim == 0:
            dt_ratio = dt_tensor / self.dt_ref
            dt_ratio = dt_ratio.unsqueeze(0).unsqueeze(0)
        elif dt_tensor.ndim == 1:
            dt_ratio = dt_tensor.unsqueeze(-1) / self.dt_ref
        elif dt_tensor.ndim == 2:
            dt_ratio = dt_tensor.unsqueeze(-1) / self.dt_ref
        else:
            dt_ratio = dt_tensor / self.dt_ref

        dt_ratio = dt_ratio.to(lam.device)
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)

        lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
        forcing = torch.expm1(exp_arg) / lam_safe

        return decay, forcing

    def generate_stochastic_term(self, shape, dt, dtype, h_state=None):
        if self.sde_mode != "sde":
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)

        device = self.lam_re.device

        if isinstance(dt, torch.Tensor):
            dt_tensor = dt.to(device)
        else:
            dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)

        if dt_tensor.is_complex():
            dt_tensor = dt_tensor.real

        base_scale = self.base_noise.abs() * torch.sqrt(dt_tensor.abs().float() + 1e-8)

        noise_real = torch.randn(shape, device=device, dtype=torch.float32)
        noise_imag = torch.randn(shape, device=device, dtype=torch.float32)

        if h_state is not None and self.uncertainty_net is not None:
            h_flat = h_state.reshape(-1, self.dim)
            if h_flat.is_complex():
                h_real = torch.cat([h_flat.real, h_flat.imag], dim=-1)
            else:
                h_real = h_flat
            h_mag = h_real.abs().mean(dim=-1, keepdim=True)
            uncertainty = self.uncertainty_net(h_mag.expand(-1, self.dim))
            uncertainty = uncertainty.reshape(shape)
            scale = base_scale * (1 + uncertainty)
        else:
            scale = base_scale

        while scale.ndim < noise_real.ndim:
            scale = scale.unsqueeze(0)

        scaled_real = noise_real * scale
        scaled_imag = noise_imag * scale

        if dtype.is_complex:
            noise = torch.complex(scaled_real, scaled_imag)
        else:
            noise = scaled_real

        return noise


class RiemannianCliffordConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.img_height = img_height
        self.img_width = img_width

        self.conv_e0 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv_e1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv_e2 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv_e12 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )

        self.bias = nn.Parameter(torch.zeros(out_channels))

        lat = torch.linspace(-math.pi / 2, math.pi / 2, img_height)
        self.register_buffer("cos_lat", torch.cos(lat).view(1, 1, -1, 1))

        self.metric_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        y_e0 = self.conv_e0(x)
        y_e1 = self.conv_e1(x)
        y_e2 = self.conv_e2(x)
        y_e12 = self.conv_e12(x)

        cos_lat = self.cos_lat
        if H != self.img_height:
            cos_lat = F.interpolate(
                cos_lat, size=(H, 1), mode="bilinear", align_corners=False
            )
        cos_lat = cos_lat.expand(B, -1, H, W)

        y_e1_scaled = y_e1 * cos_lat
        y_e12_scaled = y_e12 * cos_lat

        out = y_e0 + y_e1_scaled + y_e2 + y_e12_scaled
        out = out * self.metric_scale
        out = out + self.bias.view(1, -1, 1, 1)

        return out


class SphericalPositionEmbedding(nn.Module):
    def __init__(self, dim, h_dim, w_dim):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.lat_emb = nn.Parameter(torch.randn(1, dim // 2, h_dim, 1) * 0.02)
        self.lon_emb = nn.Parameter(torch.randn(1, dim // 2, 1, w_dim) * 0.02)

        lat = torch.linspace(-math.pi / 2, math.pi / 2, h_dim)
        lon = torch.linspace(0, 2 * math.pi, w_dim)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        self.register_buffer("cos_lat", torch.cos(lat_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_lat", torch.sin(lat_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("cos_lon", torch.cos(lon_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_lon", torch.sin(lon_grid).unsqueeze(0).unsqueeze(0))

        self.geo_proj = nn.Conv2d(4, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        lat_emb = self.lat_emb
        lon_emb = self.lon_emb

        if H != self.h_dim or W != self.w_dim:
            lat_emb = F.interpolate(
                lat_emb, size=(H, 1), mode="bilinear", align_corners=False
            )
            lon_emb = F.interpolate(
                lon_emb, size=(1, W), mode="bilinear", align_corners=False
            )

        pos_emb = torch.cat([
            lat_emb.expand(B, -1, H, W),
            lon_emb.expand(B, -1, H, W),
        ], dim=1)

        cos_lat = self.cos_lat
        sin_lat = self.sin_lat
        cos_lon = self.cos_lon
        sin_lon = self.sin_lon

        if H != self.h_dim or W != self.w_dim:
            cos_lat = F.interpolate(cos_lat, size=(H, W), mode="bilinear", align_corners=False)
            sin_lat = F.interpolate(sin_lat, size=(H, W), mode="bilinear", align_corners=False)
            cos_lon = F.interpolate(cos_lon, size=(H, W), mode="bilinear", align_corners=False)
            sin_lon = F.interpolate(sin_lon, size=(H, W), mode="bilinear", align_corners=False)

        geo_features = torch.cat([cos_lat, sin_lat, cos_lon, sin_lon], dim=1)
        geo_features = geo_features.expand(B, -1, -1, -1)
        geo_emb = self.geo_proj(geo_features)

        out = x + pos_emb + geo_emb

        return out


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.dim = dim
        cond_dim = cond_dim or dim

        self.norm = nn.LayerNorm(dim)
        self.scale_proj = nn.Linear(cond_dim, dim)
        self.shift_proj = nn.Linear(cond_dim, dim)

        nn.init.zeros_(self.scale_proj.weight)
        nn.init.ones_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x, cond=None):
        x_norm = self.norm(x)

        if cond is not None:
            scale = self.scale_proj(cond)
            shift = self.shift_proj(cond)

            while scale.ndim < x_norm.ndim:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)

            x_norm = x_norm * scale + shift

        return x_norm


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight_re = nn.Parameter(torch.ones(dim))
        self.weight_im = nn.Parameter(torch.zeros(dim))
        self.bias_re = nn.Parameter(torch.zeros(dim))
        self.bias_im = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean

        var = (x_centered * x_centered.conj()).real.mean(dim=-1, keepdim=True)
        x_norm = x_centered / torch.sqrt(var + self.eps)

        weight = torch.complex(self.weight_re, self.weight_im)
        bias = torch.complex(self.bias_re, self.bias_im)

        out = x_norm * weight + bias

        return out
    