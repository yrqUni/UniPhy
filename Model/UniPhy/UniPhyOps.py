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

        n = torch.arange(dim).float()
        k = torch.arange(dim).float().reshape(-1, 1)
        dft_matrix = torch.exp(-2j * torch.pi * n * k / dim) / (dim ** 0.5)
        self.register_buffer("dft_basis", dft_matrix)
        self.dft_weight = nn.Parameter(torch.tensor(0.0))

    def _cayley_orthogonalize(self, raw_re, raw_im):
        A = torch.complex(raw_re, raw_im)
        A_skew = A - A.T.conj()
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        Q = torch.linalg.solve(I + A_skew, I - A_skew)
        return Q

    def _get_basis(self):
        U = self._cayley_orthogonalize(self.u_raw_re, self.u_raw_im)
        V = self._cayley_orthogonalize(self.v_raw_re, self.v_raw_im)
        S = torch.exp(self.log_sigma)
        return U, S, V

    def _get_transform_matrix(self, dtype):
        U, S, V = self._get_basis()
        alpha = torch.sigmoid(self.dft_weight)

        V_scaled = V @ torch.diag(S.to(V.dtype))
        dft_basis = self.dft_basis.to(dtype=dtype)

        W = V_scaled * (1 - alpha) + dft_basis * alpha
        return W

    def encode(self, x):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        W = self._get_transform_matrix(x.dtype)
        h = torch.einsum("...d,de->...e", x, W)
        return h

    def decode(self, h):
        W = self._get_transform_matrix(h.dtype)
        W_inv = torch.linalg.inv(W)
        x = torch.einsum("...d,de->...e", h, W_inv)
        return x


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

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.gate_min = 0.01
        self.gate_max = 0.99

    def _get_decay(self):
        return torch.complex(torch.sigmoid(self.decay_re), self.decay_im)

    def get_operators(self, x_mean_seq):
        B, T, D = x_mean_seq.shape
        decay = self._get_decay()

        x_flat = x_mean_seq.reshape(B * T, D)
        x_cat = torch.cat([x_flat.real, x_flat.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)

        X = torch.complex(x_re, x_im).reshape(B, T, D).contiguous()
        A = decay.unsqueeze(0).unsqueeze(0).expand(B, T, D).contiguous()

        return A, X

    def project(self, state):
        s_cat = torch.cat([state.real, state.imag], dim=-1)
        out = self.output_proj(s_cat)
        out_re, out_im = torch.chunk(out, 2, dim=-1)
        return torch.complex(out_re, out_im)

    def forward_step(self, prev_state, x_t):
        decay = self._get_decay()

        if not x_t.is_complex():
            x_t = torch.complex(x_t, torch.zeros_like(x_t))

        x_cat = torch.cat([x_t.real, x_t.imag], dim=-1)
        x_in = self.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        x_mixed = torch.complex(x_re, x_im)

        new_state = prev_state * decay + x_mixed

        source = self.project(new_state)

        gate_input = torch.cat([new_state.real, new_state.imag], dim=-1)
        gate = self.gate_net(gate_input)
        gate = gate * (self.gate_max - self.gate_min) + self.gate_min

        return new_state, source, gate


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
        dt_tensor = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=lam.device)

        if dt_tensor.is_complex():
            dt_tensor = dt_tensor.real

        if dt_tensor.ndim == 0:
            dt_ratio = dt_tensor / self.dt_ref
        elif dt_tensor.ndim == 1:
            dt_ratio = dt_tensor.unsqueeze(-1) / self.dt_ref
        elif dt_tensor.ndim == 2:
            dt_ratio = dt_tensor.unsqueeze(-1) / self.dt_ref
        else:
            dt_ratio = dt_tensor / self.dt_ref

        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)

        lam_safe = lam + 1e-8 * torch.sign(lam.real + 1e-12)
        forcing = torch.expm1(exp_arg) / lam_safe

        return decay, forcing

    def generate_stochastic_term(self, shape, dt, dtype, h_state=None):
        if self.sde_mode != "sde":
            return torch.zeros(shape, dtype=dtype, device=self.lam_re.device)

        device = self.lam_re.device
        dt_tensor = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=device)

        if dt_tensor.is_complex():
            dt_tensor = dt_tensor.real

        base_scale = self.base_noise.abs() * torch.sqrt(dt_tensor.abs().float() + 1e-8)

        noise_re = torch.randn(shape, device=device, dtype=torch.float32)
        noise_im = torch.randn(shape, device=device, dtype=torch.float32)

        if h_state is not None and self.uncertainty_net is not None:
            h_flat = h_state.reshape(-1, self.dim)
            if h_flat.is_complex():
                h_real = h_flat.abs()
            else:
                h_real = h_flat.abs()
            h_mag = h_real.mean(dim=-1, keepdim=True)
            uncertainty = self.uncertainty_net(h_mag.expand(-1, self.dim))
            uncertainty = uncertainty.reshape(shape)
            scale = base_scale * (1 + uncertainty)
        else:
            scale = base_scale

        while scale.ndim < noise_re.ndim:
            scale = scale.unsqueeze(0)

        if dtype.is_complex:
            return torch.complex(noise_re * scale, noise_im * scale)
        return noise_re * scale


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

        self.conv_e0 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_e1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_e2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_e12 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_channels))

        lat = torch.linspace(-math.pi / 2, math.pi / 2, img_height)
        self.register_buffer("cos_lat", torch.cos(lat).view(1, 1, -1, 1))

        self.metric_scale = nn.Parameter(torch.tensor(1.0))
        self.viscosity_scale = nn.Parameter(torch.tensor(0.1))
        self.dispersion_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        B, C, H, W = x.shape

        y_e0 = self.conv_e0(x)
        y_e1 = self.conv_e1(x)
        y_e2 = self.conv_e2(x)
        y_e12 = self.conv_e12(x)

        cos_lat = self.cos_lat
        if H != self.cos_lat.shape[2]:
            cos_lat = F.interpolate(cos_lat, size=(H, 1), mode="bilinear", align_corners=False)
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
            lat_emb = F.interpolate(lat_emb, size=(H, 1), mode="bilinear", align_corners=False)
            lon_emb = F.interpolate(lon_emb, size=(1, W), mode="bilinear", align_corners=False)

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

        geo_feat = torch.cat([cos_lat, sin_lat, cos_lon, sin_lon], dim=1)
        geo_feat = geo_feat.expand(B, -1, -1, -1)
        geo_emb = self.geo_proj(geo_feat)

        return x + pos_emb + geo_emb
    