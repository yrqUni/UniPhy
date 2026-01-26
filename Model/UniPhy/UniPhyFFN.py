import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        out = self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)
        return out


class ComplexLayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm_re = LayerNorm2d(dim, eps)
        self.norm_im = LayerNorm2d(dim, eps)

    def forward(self, x):
        if not x.is_complex():
            return self.norm_re(x)
        re_norm = self.norm_re(x.real)
        im_norm = self.norm_im(x.imag)
        return torch.complex(re_norm, im_norm)


class ComplexDynamicFFN(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.spatial_re = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.spatial_im = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.experts_up_re = nn.Conv2d(dim, hidden_dim * num_experts, 1, groups=1)
        self.experts_up_im = nn.Conv2d(dim, hidden_dim * num_experts, 1, groups=1)

        self.experts_down_re = nn.Conv2d(
            hidden_dim * num_experts, dim * num_experts, 1, groups=num_experts
        )
        self.experts_down_im = nn.Conv2d(
            hidden_dim * num_experts, dim * num_experts, 1, groups=num_experts
        )

        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, num_experts, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.experts_up_re.weight)
        nn.init.xavier_uniform_(self.experts_up_im.weight)
        nn.init.xavier_uniform_(self.experts_down_re.weight)
        nn.init.zeros_(self.experts_down_im.weight)
        nn.init.zeros_(self.experts_up_re.bias)
        nn.init.zeros_(self.experts_up_im.bias)
        nn.init.zeros_(self.experts_down_re.bias)
        nn.init.zeros_(self.experts_down_im.bias)

    def forward(self, z):
        if z.is_complex():
            re = z.real
            im = z.imag
        else:
            re = z
            im = torch.zeros_like(z)

        B, C, H, W = re.shape

        re_spatial = self.spatial_re(re) - self.spatial_im(im)
        im_spatial = self.spatial_re(im) + self.spatial_im(re)

        router_input = torch.cat([re_spatial, im_spatial], dim=1)
        route_weights = F.softmax(self.router(router_input), dim=1)
        route_weights = route_weights.view(B, self.num_experts, 1, H, W)

        up_re = self.experts_up_re(re_spatial) - self.experts_up_im(im_spatial)
        up_im = self.experts_up_re(im_spatial) + self.experts_up_im(re_spatial)

        up_re = F.silu(up_re)
        up_im = F.silu(up_im)

        down_re = self.experts_down_re(up_re) - self.experts_down_im(up_im)
        down_im = self.experts_down_re(up_im) + self.experts_down_im(up_re)

        down_re = down_re.view(B, self.num_experts, self.dim, H, W)
        down_im = down_im.view(B, self.num_experts, self.dim, H, W)

        out_re = (down_re * route_weights).sum(dim=1)
        out_im = (down_im * route_weights).sum(dim=1)

        return torch.complex(out_re, out_im)


class GatedComplexFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.fc1_re = nn.Linear(dim, hidden_dim * 2)
        self.fc1_im = nn.Linear(dim, hidden_dim * 2)
        self.fc2_re = nn.Linear(hidden_dim, dim)
        self.fc2_im = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1_re.weight)
        nn.init.xavier_uniform_(self.fc1_im.weight)
        nn.init.xavier_uniform_(self.fc2_re.weight)
        nn.init.zeros_(self.fc2_im.weight)

    def forward(self, x):
        if x.is_complex():
            re = x.real
            im = x.imag
        else:
            re = x
            im = torch.zeros_like(x)

        h1_re = self.fc1_re(re) - self.fc1_im(im)
        h1_im = self.fc1_re(im) + self.fc1_im(re)

        h1_re_gate, h1_re_val = h1_re.chunk(2, dim=-1)
        h1_im_gate, h1_im_val = h1_im.chunk(2, dim=-1)

        gate_re = torch.sigmoid(h1_re_gate)
        gate_im = torch.sigmoid(h1_im_gate)

        h_re = gate_re * F.silu(h1_re_val)
        h_im = gate_im * F.silu(h1_im_val)

        h_re = self.dropout(h_re)
        h_im = self.dropout(h_im)

        out_re = self.fc2_re(h_re) - self.fc2_im(h_im)
        out_im = self.fc2_re(h_im) + self.fc2_im(h_re)

        return torch.complex(out_re, out_im)


class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand, num_experts):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expand)
        self.num_experts = num_experts

        self.pre_norm = ComplexLayerNorm2d(dim)
        self.ffn = ComplexDynamicFFN(dim, self.hidden_dim, num_experts=num_experts)
        self.post_norm = ComplexLayerNorm2d(dim)

        self.centering_scale = nn.Parameter(torch.tensor(0.5))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_norm = self.pre_norm(x)
        delta = self.ffn(x_norm)

        delta_mean = torch.complex(
            delta.real.mean(dim=(-2, -1), keepdim=True),
            delta.imag.mean(dim=(-2, -1), keepdim=True)
        )
        delta_centered = delta - delta_mean * self.centering_scale

        delta_out = self.post_norm(delta_centered)
        delta_out = delta_out * self.output_scale

        return delta_out


class SpectralGatingUnit(nn.Module):
    def __init__(self, dim, h_dim, w_dim):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.complex_weight = nn.Parameter(
            torch.randn(dim, h_dim, w_dim // 2 + 1, 2) * 0.02
        )
        self.gate_proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        if x.is_complex():
            re = x.real
            im = x.imag
        else:
            re = x
            im = torch.zeros_like(x)

        B, C, H, W = re.shape

        x_fft = torch.fft.rfft2(re, norm="ortho")

        weight = torch.view_as_complex(self.complex_weight)
        if x_fft.shape[-2:] != weight.shape[-2:]:
            weight = F.interpolate(
                weight.unsqueeze(0).real,
                size=x_fft.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            weight = torch.complex(weight, torch.zeros_like(weight))

        x_weighted = x_fft * weight.unsqueeze(0)
        x_spatial = torch.fft.irfft2(x_weighted, s=(H, W), norm="ortho")

        gate_input = torch.cat([re, x_spatial], dim=1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        out_re = re * gate + x_spatial * (1 - gate)
        out_im = im * gate

        return torch.complex(out_re, out_im)


class MultiScaleFFN(nn.Module):
    def __init__(self, dim, expand, num_scales=3):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expand)
        self.num_scales = num_scales

        self.scale_convs_re = nn.ModuleList([
            nn.Conv2d(dim, self.hidden_dim // num_scales, kernel_size=2 * i + 1, padding=i)
            for i in range(num_scales)
        ])
        self.scale_convs_im = nn.ModuleList([
            nn.Conv2d(dim, self.hidden_dim // num_scales, kernel_size=2 * i + 1, padding=i)
            for i in range(num_scales)
        ])

        self.proj_re = nn.Conv2d(self.hidden_dim, dim, 1)
        self.proj_im = nn.Conv2d(self.hidden_dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, x):
        if x.is_complex():
            re = x.real
            im = x.imag
        else:
            re = x
            im = torch.zeros_like(x)

        scale_outputs_re = []
        scale_outputs_im = []

        weights = F.softmax(self.scale_weights, dim=0)

        for i in range(self.num_scales):
            h_re = self.scale_convs_re[i](re) - self.scale_convs_im[i](im)
            h_im = self.scale_convs_re[i](im) + self.scale_convs_im[i](re)
            scale_outputs_re.append(F.silu(h_re) * weights[i])
            scale_outputs_im.append(F.silu(h_im) * weights[i])

        h_re = torch.cat(scale_outputs_re, dim=1)
        h_im = torch.cat(scale_outputs_im, dim=1)

        out_re = self.proj_re(h_re) - self.proj_im(h_im)
        out_im = self.proj_re(h_im) + self.proj_im(h_re)

        return torch.complex(out_re, out_im)
    