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
        self.experts_down_re = nn.Conv2d(hidden_dim * num_experts, dim * num_experts, 1, groups=num_experts)
        self.experts_down_im = nn.Conv2d(hidden_dim * num_experts, dim * num_experts, 1, groups=num_experts)
        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, num_experts, 1),
        )
        self.aux_loss = 0.0
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
        logits = self.router(router_input)
        route_weights = F.softmax(logits, dim=1)
        prob_mean = route_weights.mean(dim=(0, 2, 3))
        self.aux_loss = (self.num_experts * (prob_mean ** 2).sum()) * 0.1
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
    