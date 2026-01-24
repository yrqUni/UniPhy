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
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


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
            nn.Conv2d(dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, num_experts, 1),
        )
        nn.init.xavier_uniform_(self.experts_down_re.weight)
        nn.init.zeros_(self.experts_down_im.weight)

    def forward(self, z):
        re, im = z.real, z.imag
        B, C, H, W = re.shape
        re = self.spatial_re(re) - self.spatial_im(im)
        im = self.spatial_re(im) + self.spatial_im(re)
        route_weights = F.softmax(self.router(re + im), dim=1)
        up_re = self.experts_up_re(re) - self.experts_up_im(im)
        up_im = self.experts_up_re(im) + self.experts_up_im(re)
        up_re = F.silu(up_re)
        up_im = F.silu(up_im)
        down_re = self.experts_down_re(up_re) - self.experts_down_im(up_im)
        down_im = self.experts_down_re(up_im) + self.experts_down_im(up_re)
        down_re = down_re.reshape(B, self.num_experts, self.dim, H, W)
        down_im = down_im.reshape(B, self.num_experts, self.dim, H, W)
        w = route_weights.unsqueeze(2)
        out_re = (down_re * w).sum(dim=1)
        out_im = (down_im * w).sum(dim=1)
        return torch.complex(out_re, out_im)


class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand, num_experts):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expand)
        self.ffn = ComplexDynamicFFN(dim, self.hidden_dim, num_experts=num_experts)
        self.centering_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        delta = self.ffn(x)
        delta_mean = delta.mean(dim=(-2, -1), keepdim=True)
        delta = delta - (delta_mean * self.centering_scale)
        return delta
    