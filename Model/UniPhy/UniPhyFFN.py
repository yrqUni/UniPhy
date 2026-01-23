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
    def __init__(self, dim, hidden_dim, num_experts=4):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.spatial_re = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.spatial_im = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.experts_re = nn.Conv2d(dim, hidden_dim * num_experts, 1, groups=1)
        self.experts_im = nn.Conv2d(dim, hidden_dim * num_experts, 1, groups=1)
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            LayerNorm2d(dim // 4) if dim // 4 > 0 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_experts, 1) 
        )
        self.out_re = nn.Conv2d(hidden_dim, dim, 1)
        self.out_im = nn.Conv2d(hidden_dim, dim, 1)
        nn.init.zeros_(self.out_re.weight)
        nn.init.zeros_(self.out_im.weight)

    def forward(self, z):
        re, im = z.real, z.imag
        B, C, H, W = re.shape
        re = self.spatial_re(re) - self.spatial_im(im)
        im = self.spatial_re(im) + self.spatial_im(re)
        feat_combined = re + im 
        route_weights = F.softmax(self.router(feat_combined), dim=1)
        all_experts_re = self.experts_re(re) - self.experts_im(im)
        all_experts_im = self.experts_re(im) + self.experts_im(re)
        all_experts_re = all_experts_re.view(B, self.num_experts, self.hidden_dim, H, W)
        all_experts_im = all_experts_im.view(B, self.num_experts, self.hidden_dim, H, W)
        w = route_weights.unsqueeze(2)
        h_re = (all_experts_re * w).sum(dim=1)
        h_im = (all_experts_im * w).sum(dim=1)
        h_re = F.silu(h_re)
        h_im = F.silu(h_im)
        out_re = self.out_re(h_re) - self.out_im(h_im)
        out_i = self.out_re(h_im) + self.out_im(h_re)
        return torch.complex(out_re, out_i)
    
class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand, num_experts=4):
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
    