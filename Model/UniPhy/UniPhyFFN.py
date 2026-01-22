import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1_re = nn.Conv2d(dim, hidden_dim, 1)
        self.w1_im = nn.Conv2d(dim, hidden_dim, 1)
        self.w2_re = nn.Conv2d(hidden_dim, dim, 1)
        self.w2_im = nn.Conv2d(hidden_dim, dim, 1)
        nn.init.zeros_(self.w2_re.weight)
        nn.init.zeros_(self.w2_im.weight)

    def forward(self, z):
        dtype = self.w1_re.weight.dtype
        re, im = z.real.to(dtype), z.imag.to(dtype)
        h_re = self.w1_re(re) - self.w1_im(im)
        h_im = self.w1_re(im) + self.w1_im(re)
        h_re, h_im = F.silu(h_re), F.silu(h_im)
        out_re = self.w2_re(h_re) - self.w2_im(h_im)
        out_im = self.w2_re(h_im) + self.w2_im(re)
        return torch.complex(out_re, out_im)

class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * expand
        self.ffn = ComplexFFN(dim, self.hidden_dim)

    def forward(self, x):
        delta = self.ffn(x)
        delta_mean = delta.mean(dim=(-2, -1), keepdim=True)
        delta = delta - delta_mean
        return delta
    