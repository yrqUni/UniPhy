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


class ComplexConvFFN(nn.Module):
    def __init__(self, dim, expand):
        super().__init__()
        hidden_dim = int(dim * expand)

        self.fc1_re = nn.Conv2d(dim, hidden_dim, 1)
        self.fc1_im = nn.Conv2d(dim, hidden_dim, 1)

        self.dw_conv_re = nn.Conv2d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=3, 
            padding=1, 
            groups=hidden_dim, 
            bias=False
        )
        self.dw_conv_im = nn.Conv2d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=3, 
            padding=1, 
            groups=hidden_dim, 
            bias=False
        )

        self.fc2_re = nn.Conv2d(hidden_dim, dim, 1)
        self.fc2_im = nn.Conv2d(hidden_dim, dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1_re.weight)
        nn.init.xavier_uniform_(self.fc1_im.weight)
        nn.init.xavier_uniform_(self.fc2_re.weight)
        nn.init.zeros_(self.fc2_im.weight)
        
        if self.fc1_re.bias is not None: nn.init.zeros_(self.fc1_re.bias)
        if self.fc1_im.bias is not None: nn.init.zeros_(self.fc1_im.bias)
        if self.fc2_re.bias is not None: nn.init.zeros_(self.fc2_re.bias)
        if self.fc2_im.bias is not None: nn.init.zeros_(self.fc2_im.bias)

    def forward(self, x):
        if x.is_complex():
            re = x.real
            im = x.imag
        else:
            re = x
            im = torch.zeros_like(x)

        h_re = self.fc1_re(re) - self.fc1_im(im)
        h_im = self.fc1_re(im) + self.fc1_im(re)

        h_re = self.dw_conv_re(h_re)
        h_im = self.dw_conv_im(h_im)

        h_re = F.gelu(h_re)
        h_im = F.gelu(h_im)

        out_re = self.fc2_re(h_re) - self.fc2_im(h_im)
        out_im = self.fc2_re(h_im) + self.fc2_im(h_re)

        return torch.complex(out_re, out_im)


class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand):
        super().__init__()
        self.dim = dim
        self.pre_norm = ComplexLayerNorm2d(dim)
        self.ffn = ComplexConvFFN(dim, expand)
        self.post_norm = ComplexLayerNorm2d(dim)
        self.centering_scale = nn.Parameter(torch.tensor(0.5))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, D, H, W = x.shape
            x = x.reshape(B * T, D, H, W)

        x_norm = self.pre_norm(x)
        delta = self.ffn(x_norm)
        
        delta_mean = torch.complex(
            delta.real.mean(dim=(-2, -1), keepdim=True),
            delta.imag.mean(dim=(-2, -1), keepdim=True)
        )
        delta_centered = delta - delta_mean * self.centering_scale
        
        delta_out = self.post_norm(delta_centered)
        delta_out = delta_out * self.output_scale

        if is_5d:
            delta_out = delta_out.reshape(B, T, D, H, W)
            
        return delta_out

