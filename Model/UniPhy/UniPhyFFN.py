import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)


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
        self.fc1_re = nn.Linear(dim, hidden_dim)
        self.fc1_im = nn.Linear(dim, hidden_dim)
        self.dw_conv_re = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            groups=hidden_dim, bias=False
        )
        self.dw_conv_im = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            groups=hidden_dim, bias=False
        )
        self.fc2_re = nn.Linear(hidden_dim, dim)
        self.fc2_im = nn.Linear(hidden_dim, dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1_re.weight)
        nn.init.xavier_uniform_(self.fc1_im.weight)
        nn.init.xavier_uniform_(self.fc2_re.weight)
        nn.init.zeros_(self.fc2_im.weight)
        nn.init.zeros_(self.fc1_re.bias)
        nn.init.zeros_(self.fc1_im.bias)
        nn.init.zeros_(self.fc2_re.bias)
        nn.init.zeros_(self.fc2_im.bias)

    def _apply_fc(self, x, fc_re, fc_im):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        res_re = fc_re(x_flat)
        res_im = fc_im(x_flat)
        out_c = fc_re.out_features
        res_re = res_re.view(b, h, w, out_c).permute(0, 3, 1, 2).contiguous()
        res_im = res_im.view(b, h, w, out_c).permute(0, 3, 1, 2).contiguous()
        return res_re, res_im

    def forward(self, x):
        if x.is_complex():
            re, im = x.real.contiguous(), x.imag.contiguous()
        else:
            re, im = x.contiguous(), torch.zeros_like(x)
        f1_re_re, f1_re_im = self._apply_fc(re, self.fc1_re, self.fc1_im)
        f1_im_re, f1_im_im = self._apply_fc(im, self.fc1_re, self.fc1_im)
        h_re = f1_re_re - f1_im_im
        h_im = f1_re_im + f1_im_re
        h_re = F.gelu(self.dw_conv_re(h_re))
        h_im = F.gelu(self.dw_conv_im(h_im))
        f2_re_re, f2_re_im = self._apply_fc(h_re, self.fc2_re, self.fc2_im)
        f2_im_re, f2_im_im = self._apply_fc(h_im, self.fc2_re, self.fc2_im)
        out_re = f2_re_re - f2_im_im
        out_im = f2_re_im + f2_im_re
        return torch.complex(out_re.contiguous(), out_im.contiguous())


class UniPhyFeedForwardNetwork(nn.Module):
    def __init__(self, dim, expand):
        super().__init__()
        self.pre_norm = ComplexLayerNorm2d(dim)
        self.ffn = ComplexConvFFN(dim, expand)
        self.post_norm = ComplexLayerNorm2d(dim)
        self.centering_scale = nn.Parameter(torch.tensor(0.5))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_norm = self.pre_norm(x)
        delta = self.ffn(x_norm)
        delta_mean = torch.complex(
            delta.real.mean(dim=(-2, -1), keepdim=True),
            delta.imag.mean(dim=(-2, -1), keepdim=True),
        )
        delta_centered = delta - delta_mean * self.centering_scale
        delta_out = self.post_norm(delta_centered) * self.output_scale
        return delta_out
    