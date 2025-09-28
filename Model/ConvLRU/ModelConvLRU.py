import math
import numpy as np
import torch
import torch.nn as nn
from pscan import PScan, pscan_check

def _kaiming_like_(tensor):
    nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu')
    return tensor

def icnr_conv3d_weight_(weight, rH: int, rW: int):
    out_ch, in_ch, kD, kH, kW = weight.shape
    base_out = out_ch // (rH * rW)
    base = weight.new_zeros((base_out, in_ch, 1, kH, kW))
    _kaiming_like_(base)
    base = base.repeat_interleave(rH * rW, dim=0)
    with torch.no_grad():
        weight.copy_(base)
    return weight

def _bilinear_kernel_2d(kH: int, kW: int, device, dtype):
    factor_h = (kH + 1) // 2
    factor_w = (kW + 1) // 2
    center_h = factor_h - 1 if kH % 2 == 1 else factor_h - 0.5
    center_w = factor_w - 1 if kW % 2 == 1 else factor_w - 0.5
    og_h = torch.arange(kH, device=device, dtype=dtype)
    og_w = torch.arange(kW, device=device, dtype=dtype)
    fh = (1 - torch.abs(og_h - center_h) / factor_h).unsqueeze(1)
    fw = (1 - torch.abs(og_w - center_w) / factor_w).unsqueeze(0)
    return fh @ fw

def deconv3d_bilinear_init_(weight):
    in_ch, out_ch, kD, kH, kW = weight.shape
    with torch.no_grad():
        weight.zero_()
        kernel = _bilinear_kernel_2d(kH, kW, weight.device, weight.dtype)
        c = min(in_ch, out_ch)
        for i in range(c):
            weight[i, i, 0, :, :] = kernel
    return weight

class SpectralPrior2D(nn.Module):
    def __init__(self, channels: int, S: int, W: int, rank: int = 8, gain_init: float = 0.0, mode: str = "linear"):
        super().__init__()
        self.C = channels
        self.S = S
        self.W = W
        self.R = rank
        self.mode = str(mode).lower()
        self.A = nn.Parameter(torch.zeros(self.C, self.R, self.S))
        self.B = nn.Parameter(torch.zeros(self.C, self.R, self.W))
        self.gain = nn.Parameter(torch.full((self.C,), float(gain_init)))
        nn.init.normal_(self.A, std=1e-3)
        nn.init.normal_(self.B, std=1e-3)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, C, S, W = h.shape
        F = torch.einsum('cri,crj->cij', self.A, self.B)
        if self.mode == "exp":
            G = torch.exp(self.gain.view(C, 1, 1) * F)
        else:
            G = 1.0 + self.gain.view(C, 1, 1) * F
        return h * G.view(1, 1, C, S, W)

class SphericalHarmonicsPrior(nn.Module):
    def __init__(self, channels: int, H: int, W: int, Lmax: int = 6, rank: int = 8, gain_init: float = 0.0):
        super().__init__()
        self.C = channels
        self.H = H
        self.W = W
        self.Lmax = Lmax
        self.R = rank
        self.K = Lmax * Lmax
        self.W1 = nn.Parameter(torch.zeros(self.C, self.R))
        self.W2 = nn.Parameter(torch.zeros(self.R, self.K))
        self.gain = nn.Parameter(torch.full((self.C,), float(gain_init)))
        nn.init.normal_(self.W1, std=1e-3)
        nn.init.normal_(self.W2, std=1e-3)
        theta, phi = self._latlon_to_spherical(H, W, device=None)
        Y = self._real_sph_harm_basis(theta, phi, Lmax)
        self.register_buffer('Y_real', Y)
    @staticmethod
    def _latlon_to_spherical(H, W, device):
        lat = torch.linspace(math.pi/2, -math.pi/2, steps=H, device=device)
        lon = torch.linspace(-math.pi, math.pi, steps=W, device=device)
        theta = (math.pi/2 - lat).unsqueeze(1).repeat(1, W)
        phi = lon.unsqueeze(0).repeat(H, 1)
        return theta, phi
    @staticmethod
    def _real_sph_harm_basis(theta: torch.Tensor, phi: torch.Tensor, Lmax: int) -> torch.Tensor:
        H, W = theta.shape
        device = theta.device
        dtype = theta.dtype
        x = torch.cos(theta)
        one = torch.ones_like(x)
        pi = torch.tensor(math.pi, device=device, dtype=dtype)
        P = [[None]*(l+1) for l in range(Lmax)]
        P[0][0] = one
        if Lmax >= 2:
            P[1][0] = x
        for l in range(2, Lmax):
            l_f = torch.tensor(l, device=device, dtype=dtype)
            P[l][0] = ((2*l_f-1)*x*P[l-1][0] - (l_f-1)*P[l-2][0]) / l_f
        for m in range(1, Lmax):
            m_f = torch.tensor(m, device=device, dtype=dtype)
            P_mm = ((-1)**m) * SphericalHarmonicsPrior._double_factorial(2*m-1, dtype, device) * (1 - x*x).pow(m_f/2)
            P[m][m] = P_mm
            if m+1 < Lmax:
                P[m+1][m] = (2*m_f+1)*x*P_mm
            for l in range(m+2, Lmax):
                l_f = torch.tensor(l, device=device, dtype=dtype)
                P[l][m] = ((2*l_f-1)*x*P[l-1][m] - (l_f+m_f-1)*P[l-2][m]) / (l_f-m_f)
        idx = torch.arange(0, Lmax, device=device, dtype=dtype).view(-1, 1, 1)
        cos_mphi = torch.cos(idx * phi)
        sin_mphi = torch.sin(idx * phi)
        Ys = []
        for l in range(Lmax):
            l_f = torch.tensor(l, device=device, dtype=dtype)
            for m in range(-l, l+1):
                m_abs = abs(m)
                N_lm = torch.sqrt((2*l_f+1)/(4*pi) * SphericalHarmonicsPrior._fact_ratio(l, m_abs, dtype, device))
                if m == 0:
                    Y = N_lm * P[l][0]
                elif m > 0:
                    Y = math.sqrt(2.0) * N_lm * P[l][m_abs] * cos_mphi[m_abs]
                else:
                    Y = math.sqrt(2.0) * N_lm * P[l][m_abs] * sin_mphi[m_abs]
                Ys.append(Y)
        Y_stack = torch.stack(Ys, dim=0)
        return Y_stack
    @staticmethod
    def _fact_ratio(l: int, m_abs: int, dtype, device):
        l = torch.tensor(l, dtype=dtype, device=device)
        m = torch.tensor(m_abs, dtype=dtype, device=device)
        return torch.exp(torch.lgamma(l - m + 1) - torch.lgamma(l + m + 1))
    @staticmethod
    def _double_factorial(n: int, dtype, device):
        if n < 1:
            return torch.tensor(1.0, dtype=dtype, device=device)
        seq = torch.arange(n, 0, -2, dtype=dtype, device=device)
        return torch.prod(seq)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        Y = self.Y_real
        if (Y.dtype != x.dtype) or (Y.device != x.device):
            Y = Y.to(dtype=x.dtype, device=x.device)
        coeff = torch.matmul(self.W1, self.W2)
        bias = torch.einsum('ck,khw->chw', coeff, Y)
        bias = (self.gain.view(C, 1, 1) * bias).view(1, 1, C, H, W)
        return x + bias

class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
    def forward(self, x):
        BL, C, H, W = x.shape
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)
        max_pool, _ = torch.max(x.reshape(BL, C, -1), dim=-1)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(BL, C, 1, 1)
        return x * attn

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn

class CBAM2DPerStep(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(spatial_kernel)
    def forward(self, x):
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        x = x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return x

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        output_activation = getattr(args, "output_activation", "Identity")
        self.out_activation = getattr(nn, output_activation)()
        self.truncated_normal_init()
    def _check_pscan(self):
        assert all(pscan_check())
    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        skip_contains = ['layer_norm','params_log','freq_prior.','sh_prior.','post_ifft_conv_real','post_ifft_conv_imag','post_ifft_proj']
        skip_suffix = ('.U_row', '.V_col','upsp.weight', 'upsp.bias','pre_shuffle_conv.weight', 'pre_shuffle_conv.bias')
        def should_skip(name: str) -> bool:
            if any(tok in name for tok in skip_contains): return True
            if name.endswith(skip_suffix): return True
            return False
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            for n, p in self.named_parameters():
                if should_skip(n): continue
                if n.endswith('.bias'):
                    if torch.is_complex(p):
                        p.copy_(torch.zeros_like(p))
                    else:
                        p.zero_()
                    continue
                if torch.is_complex(p):
                    p.real.uniform_(2*l-1, 2*u-1); p.imag.uniform_(2*l-1, 2*u-1)
                    p.real.erfinv_(); p.imag.erfinv_()
                    p.real.mul_(std*math.sqrt(2.)); p.imag.mul_(std*math.sqrt(2.))
                    p.real.add_(mean); p.imag.add_(mean)
                else:
                    p.uniform_(2*l-1, 2*u-1); p.erfinv_(); p.mul_(std*math.sqrt(2.)); p.add_(mean)
    @torch.no_grad()
    def _default_dt_like(self, x, listT, fill_last=False, out_gen_num=None):
        if not fill_last:
            if listT is None:
                return torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
            return listT
        K = out_gen_num if out_gen_num is not None else 1
        if listT is not None:
            last = listT[:, -1].unsqueeze(1)
            return last.repeat(1, K)
        return torch.ones(x.size(0), K, device=x.device, dtype=x.dtype)
    def forward(self, x, mode, out_gen_num=None, listT=None, listT_future=None):
        assert mode in ['p', 'i']
        if mode == 'p':
            listT_eff = self._default_dt_like(x, listT, fill_last=False)
            x = self.embedding(x)
            x, _ = self.convlru_model(x, last_hidden_ins=None, listT=listT_eff)
            x = self.decoder(x)
            x = self.out_activation(x)
            return x
        assert getattr(self.args, "input_ch", 1) == getattr(self.args, "out_ch", 1)
        assert out_gen_num is not None and out_gen_num >= 1
        out = []
        listT_eff = self._default_dt_like(x, listT, fill_last=False)
        x_emb = self.embedding(x)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, last_hidden_ins=None, listT=listT_eff)
        x_dec = self.decoder(x_hidden)
        x_step = self.out_activation(x_dec[:, -1:])
        out.append(x_step)
        if listT_future is None:
            listT_future = self._default_dt_like(x, listT, fill_last=True, out_gen_num=out_gen_num)
        for t in range(out_gen_num - 1):
            dt_step = listT_future[:, t:t+1]
            x_in = self.embedding(x_step)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt_step)
            x_dec = self.decoder(x_hidden)
            x_step = self.out_activation(x_dec[:, -1:])
            out.append(x_step)
        return torch.concat(out, dim=1)

class Conv_hidden(nn.Module):
    def __init__(self, ch, hidden_size, activation_func, use_cbam=False):
        super().__init__()
        self.ch = ch
        self.use_cbam = bool(use_cbam)
        self.conv3 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 3, 3), padding='same')
        self.activation3 = getattr(nn, activation_func)()
        self.conv1 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding='same')
        self.activation1 = getattr(nn, activation_func)()
        self.layer_norm_conv = nn.LayerNorm([*hidden_size])
        if self.use_cbam:
            self.cbam = CBAM2DPerStep(self.ch, reduction=16, spatial_kernel=7)
            self.layer_norm_attn = nn.LayerNorm([*hidden_size])
            self.gate_conv = nn.Sequential(nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding='same'), nn.Sigmoid())
        else:
            self.cbam = None
            self.layer_norm_attn = None
            self.gate_conv = None
    def forward(self, x):
        x_update = self.conv3(x); x_update = self.activation3(x_update)
        x_update = self.conv1(x_update); x_update = self.activation1(x_update)
        if self.use_cbam:
            x_update = self.layer_norm_attn(x_update.permute(0,2,1,3,4)).permute(0,2,1,3,4)
            x_update = x_update + x
            x_update = self.cbam(x_update)
            x_update = self.layer_norm_conv(x_update.permute(0,2,1,3,4)).permute(0,2,1,3,4)
            gate = self.gate_conv(x_update)
            x = (1 - gate) * x + gate * x_update
        else:
            x_update = self.layer_norm_conv(x_update.permute(0,2,1,3,4)).permute(0,2,1,3,4)
            x = x_update + x
        return x

def pixel_unshuffle_hw_3d(x, rH: int, rW: int):
    N, C, D, H, W = x.shape
    x = x.view(N, C, D, H // rH, rH, W // rW, rW)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    x = x.view(N, C * rH * rW, D, H // rH, W // rW)
    return x

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = getattr(args, "input_ch", 1)
        self.input_size = getattr(args, "input_size", (64, 64))
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.emb_hidden_ch = getattr(args, "emb_hidden_ch", 32)
        self.emb_hidden_layers_num = getattr(args, "emb_hidden_layers_num", 0)
        self.down_strategy = getattr(args, "emb_strategy", "pxus")
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        if self.down_strategy == "conv":
            self.downsp = nn.Conv3d(self.input_ch, self.input_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            with torch.no_grad():
                _, C, _, H, W = self.downsp(torch.zeros(1, self.input_ch, 1, *self.input_size)).size()
                self.input_downsp_shape = (C, H, W)
            in_ch_after_down = self.input_downsp_shape[0]
        else:
            Hd, Wd = self.input_size[0] // self.rH, self.input_size[1] // self.rW
            Cd = self.input_ch * self.rH * self.rW
            self.input_downsp_shape = (Cd, Hd, Wd)
            in_ch_after_down = Cd
        self.hidden_size = (self.input_downsp_shape[1], self.input_downsp_shape[2])
        if self.emb_hidden_layers_num == 0:
            self.c_in = nn.Conv3d(in_ch_after_down, self.emb_ch, kernel_size=(1, 7, 7), padding='same')
            self.c_hidden = None
            self.c_out = None
        else:
            self.c_in = nn.Conv3d(in_ch_after_down, self.emb_hidden_ch, kernel_size=(1, 7, 7), padding='same')
            self.c_hidden = nn.ModuleList([Conv_hidden(self.emb_hidden_ch, self.hidden_size, getattr(args, "hidden_activation", "ReLU"), use_cbam=False) for _ in range(self.emb_hidden_layers_num)])
            self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, getattr(args, "hidden_activation", "ReLU"))()
        self.layer_norm = nn.LayerNorm([self.emb_ch, *self.hidden_size])
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        if self.down_strategy == "conv":
            x = self.downsp(x)
        else:
            x = pixel_unshuffle_hw_3d(x, self.rH, self.rW)
        x = self.c_in(x); x = self.activation(x)
        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x)
            x = self.c_out(x)
        x = self.layer_norm(x.permute(0, 2, 1, 3, 4))
        return x

def pixel_shuffle_hw_3d(x, rH: int, rW: int):
    N, C_mul, D, H, W = x.shape
    C = C_mul // (rH * rW)
    x = x.view(N, C, rH, rW, D, H, W)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x = x.view(N, C, D, H * rH, W * rW)
    return x

class Decoder(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.output_ch = getattr(args, "out_ch", 1)
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.dec_hidden_ch = getattr(args, "dec_hidden_ch", 0)
        self.dec_hidden_layers_num = getattr(args, "dec_hidden_layers_num", 0)
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.dec_strategy = getattr(args, "dec_strategy", "pxsf")
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        out_ch_after_up = self.dec_hidden_ch if self.dec_hidden_layers_num != 0 else self.emb_ch
        if self.dec_strategy == "deconv":
            self.upsp = nn.ConvTranspose3d(in_channels=self.emb_ch, out_channels=out_ch_after_up, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            deconv3d_bilinear_init_(self.upsp.weight)
            with torch.no_grad():
                if self.upsp.bias is not None:
                    self.upsp.bias.zero_()
        else:
            self.pre_shuffle_conv = nn.Conv3d(in_channels=self.emb_ch, out_channels=out_ch_after_up * self.rH * self.rW, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.rH, self.rW)
            with torch.no_grad():
                if self.pre_shuffle_conv.bias is not None:
                    self.pre_shuffle_conv.bias.zero_()
        if self.dec_hidden_layers_num != 0:
            H = self.hidden_size[0] * self.rH if self.dec_strategy == "pxsf" else None
            W = self.hidden_size[1] * self.rW if self.dec_strategy == "pxsf" else None
            self.c_hidden = nn.ModuleList([Conv_hidden(out_ch_after_up, (H if H else self.hidden_size[0], W if W else self.hidden_size[1]), getattr(args, "hidden_activation", "ReLU"), use_cbam=False) for _ in range(self.dec_hidden_layers_num)])
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding='same')
        else:
            self.c_hidden = None
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, getattr(args, "hidden_activation", "ReLU"))()
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        if self.dec_strategy == "deconv":
            x = self.upsp(x)
        else:
            x = self.pre_shuffle_conv(x)
            x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x)
            x = self.c_out(x)
        else:
            x = self.c_out(x)
        return x.permute(0, 2, 1, 3, 4)

class ConvLRUModel(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        layers = getattr(args, "convlru_num_blocks", 1)
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, input_downsp_shape) for _ in range(layers)])
    def forward(self, x, last_hidden_ins=None, listT=None):
        last_hidden_outs = []
        for idx, convlru_block in enumerate(self.convlru_blocks):
            h_in = last_hidden_ins[idx] if (last_hidden_ins is not None) else None
            x, last_hidden_out = convlru_block(x, h_in, listT=listT)
            last_hidden_outs.append(last_hidden_out)
        return x, last_hidden_outs

class ConvLRUBlock(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)
    def forward(self, x, last_hidden_in, listT=None):
        x, last_hidden_out = self.lru_layer(x, last_hidden_in, listT=listT)
        x = self.feed_forward(x)
        return x, last_hidden_out

class ConvLRULayer(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        S, W = self.hidden_size
        self.rank = int(getattr(args, "lru_rank", min(S, W, 32)))
        u1_s = torch.rand(self.emb_ch, S)
        u2_s = torch.rand(self.emb_ch, S)
        nu_s_log = torch.log(-0.5 * torch.log(u1_s * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_s_log = torch.log(u2_s * torch.tensor(np.pi) * 2)
        diag_lambda_s = torch.exp(torch.complex(-torch.exp(nu_s_log), torch.exp(theta_s_log)))
        gamma_s_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda_s) ** 2))
        self.params_log_square = nn.Parameter(torch.vstack((nu_s_log, theta_s_log, gamma_s_log)))
        u1_r = torch.rand(self.emb_ch, self.rank)
        u2_r = torch.rand(self.emb_ch, self.rank)
        nu_r_log = torch.log(-0.5 * torch.log(u1_r * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_r_log = torch.log(u2_r * torch.tensor(np.pi) * 2)
        diag_lambda_r = torch.exp(torch.complex(-torch.exp(nu_r_log), torch.exp(theta_r_log)))
        gamma_r_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda_r) ** 2))
        self.params_log_rank = nn.Parameter(torch.vstack((nu_r_log, theta_r_log, gamma_r_log)))
        self.U_row = nn.Parameter(torch.randn(self.emb_ch, S, self.rank, dtype=torch.cfloat) * (1.0 / math.sqrt(S)))
        self.V_col = nn.Parameter(torch.randn(self.emb_ch, W, self.rank, dtype=torch.cfloat) * (1.0 / math.sqrt(W)))
        C = self.emb_ch
        self.proj_W = nn.Parameter(torch.randn(C, C, dtype=torch.cfloat) / math.sqrt(C))
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat)) if self.use_bias else None
        self.post_ifft_conv_real = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=True)
        self.post_ifft_conv_imag = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=True)
        self.post_ifft_proj = nn.Conv3d(in_channels=self.emb_ch*2, out_channels=self.emb_ch, kernel_size=(1,1,1), padding='same', bias=True)
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
        self.gate_conv = nn.Sequential(nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1,1,1), padding='same'), nn.Sigmoid()) if bool(getattr(args, "use_gate", False)) else None
        self.use_freq_prior = bool(getattr(args, "use_freq_prior", False))
        self.use_sh_prior = bool(getattr(args, "use_sh_prior", False))
        self.freq_mode = getattr(args, "freq_mode", "linear").lower()
        if self.use_freq_prior:
            freq_rank = int(getattr(args, "freq_rank", 8))
            freq_gain_init = float(getattr(args, "freq_gain_init", 0.0))
            self.freq_prior = SpectralPrior2D(self.emb_ch, S, W, rank=freq_rank, gain_init=freq_gain_init, mode=self.freq_mode)
        else:
            self.freq_prior = None
        if self.use_sh_prior:
            Lmax = int(getattr(args, "sh_Lmax", 6))
            sh_rank = int(getattr(args, "sh_rank", 8))
            sh_gain_init = float(getattr(args, "sh_gain_init", 0.0))
            self.sh_prior = SphericalHarmonicsPrior(self.emb_ch, S, W, Lmax=Lmax, rank=sh_rank, gain_init=sh_gain_init)
        else:
            self.sh_prior = None
        self.lambda_type = str(getattr(args, "lambda_type", "static")).lower()
        self.exo_mode = str(getattr(args, "exo_mode", "mlp")).lower()
        if self.lambda_type not in ("static", "exogenous"):
            raise ValueError(f"lambda_type must be 'static' or 'exogenous', got {self.lambda_type}")
        if self.lambda_type == "exogenous":
            self.delta_scale_nu = nn.Parameter(torch.tensor(0.1))
            self.delta_scale_th = nn.Parameter(torch.tensor(0.1))
            self.mod_hidden = int(getattr(self.args, "lambda_mlp_hidden", 16))
            C = self.emb_ch
            S, W = self.hidden_size
            R = self.rank
            self.mod_nu_fc1_S = nn.Parameter(torch.empty(C, S, self.mod_hidden))
            self.mod_nu_fc2_S = nn.Parameter(torch.empty(C, self.mod_hidden, S))
            self.mod_th_fc1_S = nn.Parameter(torch.empty(C, S, self.mod_hidden))
            self.mod_th_fc2_S = nn.Parameter(torch.empty(C, self.mod_hidden, S))
            self.mod_nu_fc1_R = nn.Parameter(torch.empty(C, R, self.mod_hidden))
            self.mod_nu_fc2_R = nn.Parameter(torch.empty(C, self.mod_hidden, R))
            self.mod_th_fc1_R = nn.Parameter(torch.empty(C, R, self.mod_hidden))
            self.mod_th_fc2_R = nn.Parameter(torch.empty(C, self.mod_hidden, R))
            for p in [self.mod_nu_fc1_S, self.mod_nu_fc2_S, self.mod_th_fc1_S, self.mod_th_fc2_S, self.mod_nu_fc1_R, self.mod_nu_fc2_R, self.mod_th_fc1_R, self.mod_th_fc2_R]:
                nn.init.xavier_uniform_(p, gain=0.5)
            self.exo_affine_a = nn.Parameter(torch.zeros(C, 1, 1))
            self.exo_affine_b = nn.Parameter(torch.zeros(C, 1, 1))
        else:
            self.mod_nu_fc1_S = self.mod_nu_fc2_S = None
            self.mod_th_fc1_S = self.mod_th_fc2_S = None
            self.mod_nu_fc1_R = self.mod_nu_fc2_R = None
            self.mod_th_fc1_R = self.mod_th_fc2_R = None
            self.exo_affine_a = None
            self.exo_affine_b = None
        self.pscan = PScan.apply
    def _project_to_square(self, h):
        t = torch.einsum('blcsw,csr->blcrw', h, self.U_row.conj())
        z = torch.einsum('blcrw,cwp->blcrp', t, self.V_col)
        return z
    def _deproject_from_square(self, z):
        Vt = self.V_col.conj().transpose(1, 2)
        t = torch.einsum('blcrp,crw->blcrw', z, Vt)
        h = torch.einsum('blcrw,csr->blcsw', t, self.U_row)
        return h
    def _ifft_and_fuse(self, h_complex: torch.Tensor) -> torch.Tensor:
        h_spatial = torch.fft.ifft2(h_complex, dim=(-2, -1), norm='ortho')
        hr = h_spatial.real
        hi = h_spatial.imag
        hr = self.post_ifft_conv_real(hr.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        hi = self.post_ifft_conv_imag(hi.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        h_cat = torch.cat([hr, hi], dim=2)
        h_out = self.post_ifft_proj(h_cat.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        return h_out
    def _apply_static_dt_scaling(self, h, lam1, dt):
        lamk = lam1.pow(dt.view(dt.size(0), dt.size(1), 1, 1, 1))
        gamma1 = torch.sqrt(torch.clamp(1.0 - (lam1.abs() ** 2), min=1e-12))
        gammak = torch.sqrt(torch.clamp(1.0 - (lamk.abs() ** 2), min=1e-12))
        num = 1.0 - lamk
        den = 1.0 - lam1
        eps = 1e-8
        den_safe = torch.where(den.abs() < eps, den + eps, den)
        scale = (gamma1 / gammak) * (num / den_safe)
        is_one = (dt == 1) if not dt.dtype.is_floating_point else (dt == 1.0)
        if is_one.any():
            mask = is_one.view(h.size(0), h.size(1), 1, 1, 1)
            scale = torch.where(mask, torch.ones_like(scale, dtype=scale.dtype), scale)
        return h * scale
    def convlru(self, x, last_hidden_in, listT=None):
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        h = torch.fft.fft2(x.to(torch.cfloat), dim=(-2, -1), norm='ortho')
        h = torch.einsum('blcsw,dc->blcsw', h, self.proj_W)
        if self.proj_b is not None:
            h = h + self.proj_b.view(1, 1, C, 1, 1)
        if self.use_freq_prior:
            h = self.freq_prior(h)
        if S == W:
            nu_s, theta_s, _gamma_s = torch.exp(self.params_log_square).split((self.emb_ch, self.emb_ch, self.emb_ch), dim=0)
            nu0 = nu_s.view(1,1,C,S,1)
            th0 = theta_s.view(1,1,C,S,1)
            if self.lambda_type == "static":
                lam1 = torch.exp(torch.complex(-nu0, th0))
                if listT is None:
                    nu_t = torch.clamp(nu0, min=1e-6)
                    th_t = th0
                    lamb = torch.exp(torch.complex(-nu_t, th_t))
                    gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
                    x_in = h * gamma_t
                else:
                    lamb = lam1.pow(dt)
                    gamma_t = torch.sqrt(torch.clamp(1.0 - (lamb.abs() ** 2), min=1e-12))
                    x_in = self._apply_static_dt_scaling(h, lam1, listT)
            else:
                phi = h.abs().mean(dim=-1, keepdim=True)
                phi_prev = torch.empty_like(phi)
                phi_prev[:, 1:] = phi[:, :-1]
                phi_prev[:, 0].zero_()
                if (last_hidden_in is not None) and isinstance(last_hidden_in, tuple) and ('phi_last' in last_hidden_in[1]):
                    phi_prev[:, 0] = last_hidden_in[1]['phi_last']
                z = phi_prev.squeeze(-1)
                if self.exo_mode == "affine":
                    a = self.exo_affine_a.view(1,1,C,1,1)
                    b = self.exo_affine_b.view(1,1,C,1,1)
                    dnu = self.delta_scale_nu * torch.tanh(a * z.unsqueeze(-1) + b)
                    dth = self.delta_scale_th * torch.tanh(a * z.unsqueeze(-1) + b)
                    dnu = dnu * dt
                    dth = dth * dt
                else:
                    h1_nu = torch.einsum('blcs, csh -> blch', z, self.mod_nu_fc1_S).tanh()
                    dnu = torch.einsum('blch, chs -> blcs', h1_nu, self.mod_nu_fc2_S).unsqueeze(-1)
                    h1_th = torch.einsum('blcs, csh -> blch', z, self.mod_th_fc1_S).tanh()
                    dth = torch.einsum('blch, chs -> blcs', h1_th, self.mod_th_fc2_S).unsqueeze(-1)
                    dnu = self.delta_scale_nu * torch.tanh(dnu) * dt
                    dth = self.delta_scale_th * torch.tanh(dth) * dt
                nu_t = torch.clamp(nu0 * dt + dnu, min=1e-6)
                th_t = torch.remainder(th0 * dt + dth, 2*math.pi)
                lamb = torch.exp(torch.complex(-nu_t, th_t))
                gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
                x_in = h * gamma_t
            if last_hidden_in is not None:
                prev_state = last_hidden_in[0] if isinstance(last_hidden_in, tuple) else last_hidden_in
                x_in = torch.concat([prev_state, x_in], dim=1)
                lamb = torch.concat([lamb[:, :1], lamb], dim=1)
                L2 = L + 1
            else:
                L2 = L
            h = self.pscan(lamb[:, :L2], x_in)
            last_hidden_out = h[:, -1:]
            h = self._ifft_and_fuse(h)
            if self.use_sh_prior:
                h = self.sh_prior(h)
            h = self.layer_norm(h)
            if last_hidden_in is not None:
                h = h[:, 1:]
            aux = {}
            if self.lambda_type == "exogenous":
                aux['phi_last'] = phi[:, -1:].detach()
            last_hidden_pkg = (last_hidden_out, aux) if aux else last_hidden_out
        else:
            nu_r, theta_r, _gamma_r = torch.exp(self.params_log_rank).split((self.emb_ch, self.emb_ch, self.emb_ch), dim=0)
            nu0 = nu_r.view(1,1,C,self.rank,1)
            th0 = theta_r.view(1,1,C,self.rank,1)
            z = self._project_to_square(h)
            if self.lambda_type == "static":
                lam1 = torch.exp(torch.complex(-nu0, th0))
                if listT is None:
                    nu_t = torch.clamp(nu0, min=1e-6)
                    th_t = th0
                    lamb = torch.exp(torch.complex(-nu_t, th_t))
                    gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
                    x_in = z * gamma_t
                else:
                    lamb = lam1.pow(dt)
                    gamma_t = torch.sqrt(torch.clamp(1.0 - (lamb.abs() ** 2), min=1e-12))
                    x_in = self._apply_static_dt_scaling(z, lam1, listT)
            else:
                phi = z.abs().mean(dim=-1, keepdim=True)
                phi_prev = torch.empty_like(phi)
                phi_prev[:, 1:] = phi[:, :-1]
                phi_prev[:, 0].zero_()
                if (last_hidden_in is not None) and isinstance(last_hidden_in, tuple) and ('phi_last' in last_hidden_in[1]):
                    phi_prev[:, 0] = last_hidden_in[1]['phi_last']
                zz = phi_prev.squeeze(-1)
                if self.exo_mode == "affine":
                    a = self.exo_affine_a.view(1,1,C,1,1)
                    b = self.exo_affine_b.view(1,1,C,1,1)
                    dnu = self.delta_scale_nu * torch.tanh(a * zz.unsqueeze(-1) + b)
                    dth = self.delta_scale_th * torch.tanh(a * zz.unsqueeze(-1) + b)
                    dnu = dnu * dt
                    dth = dth * dt
                else:
                    h1_nu = torch.einsum('blcr, crh -> blch', zz, self.mod_nu_fc1_R).tanh()
                    dnu = torch.einsum('blch, chr -> blcr', h1_nu, self.mod_nu_fc2_R).unsqueeze(-1)
                    h1_th = torch.einsum('blcr, crh -> blch', zz, self.mod_th_fc1_R).tanh()
                    dth = torch.einsum('blch, chr -> blcr', h1_th, self.mod_th_fc2_R).unsqueeze(-1)
                    dnu = self.delta_scale_nu * torch.tanh(dnu) * dt
                    dth = self.delta_scale_th * torch.tanh(dth) * dt
                nu_t = torch.clamp(nu0 * dt + dnu, min=1e-6)
                th_t = torch.remainder(th0 * dt + dth, 2*math.pi)
                lamb = torch.exp(torch.complex(-nu_t, th_t))
                gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
                x_in = z * gamma_t
            if last_hidden_in is not None:
                prev_state = last_hidden_in[0] if isinstance(last_hidden_in, tuple) else last_hidden_in
                x_in = torch.concat([prev_state, x_in], dim=1)
                lamb = torch.concat([lamb[:, :1], lamb], dim=1)
                L2 = L + 1
            else:
                L2 = L
            z = self.pscan(lamb[:, :L2], x_in)
            last_hidden_out = z[:, -1:]
            h = self._deproject_from_square(z)
            h = self._ifft_and_fuse(h)
            if self.use_sh_prior:
                h = self.sh_prior(h)
            h = self.layer_norm(h)
            if last_hidden_in is not None:
                h = h[:, 1:]
            aux = {}
            if self.lambda_type == "exogenous":
                aux['phi_last'] = phi[:, -1:].detach()
            last_hidden_pkg = (last_hidden_out, aux) if aux else last_hidden_out
        if self.gate_conv is not None:
            gate = self.gate_conv(h.permute(0,2,1,3,4)).permute(0,2,1,3,4)
            x = (1 - gate) * x + gate * h
        else:
            x = x + h
        return x, last_hidden_pkg
    def forward(self, x, last_hidden_in, listT=None):
        x, last_hidden_out = self.convlru(x, last_hidden_in, listT=listT)
        return x, last_hidden_out

class FeedForward(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.ffn_hidden_ch = getattr(args, "ffn_hidden_ch", 32)
        self.ffn_hidden_layers_num = getattr(args, "ffn_hidden_layers_num", 1)
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 7, 7), padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.ffn_hidden_ch, self.hidden_size, getattr(args, "hidden_activation", "ReLU"), use_cbam=self.use_cbam) for _ in range(self.ffn_hidden_layers_num)])
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, getattr(args, "hidden_activation", "ReLU"))()
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
    def forward(self, x):
        x_update = self.c_in(x.permute(0,2,1,3,4)); x_update = self.activation(x_update)
        for layer in self.c_hidden:
            x_update = layer(x_update)
        x_update = self.c_out(x_update)
        x_update = self.layer_norm(x_update.permute(0,2,1,3,4))
        x = x_update + x
        return x
