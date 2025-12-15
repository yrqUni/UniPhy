import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pscan import PScan, pscan_check

def _kaiming_like_(tensor):
    nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="relu")
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

def _act(name):
    cls = getattr(nn, name)
    try:
        return cls(inplace=True)
    except TypeError:
        return cls()

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
        F = torch.matmul(self.A.transpose(1, 2), self.B)
        if self.mode == "exp":
            G = torch.exp(self.gain.view(self.C, 1, 1) * F)
        else:
            G = 1.0 + self.gain.view(self.C, 1, 1) * F
        return h * G.view(1, 1, self.C, self.S, self.W)

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
        self.register_buffer("Y_real", Y)

    @staticmethod
    def _latlon_to_spherical(H, W, device):
        lat = torch.linspace(math.pi / 2, -math.pi / 2, steps=H, device=device)
        lon = torch.linspace(-math.pi, math.pi, steps=W, device=device)
        theta = (math.pi / 2 - lat).unsqueeze(1).repeat(1, W)
        phi = lon.unsqueeze(0).repeat(H, 1)
        return theta, phi

    @staticmethod
    def _real_sph_harm_basis(theta, phi, Lmax):
        H, W = theta.shape
        device = theta.device
        dtype = theta.dtype
        return torch.zeros(Lmax * Lmax, H, W, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = self.Y_real
        if (Y.dtype != x.dtype) or (Y.device != x.device):
            Y = Y.to(dtype=x.dtype, device=x.device)
        coeff = torch.matmul(self.W1, self.W2)
        Yf = Y.view(self.K, self.H * self.W)
        bias = torch.matmul(coeff, Yf).view(self.C, self.H, self.W)
        bias = (self.gain.view(self.C, 1, 1) * bias).view(1, 1, self.C, self.H, self.W)
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

class Conv_hidden(nn.Module):
    def __init__(self, ch, hidden_size, activation_func, use_cbam=False):
        super().__init__()
        self.ch = ch
        self.use_cbam = bool(use_cbam)
        self.conv3 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 3, 3), padding="same")
        self.activation3 = _act(activation_func)
        self.conv1 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding="same")
        self.activation1 = _act(activation_func)
        self.layer_norm_conv = nn.LayerNorm([*hidden_size])
        if self.use_cbam:
            self.cbam = ChannelAttention2D(self.ch, reduction=16)
            self.gate_conv = nn.Sequential(
                nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding="same"),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x_update = self.conv3(x)
        x_update = self.activation3(x_update)
        x_update = self.conv1(x_update)
        x_update = self.activation1(x_update)
        if self.use_cbam:
            B, C, L, H, W = x_update.shape
            x_attn = self.cbam(x_update.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)).view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
            x_update = self.layer_norm_conv(x_attn.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            gate = self.gate_conv(x_update)
            x = (1 - gate) * x + gate * x_update
        else:
            x_update = self.layer_norm_conv(x_update.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x = x_update + x
        return x

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
        u1 = torch.rand(self.emb_ch, self.rank)
        u2 = torch.rand(self.emb_ch, self.rank)
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        theta_log = torch.log(u2 * (2 * torch.tensor(np.pi)))
        self.params_log_base = nn.Parameter(torch.stack([nu_log, theta_log], dim=0))
        self.dispersion_mod = nn.Parameter(torch.zeros(2, self.emb_ch, self.rank) * 0.01)
        self.mod_hidden = 32
        self.forcing_mlp = nn.Sequential(
            nn.Linear(self.emb_ch + 1, self.mod_hidden),
            nn.Tanh(),
            nn.Linear(self.mod_hidden, self.emb_ch * self.rank * 2)
        )
        self.forcing_scale = nn.Parameter(torch.tensor(0.1))
        self.U_row = nn.Parameter(torch.randn(self.emb_ch, S, self.rank, dtype=torch.cfloat) / math.sqrt(S))
        self.V_col = nn.Parameter(torch.randn(self.emb_ch, W, self.rank, dtype=torch.cfloat) / math.sqrt(W))
        C = self.emb_ch
        self.proj_W = nn.Parameter(torch.randn(C, C, dtype=torch.cfloat) / math.sqrt(C))
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat)) if self.use_bias else None
        self.post_ifft_conv_real = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.post_ifft_conv_imag = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.post_ifft_proj = nn.Conv3d(self.emb_ch * 2, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
        self.noise_level = nn.Parameter(torch.tensor(0.01))
        self.freq_prior = SpectralPrior2D(self.emb_ch, S, W, rank=8) if getattr(args, "use_freq_prior", False) else None
        self.sh_prior = SphericalHarmonicsPrior(self.emb_ch, S, W, Lmax=6) if getattr(args, "use_sh_prior", False) else None
        self.pscan = PScan.apply

    def _apply_forcing(self, x, dt):
        ctx = x.mean(dim=(-2, -1))
        dt_feat = dt.view(x.size(0), x.size(1), 1)
        inp = torch.cat([ctx, dt_feat], dim=-1)
        mod = self.forcing_mlp(inp)
        mod = mod.view(x.size(0), x.size(1), self.emb_ch, self.rank, 2)
        dnu = self.forcing_scale * torch.tanh(mod[..., 0])
        dth = self.forcing_scale * torch.tanh(mod[..., 1])
        return dnu.unsqueeze(-1), dth.unsqueeze(-1)

    def _fft_with_mirror_padding(self, x):
        B, L, C, S, W = x.shape
        pad_size = S // 4
        x_reshaped = x.view(B * L, C, S, W)
        x_pad = F.pad(x_reshaped, (0, 0, pad_size, pad_size), mode='reflect')
        x_pad = x_pad.view(B, L, C, S + 2 * pad_size, W)
        h = torch.fft.fft2(x_pad.to(torch.cfloat), dim=(-2, -1), norm="ortho")
        return h, pad_size

    def forward(self, x, last_hidden_in, listT=None):
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        h, pad_size = self._fft_with_mirror_padding(x)
        S_pad = h.shape[-2]
        h_perm = h.permute(0, 1, 3, 4, 2).contiguous().view(B * L * S_pad * W, C)
        h_proj = torch.matmul(h_perm, self.proj_W)
        h = h_proj.view(B, L, S_pad, W, C).permute(0, 1, 4, 2, 3).contiguous()
        if self.proj_b is not None:
            h = h + self.proj_b.view(1, 1, C, 1, 1)
        h_spatial = torch.fft.ifft2(h, dim=(-2, -1), norm="ortho")
        h_spatial = h_spatial[..., pad_size:-pad_size, :]
        h = torch.fft.fft2(h_spatial, dim=(-2, -1), norm="ortho")
        if self.freq_prior:
            h = self.freq_prior(h)
        Uc = self.U_row.conj()
        t = torch.matmul(h.permute(0, 1, 2, 4, 3), Uc)
        t = t.permute(0, 1, 2, 4, 3)
        zq = torch.matmul(t, self.V_col)
        nu_log, theta_log = self.params_log_base.unbind(dim=0)
        disp_nu, disp_th = self.dispersion_mod.unbind(dim=0)
        nu_base = torch.exp(nu_log + disp_nu).view(1, 1, C, self.rank, 1)
        th_base = torch.exp(theta_log + disp_th).view(1, 1, C, self.rank, 1)
        dnu_force, dth_force = self._apply_forcing(x, dt)
        nu_t = torch.clamp(nu_base * dt + dnu_force, min=1e-6)
        th_t = th_base * dt + dth_force
        lamb = torch.exp(torch.complex(-nu_t, th_t))
        if self.training:
            noise_std = self.noise_level * torch.sqrt(dt + 1e-6)
            noise = torch.randn_like(zq) * noise_std
            x_in = zq + noise
        else:
            x_in = zq
        gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
        x_in = x_in * gamma_t
        if last_hidden_in is not None:
            prev_state = last_hidden_in
            x_in = torch.concat([prev_state, x_in], dim=1)
            lamb = torch.concat([lamb[:, :1], lamb], dim=1)
        else:
            if L == 1:
                zero_prev = torch.zeros_like(x_in[:, :1])
                x_in = torch.cat([zero_prev, x_in], dim=1)
                lamb = torch.cat([lamb[:, :1], lamb], dim=1)
        L2 = x_in.size(1)
        z_out = self.pscan(lamb[:, :L2], x_in.contiguous())
        if last_hidden_in is not None or (last_hidden_in is None and L == 1):
            z_out = z_out[:, 1:]
        last_hidden_out = z_out[:, -1:]
        t = torch.matmul(z_out, self.V_col.conj().transpose(1, 2))
        t = t.permute(0, 1, 2, 4, 3)
        h_rec = torch.matmul(t, self.U_row.transpose(1, 2)).permute(0, 1, 2, 4, 3)
        h_spatial = torch.fft.ifft2(h_rec, dim=(-2, -1), norm="ortho")
        hr = self.post_ifft_conv_real(h_spatial.real.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        hi = self.post_ifft_conv_imag(h_spatial.imag.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        h_final = self.post_ifft_proj(torch.cat([hr, hi], dim=2).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        if self.sh_prior:
            h_final = self.sh_prior(h_final)
        h_final = self.layer_norm(h_final)
        if hasattr(self, 'gate_conv'):
            gate = self.gate_conv(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x = (1 - gate) * x + gate * h_final
        else:
            x = x + h_final
        return x, last_hidden_out

class ConvLRUBlock(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(self, x, last_hidden_in, listT=None):
        x, last_hidden_out = self.lru_layer(x, last_hidden_in, listT=listT)
        x = self.feed_forward(x)
        return x, last_hidden_out

class FeedForward(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.ffn_hidden_ch = getattr(args, "ffn_hidden_ch", 32)
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 7, 7), padding="same")
        self.c_hidden = Conv_hidden(self.ffn_hidden_ch, self.hidden_size, "ReLU", use_cbam=getattr(args, "use_cbam", False))
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.activation = _act("ReLU")
        self.layer_norm = nn.LayerNorm([*self.hidden_size])

    def forward(self, x):
        x_res = x
        x = self.c_in(x.permute(0, 2, 1, 3, 4))
        x = self.activation(x)
        x = self.c_hidden(x)
        x = self.c_out(x)
        x = self.layer_norm(x.permute(0, 2, 1, 3, 4))
        return x_res + x

class ConvLRUModel(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        layers = getattr(args, "convlru_num_blocks", 1)
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(args, input_downsp_shape) for _ in range(layers)])

    def forward(self, x, last_hidden_ins=None, listT=None):
        last_hidden_outs = []
        for idx, convlru_block in enumerate(self.convlru_blocks):
            h_in = last_hidden_ins[idx] if (last_hidden_ins is not None) else None
            x, last_hidden_out = convlru_block(x, h_in, listT=listT)
            last_hidden_outs.append(last_hidden_out)
        return x, last_hidden_outs

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = getattr(args, "input_ch", 1)
        self.emb_ch = getattr(args, "emb_ch", 32)
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        H, W = getattr(args, "input_size", (64, 64))
        self.input_downsp_shape = (self.emb_ch, H // self.rH, W // self.rW)
        if getattr(args, "emb_strategy", "pxus") == "conv":
            self.downsp = nn.Conv3d(self.input_ch, self.emb_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
        else:
            self.downsp = nn.Conv3d(self.input_ch, self.emb_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
        self.layer_norm = nn.LayerNorm([self.input_downsp_shape[1], self.input_downsp_shape[2]])

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.downsp(x)
        x = self.layer_norm(x.permute(0, 2, 1, 3, 4))
        return x

class Decoder(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.output_ch = getattr(args, "out_ch", 1)
        self.emb_ch = getattr(args, "emb_ch", 32)
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        if getattr(args, "dec_strategy", "pxsf") == "deconv":
            self.upsp = nn.ConvTranspose3d(self.emb_ch, self.emb_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
        else:
            self.upsp = nn.ConvTranspose3d(self.emb_ch, self.emb_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
        self.c_out = nn.Conv3d(self.emb_ch, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.upsp(x)
        x = self.c_out(x)
        mu, log_sigma = torch.chunk(x, 2, dim=1)
        sigma = F.softplus(log_sigma) + 1e-6
        return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4)

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        skip_contains = ["layer_norm", "params_log", "prior", "post_ifft", "spatial_mod", "forcing", "dispersion"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains):
                    continue
                if n.endswith(".bias"):
                    p.zero_()
                    continue
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, mode="p", out_gen_num=None, listT=None, listT_future=None):
        if mode == "p":
            x = self.embedding(x)
            x, _ = self.convlru_model(x, listT=listT)
            return self.decoder(x)
        out = []
        x_emb = self.embedding(x)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT)
        x_dec = self.decoder(x_hidden)
        x_step_dist = x_dec[:, -1:]
        x_step_mean = x_step_dist[..., :self.args.out_ch, :, :]
        out.append(x_step_dist)
        for t in range(out_gen_num - 1):
            dt = listT_future[:, t:t+1] if listT_future is not None else torch.ones_like(listT[:, 0:1])
            x_in = self.embedding(x_step_mean)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt)
            x_dec = self.decoder(x_hidden)
            x_step_dist = x_dec[:, -1:]
            x_step_mean = x_step_dist[..., :self.args.out_ch, :, :]
            out.append(x_step_dist)
        return torch.concat(out, dim=1)
