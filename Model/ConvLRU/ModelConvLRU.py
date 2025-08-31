import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pscan import PScan

def pad_circ_w(x, pW):
    if pW > 0:
        return F.pad(x, (pW, pW, 0, 0, 0, 0), mode='circular')
    return x

def icnr_(w, rH, rW, initializer=nn.init.kaiming_normal_):
    oc, ic, kD, kH, kW = w.shape
    r = rH * rW
    oc_small = oc // r
    k_small = torch.zeros(oc_small, ic, kD, kH, kW, device=w.device, dtype=w.dtype)
    initializer(k_small)
    k_small = k_small.repeat_interleave(r, dim=0)
    with torch.no_grad():
        w.copy_(k_small)

def px_unshuffle3d(x, rH, rW):
    n, c, d, h, w = x.shape
    x = x.view(n, c, d, h // rH, rH, w // rW, rW).permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    return x.view(n, c * rH * rW, d, h // rH, w // rW)

def px_shuffle3d(x, rH, rW):
    n, c_mul, d, h, w = x.shape
    c = c_mul // (rH * rW)
    x = x.view(n, c, rH, rW, d, h, w).permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    return x.view(n, c, d, h * rH, w * rW)

class VarMix(nn.Module):
    def __init__(self, c, ratio=4, groups=1, act="SiLU"):
        super().__init__()
        h = max(1, c // ratio)
        assert c % groups == 0 and h % groups == 0
        self.pw1 = nn.Conv3d(c, h, 1, groups=groups)
        self.act = getattr(nn, act)()
        self.pw2 = nn.Conv3d(h, c, 1)
    def forward(self, x):
        y = self.pw1(x)
        y = self.act(y)
        y = self.pw2(y)
        return x + y

class ConvBlock(nn.Module):
    def __init__(self, ch, hidden_size, act, gate_conv=True):
        super().__init__()
        self.gated = gate_conv
        self.conv3 = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), padding=(0, 1, 0))
        self.act3 = getattr(nn, act)()
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=1)
        self.act1 = getattr(nn, act)()
        self.ln = nn.LayerNorm([*hidden_size])
        self.gate = nn.Sequential(nn.Conv3d(ch, ch, 1), nn.Sigmoid()) if self.gated else None
    def forward(self, x):
        t = pad_circ_w(x, 1)
        t = self.conv3(t)
        t = self.act3(t)
        t = self.conv1(t)
        t = self.act1(t)
        t = self.ln(t.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        if self.gate is None:
            return x + t
        g = self.gate(t)
        return x + g * (t - x)

class Embed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_ch = args.input_ch
        self.in_sz = args.input_size
        self.emb_ch = args.emb_ch
        self.hid_ch = args.emb_hidden_ch
        self.hid_layers = args.emb_hidden_layers_num
        self.strategy = args.emb_strategy
        self.rH, self.rW = int(args.hidden_factor[0]), int(args.hidden_factor[1])
        assert self.strategy in ['conv', 'pxus']
        if self.strategy == "conv":
            self.down = nn.Conv3d(self.in_ch, self.in_ch, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            with torch.no_grad():
                d = self.down(torch.zeros(1, self.in_ch, 1, *self.in_sz))
                _, C, _, H, W = d.size()
            self.down_shape = (C, H, W)
            in_after = C
        else:
            Hd, Wd = self.in_sz[0] // self.rH, self.in_sz[1] // self.rW
            Cd = self.in_ch * self.rH * self.rW
            self.down_shape = (Cd, Hd, Wd)
            in_after = Cd
        self.hid_sz = (self.down_shape[1], self.down_shape[2])
        if self.hid_layers == 0:
            self.c_in = nn.Conv3d(in_after, self.emb_ch, kernel_size=(1, 7, 7), padding=(0, 3, 0))
            self.c_hid = None
            self.c_out = None
        else:
            self.c_in = nn.Conv3d(in_after, self.hid_ch, kernel_size=(1, 7, 7), padding=(0, 3, 0))
            self.c_hid = nn.ModuleList([ConvBlock(self.hid_ch, self.hid_sz, args.hidden_activation, gate_conv=getattr(args, "gate_conv", True)) for _ in range(self.hid_layers)])
            self.c_out = nn.Conv3d(self.hid_ch, self.emb_ch, kernel_size=1)
        self.act = getattr(nn, args.hidden_activation)()
        self.ln = nn.LayerNorm([*self.hid_sz])
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        if self.strategy == "conv":
            x = pad_circ_w(x, (self.rW - 1) // 2)
            x = self.down(x)
        else:
            x = px_unshuffle3d(x, self.rH, self.rW)
        x = pad_circ_w(x, 3)
        x = self.c_in(x)
        x = self.act(x)
        if self.c_hid is not None:
            for layer in self.c_hid:
                x = layer(x)
            x = self.c_out(x)
        x = self.ln(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return x.permute(0, 2, 1, 3, 4)

class Decode(nn.Module):
    def __init__(self, args, down_shape):
        super().__init__()
        self.out_ch = args.out_ch
        self.emb_ch = args.emb_ch
        self.hid_ch = args.dec_hidden_ch
        self.hid_layers = args.dec_hidden_layers_num
        self.hid_sz = [down_shape[1], down_shape[2]]
        self.strategy = args.dec_strategy
        assert self.strategy in ['deconv', 'pxsf']
        self.rH, self.rW = int(args.hidden_factor[0]), int(args.hidden_factor[1])
        out_after = self.hid_ch if self.hid_layers != 0 else self.emb_ch
        if self.strategy == "deconv":
            self.up = nn.ConvTranspose3d(self.emb_ch, out_after, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            H = self.hid_sz[0] * self.rH
            W = self.hid_sz[1] * self.rW
        else:
            self.pre = nn.Conv3d(self.emb_ch, out_after * self.rH * self.rW, kernel_size=(1, 3, 3), padding=(0, 1, 0))
            icnr_(self.pre.weight, self.rH, self.rW)
            if self.pre.bias is not None:
                nn.init.zeros_(self.pre.bias)
            H = self.hid_sz[0] * self.rH
            W = self.hid_sz[1] * self.rW
        if self.hid_layers != 0:
            self.c_hid = nn.ModuleList([ConvBlock(out_after, (H, W), args.hidden_activation, gate_conv=getattr(args, "gate_conv", True)) for _ in range(self.hid_layers)])
            self.c_out = nn.Conv3d(out_after, self.out_ch, kernel_size=1)
        else:
            self.c_hid = None
            self.c_out = nn.Conv3d(out_after, self.out_ch, kernel_size=1)
        self.act = getattr(nn, args.hidden_activation)()
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        if self.strategy == "deconv":
            x = self.up(x)
        else:
            x = pad_circ_w(x, 1)
            x = self.pre(x)
            x = px_shuffle3d(x, self.rH, self.rW)
        x = self.act(x)
        if self.c_hid is not None:
            for layer in self.c_hid:
                x = layer(x)
            x = self.c_out(x)
        else:
            x = self.c_out(x)
        return x.permute(0, 2, 1, 3, 4)

def _fourier_basis(W, S, device, dtype):
    phi = torch.arange(W, device=device, dtype=dtype) * (2 * math.pi / W)
    cols = [torch.ones(W, device=device, dtype=dtype)]
    m = 1
    while len(cols) < S:
        cols.append(torch.cos(m * phi))
        if len(cols) >= S:
            break
        cols.append(torch.sin(m * phi))
        m += 1
    V = torch.stack(cols[:S], dim=1)
    V = V / (V.norm(dim=0, keepdim=True) + 1e-8)
    return V

def _dct_basis(H, S, device, dtype):
    n = torch.arange(H, device=device, dtype=dtype)
    k = torch.arange(S, device=device, dtype=dtype)
    M = torch.cos(math.pi * (n[:, None] + 0.5) * k[None, :] / H)
    M[:, 0] = M[:, 0] / math.sqrt(2.)
    M = M * math.sqrt(2.0 / H)
    return M

def _assoc_legendre_cols(H, W, S, device, dtype):
    lat = torch.linspace(math.pi / 2, -math.pi / 2, H, device=device, dtype=dtype)
    mu = torch.sin(lat)
    phi = torch.arange(W, device=device, dtype=dtype) * (2 * math.pi / W)
    Vcos = [torch.ones(W, device=device, dtype=dtype)]
    Vsin = [torch.zeros(W, device=device, dtype=dtype)]
    mmax = max(1, S // 2)
    for m in range(1, mmax + 1):
        Vcos.append(torch.cos(m * phi))
        Vsin.append(torch.sin(m * phi))
    U_cols = []
    V_cols = []
    count = 0
    def double_factorial(n):
        v = 1.0
        for t in range(n, 0, -2):
            v *= t
        return v
    for m in range(0, mmax + 1):
        Pmm = ((-1.0) ** m) * double_factorial(2 * m - 1) * torch.pow(1 - mu * mu, 0.5 * m)
        Pm1m = mu * (2 * m + 1) * Pmm
        l = m
        if m == 0:
            if count < S:
                U_cols.append(Pmm); V_cols.append(Vcos[0]); count += 1
            l = m + 1
        if count >= S:
            break
        if l == m + 1 and count < S:
            if m == 0:
                if count < S:
                    U_cols.append(Pm1m); V_cols.append(Vcos[0]); count += 1
            else:
                if count < S:
                    U_cols.append(Pm1m); V_cols.append(Vcos[m]); count += 1
                if count < S:
                    U_cols.append(Pm1m); V_cols.append(Vsin[m]); count += 1
        P_lm_2 = Pmm
        P_lm_1 = Pm1m
        l = m + 2
        while count < S:
            Plm = ((2 * l - 1) * mu * P_lm_1 - (l + m - 1) * P_lm_2) / (l - m)
            if m == 0:
                U_cols.append(Plm); V_cols.append(Vcos[0]); count += 1
            else:
                U_cols.append(Plm); V_cols.append(Vcos[m]); count += 1
                if count >= S:
                    break
                U_cols.append(Plm); V_cols.append(Vsin[m]); count += 1
            P_lm_2 = P_lm_1
            P_lm_1 = Plm
            l += 1
        if count >= S:
            break
    U = torch.stack([u / (u.norm() + 1e-8) for u in U_cols[:S]], dim=1)
    V = torch.stack([v / (v.norm() + 1e-8) for v in V_cols[:S]], dim=1)
    return U, V

class SpectralProj(nn.Module):
    def __init__(self, H, W, S, mode='fft_dct', orth_every=0):
        super().__init__()
        U0, V0 = self._init_bases(H, W, S, mode)
        self.S = S
        self.mode = mode
        self.U0 = nn.Parameter(U0, requires_grad=False)
        self.V0 = nn.Parameter(V0, requires_grad=False)
        self.dU = nn.Parameter(torch.zeros_like(U0))
        self.dV = nn.Parameter(torch.zeros_like(V0))
        self.orth_every = orth_every
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)
    def _init_bases(self, H, W, S, mode):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        if mode == 'fft_dct':
            U0 = _dct_basis(H, S, device, dtype)
            V0 = _fourier_basis(W, S, device, dtype)
        elif mode == 'sph_harm':
            U0, V0 = _assoc_legendre_cols(H, W, S, device, dtype)
        else:
            U0 = torch.randn(H, S, device=device, dtype=dtype)
            V0 = torch.randn(W, S, device=device, dtype=dtype)
            U0 = U0 / (U0.norm(dim=0, keepdim=True) + 1e-8)
            V0 = V0 / (V0.norm(dim=0, keepdim=True) + 1e-8)
        return U0, V0
    @torch.no_grad()
    def _orth(self, U, V):
        if self.orth_every <= 0 or int(self.step.item()) % self.orth_every != 0:
            return U, V
        s = U.shape[1]
        if s > min(U.shape[0], V.shape[0]):
            return U, V
        qU, _ = torch.linalg.qr(U, mode='reduced')
        qV, _ = torch.linalg.qr(V, mode='reduced')
        return qU[:, :s], qV[:, :s]
    def forward(self, x):
        self.step.add_(1)
        U = (self.U0 + self.dU).to(x.dtype).to(x.device)
        V = (self.V0 + self.dV).to(x.dtype).to(x.device)
        U, V = self._orth(U, V)
        x = torch.einsum('blchw,hs->blcsw', x, U)
        x = torch.einsum('blcsw,wt->blcst', x, V)
        return x, U, V
    def inverse(self, h, size_hw, U, V):
        h = torch.einsum('blcst,tw->blcsw', h, V.t())
        h = torch.einsum('blcsw,sh->blchw', h, U.t())
        return h

class ConvLRULayer(nn.Module):
    def __init__(self, args, down_shape):
        super().__init__()
        C = args.emb_ch
        H, W = down_shape[1], down_shape[2]
        self.S = getattr(args, "state_size", 64)
        self.proj = SpectralProj(H, W, self.S, mode=getattr(args, "prior_mode", "fft_dct"), orth_every=getattr(args, "orth_every", 0))
        self.a_logit = nn.Parameter(torch.zeros(C, self.S))
        self.g_logit = nn.Parameter(torch.zeros(C, self.S))
        self.ln = nn.LayerNorm([H, W])
        self.use_gate = getattr(args, 'gate_lru', False)
        self.gate = nn.Sequential(nn.Conv3d(C, C, 1), nn.Sigmoid()) if self.use_gate else None
        self.pscan = PScan.apply
        self.ms_enable = getattr(args, "ms_enable", True)
        self.ms_scale = getattr(args, "ms_scale", 2)
        self.S_ms = getattr(args, "ms_state_size", max(16, self.S // 2))
        if self.ms_enable:
            assert H % self.ms_scale == 0 and W % self.ms_scale == 0
            Hm, Wm = H // self.ms_scale, W // self.ms_scale
            self.proj_ms = SpectralProj(Hm, Wm, self.S_ms, mode=getattr(args, "prior_mode", "fft_dct"), orth_every=getattr(args, "orth_every", 0))
            self.a_ms = nn.Parameter(torch.zeros(C, self.S_ms))
            self.g_ms = nn.Parameter(torch.zeros(C, self.S_ms))
            self.ms_alpha = nn.Parameter(torch.tensor(0.5))
        self.eps = 1e-4
    def _scan(self, x, a_logit, g_logit, last_h):
        if last_h is not None:
            x = torch.cat([last_h, x], dim=1)
        B, Lx, C, S, _ = x.shape
        a = torch.sigmoid(a_logit)
        a = torch.clamp(a, self.eps, 1.0 - self.eps)
        a = a.view(1, 1, C, S, 1).expand(B, Lx, C, S, 1)
        g = F.softplus(g_logit)
        g = torch.clamp(g, min=self.eps)
        g = g.view(1, 1, C, S, 1).expand(B, Lx, C, S, 1)
        x = g * x
        x = self.pscan(a, x)
        last = x[:, -1:]
        if last_h is not None:
            x = x[:, 1:]
        return x, last
    def forward(self, x, last_h):
        B, L, C, H, W = x.shape
        h, U, V = self.proj(x)
        h, last_main = self._scan(h, self.a_logit, self.g_logit, last_h)
        h = self.proj.inverse(h, (H, W), U, V)
        if self.ms_enable:
            t = x.view(B * L, C, H, W)
            t = F.avg_pool2d(t, kernel_size=self.ms_scale, stride=self.ms_scale)
            Hm, Wm = H // self.ms_scale, W // self.ms_scale
            t = t.view(B, L, C, Hm, Wm)
            hm, Um, Vm = self.proj_ms(t)
            hm, _ = self._scan(hm, self.a_ms, self.g_ms, None)
            hm = self.proj_ms.inverse(hm, (Hm, Wm), Um, Vm)
            hm = F.interpolate(hm.view(B * L * C, 1, Hm, Wm), size=(H, W), mode='bilinear', align_corners=False).view(B, L, C, H, W)
            w = torch.sigmoid(self.ms_alpha)
            h = h + w * hm
        h = self.ln(h)
        if self.gate is None:
            y = x + h
        else:
            gmask = self.gate(h.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            y = x + gmask * (h - x)
        return y, last_main

class FFN(nn.Module):
    def __init__(self, args, down_shape):
        super().__init__()
        C = args.emb_ch
        H, W = down_shape[1], down_shape[2]
        self.c_in = nn.Conv3d(C, args.ffn_hidden_ch, kernel_size=(1, 7, 7), padding=(0, 3, 0))
        self.act = getattr(nn, args.hidden_activation)()
        self.blocks = nn.ModuleList([ConvBlock(args.ffn_hidden_ch, (H, W), args.hidden_activation, gate_conv=getattr(args, "gate_conv", True)) for _ in range(args.ffn_hidden_layers_num)])
        self.c_out = nn.Conv3d(args.ffn_hidden_ch, C, kernel_size=1)
        self.ln = nn.LayerNorm([H, W])
        self.vmix_in = VarMix(C, ratio=getattr(args, "mix_ratio", 4), groups=getattr(args, "mix_groups", 1), act=getattr(args, "mix_act", "SiLU"))
        self.vmix_out = VarMix(C, ratio=getattr(args, "mix_ratio", 4), groups=getattr(args, "mix_groups", 1), act=getattr(args, "mix_act", "SiLU"))
    def forward(self, x):
        y = self.vmix_in(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        t = y.permute(0, 2, 1, 3, 4)
        t = pad_circ_w(t, 3)
        t = self.c_in(t)
        t = self.act(t)
        for blk in self.blocks:
            t = blk(t)
        t = self.c_out(t)
        t = self.ln(t.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        y = y + t.permute(0, 2, 1, 3, 4)
        y = self.vmix_out(y.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return y

class ConvLRUBlock(nn.Module):
    def __init__(self, args, down_shape):
        super().__init__()
        self.lru = ConvLRULayer(args, down_shape)
        self.ffn = FFN(args, down_shape)
    def forward(self, x, last_h):
        x, h_last = self.lru(x, last_h)
        x = self.ffn(x)
        return x, h_last

class ConvLRUModel(nn.Module):
    def __init__(self, args, down_shape):
        super().__init__()
        self.blocks = nn.ModuleList([ConvLRUBlock(args, down_shape) for _ in range(args.convlru_num_blocks)])
        self.grad_ckpt = getattr(args, "grad_checkpoint", False)
    def forward(self, x, last_hs=None):
        outs = []
        if self.grad_ckpt and last_hs is not None:
            for i, blk in enumerate(self.blocks):
                last_in = last_hs[i]
                def _run(inp, _blk=blk, _last=last_in):
                    y, h = _blk(inp, _last)
                    return y, h
                x, h_last = torch.utils.checkpoint.checkpoint(_run, x, use_reentrant=False)
                outs.append(h_last)
        else:
            for i, blk in enumerate(self.blocks):
                last_in = None if last_hs is None else last_hs[i]
                x, h_last = blk(x, last_in)
                outs.append(h_last)
        return x, outs

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed = Embed(args)
        self.decode = Decode(args, self.embed.down_shape)
        self.model = ConvLRUModel(args, self.embed.down_shape)
        self.out_act = getattr(nn, args.output_activation)()
        self._init_trunc()
    def _init_trunc(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            for n, p in self.named_parameters():
                if ('norm' in n) or ('ln' in n) or n.endswith('dU') or n.endswith('dV'):
                    continue
                if torch.is_complex(p):
                    p.real.uniform_(2 * l - 1, 2 * u - 1); p.imag.uniform_(2 * l - 1, 2 * u - 1)
                    p.real.erfinv_(); p.imag.erfinv_()
                    p.real.mul_(std * math.sqrt(2.)); p.imag.mul_(std * math.sqrt(2.))
                    p.real.add_(mean); p.imag.add_(mean)
                else:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_(); p.mul_(std * math.sqrt(2.)); p.add_(mean)
    def forward(self, x, mode, out_gen_num=None, gen_factor=None):
        if mode == 'p':
            x = self.embed(x)
            x, _ = self.model(x, last_hs=None)
            x = self.decode(x)
            x = self.out_act(x)
            return x
        assert self.args.input_ch == self.args.out_ch
        outs = []
        x = self.embed(x)
        x, last_hs = self.model(x, last_hs=None)
        x = self.decode(x)
        g = gen_factor if gen_factor is not None else x.size(1)
        g = max(1, min(g, x.size(1)))
        x = self.out_act(x[:, -g:])
        outs.append(x)
        n = max(0, (out_gen_num or 1) - 1)
        for _ in range(n):
            x = self.embed(x)
            x, last_hs = self.model(x, last_hs=last_hs)
            x = self.decode(x)
            g = gen_factor if gen_factor is not None else x.size(1)
            g = max(1, min(g, x.size(1)))
            x = self.out_act(x[:, -g:])
            outs.append(x)
        return torch.concat(outs, dim=1)
