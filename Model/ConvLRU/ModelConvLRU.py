import torch
import torch.nn as nn
import math
import numpy as np
from pscan import PScan, pscan_check
# torch.autograd.set_detect_anomaly(True)

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self._check_pscan()
        self.embedding = Embedding(self.args)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.out_activation = getattr(nn, args.output_activation)()
        self.truncated_normal_init()
    def _check_pscan(self):
        assert all(pscan_check()), "PScan implementation failed the test."
        print("PScan implementation passed the test.")
    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)
    def forward(self, x, mode, out_gen_num=None, gen_factor=None):
        assert mode in ['p', 'i'], f'Mode should be either p or i, but got {mode}'
        if mode == 'p':
            x = self.embedding(x)
            x, _ = self.convlru_model(x, last_hidden_ins=None)
            x = self.decoder(x)
            x = self.out_activation(x)
            return x
        elif mode == 'i':
            assert self.args.input_ch == self.args.out_ch, f'For iterative generation mode (i mode), input_ch should be equal to out_ch, but got {self.args.input_ch} and {self.args.out_ch}'
            out = []
            x = self.embedding(x)
            x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=None)
            x = self.decoder(x)
            x = x[:, -gen_factor:]
            x = self.out_activation(x)
            out.append(x)
            for _ in range(out_gen_num-1):
                x = self.embedding(x)
                x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=last_hidden_outs)
                x = self.decoder(x)[:, -gen_factor:]
                x = self.out_activation(x)
                out.append(x)
            out = torch.concat(out, dim=1)
            return out

class Conv_hidden(nn.Module):
    def __init__(self, ch, hidden_size, activation_func, use_mhsa=False, sa_dim=128):
        super().__init__()
        self.ch = ch
        self.use_mhsa = use_mhsa 
        self.conv3 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 3, 3), padding='same')
        self.activation3 = getattr(nn, activation_func)()
        self.conv1 = nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding='same')
        self.activation1 = getattr(nn, activation_func)()
        self.layer_norm_conv = nn.LayerNorm([*hidden_size])
        if self.use_mhsa:
            self.sa_dim = sa_dim
            self.mhsa_qk = nn.Linear(hidden_size[0]*hidden_size[1], sa_dim*2)
            self.pos_bias = nn.Parameter(torch.randn(1, self.ch, hidden_size[0]*hidden_size[1]))
            self.layer_norm_attn = nn.LayerNorm([*hidden_size])
            self.gate_conv = nn.Sequential(nn.Conv3d(self.ch, self.ch, kernel_size=(1, 1, 1), padding='same'), nn.Sigmoid())  
    def forward(self, x):
        B, _, L, H, W = x.size()
        x_update = self.conv3(x)
        x_update = self.activation3(x_update)
        x_update = self.conv1(x_update)
        x_update = self.activation1(x_update)
        if self.use_mhsa:
            x_update = self.layer_norm_attn(x_update.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x_update = x_update + x
            x_update = x_update.permute(0, 2, 1, 3, 4).reshape(B * L, self.ch, H * W)
            x_update = x_update + self.pos_bias
            qk = self.mhsa_qk(x_update)
            q, k = qk.split(self.sa_dim, dim=-1)
            attn = torch.einsum('bld,bmd->blm', q, k)
            attn = attn / math.sqrt(self.sa_dim)
            attn = torch.softmax(attn, dim=-1)
            x_update = torch.einsum('blm,bmd->bld', attn, x_update)
            x_update = x_update.reshape(B, L, self.ch, H, W)
            x_update = self.layer_norm_conv(x_update).permute(0, 2, 1, 3, 4)
            gate = self.gate_conv(x_update)
            x = (1 - gate) * x + gate * x_update
        else:
            x_update = x_update.permute(0, 2, 1, 3, 4)
            x_update = self.layer_norm_conv(x_update).permute(0, 2, 1, 3, 4)
            x = x_update + x
        return x

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = args.input_ch
        self.input_size = args.input_size
        self.emb_ch = args.emb_ch
        self.emb_hidden_ch = args.emb_hidden_ch
        self.emb_hidden_layers_num = args.emb_hidden_layers_num
        self.downsp = nn.Conv3d(in_channels=self.input_ch, out_channels=self.input_ch, 
                               kernel_size=(1, *args.hidden_factor), 
                               stride=(1, *args.hidden_factor))
        with torch.no_grad():
            _, C, _, H, W = self.downsp(torch.zeros(1, self.input_ch, 1, *self.input_size)).size()
            self.input_downsp_shape = (C, H, W)
        self.hidden_size = (self.input_downsp_shape[1], self.input_downsp_shape[2])
        if self.emb_hidden_layers_num == 0:
            self.c_in = nn.Conv3d(C, self.emb_ch, kernel_size=(1, 7, 7), padding='same')
            self.c_hidden = None
            self.c_out = None
        if self.emb_hidden_layers_num != 0:
            self.c_in = nn.Conv3d(C, self.emb_hidden_ch, kernel_size=(1, 7, 7), padding='same')
            self.c_hidden = nn.ModuleList([Conv_hidden(self.emb_hidden_ch, self.hidden_size, args.hidden_activation, use_mhsa=False, sa_dim=None) for _ in range(self.emb_hidden_layers_num)])
            self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, args.hidden_activation)()
        self.layer_norm = nn.LayerNorm([self.emb_ch, *self.hidden_size])
    def forward(self, x):
        x = self.downsp(x.permute(0, 2, 1, 3, 4))
        x = self.c_in(x)
        x = self.activation(x)
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
        self.output_ch = args.out_ch
        self.emb_ch = args.emb_ch
        self.dec_hidden_ch = args.dec_hidden_ch
        self.dec_hidden_layers_num = args.dec_hidden_layers_num
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.dec_strategy = args.dec_strategy
        self.rH, self.rW = int(args.hidden_factor[0]), int(args.hidden_factor[1])
        if self.dec_hidden_layers_num != 0:
            out_ch_after_up = self.dec_hidden_ch
        else:
            out_ch_after_up = self.emb_ch
        if self.dec_strategy == "deconv":
            self.upsp = nn.ConvTranspose3d(
                in_channels=self.emb_ch,
                out_channels=out_ch_after_up,
                kernel_size=(1, self.rH, self.rW),
                stride=(1, self.rH, self.rW)
            )
            with torch.no_grad():
                dummy = torch.zeros(1, self.emb_ch, 1, self.hidden_size[0], self.hidden_size[1])
                _, _, _, H, W = self.upsp(dummy).size()
        if self.dec_strategy == "pxsf":
            self.pre_shuffle_conv = nn.Conv3d(
                in_channels=self.emb_ch,
                out_channels=out_ch_after_up * self.rH * self.rW,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            )
            H = self.hidden_size[0] * self.rH
            W = self.hidden_size[1] * self.rW
        if self.dec_hidden_layers_num != 0:
            self.c_hidden = nn.ModuleList([
                Conv_hidden(out_ch_after_up, (H, W), args.hidden_activation,
                            use_mhsa=False, sa_dim=None)
                for _ in range(self.dec_hidden_layers_num)
            ])
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch,
                                   kernel_size=(1, 1, 1), padding='same')
        else:
            self.c_hidden = None
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch,
                                   kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, args.hidden_activation)()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        if self.dec_strategy == "deconv":
            x = self.upsp(x)
        if self.dec_strategy == "pxsf":
            x = self.pre_shuffle_conv(x)
            x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x)
            x = self.c_out(x)
        if self.c_hidden is None:
            x = self.c_out(x)
        return x.permute(0, 2, 1, 3, 4)

class ConvLRUModel(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        layers = args.convlru_num_blocks
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, input_downsp_shape) for _ in range(layers)])
    def forward(self, x, last_hidden_ins=None):
        last_hidden_outs = []
        convlru_block_num = 0
        for convlru_block in self.convlru_blocks:
            if last_hidden_ins is not None: 
                x, last_hidden_out = convlru_block(x, last_hidden_ins[convlru_block_num])
            else: 
                x, last_hidden_out = convlru_block(x, None)
            last_hidden_outs.append(last_hidden_out)
            convlru_block_num += 1
        return x, last_hidden_outs

class ConvLRUBlock(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)
    def forward(self, x, last_hidden_in):
        x, last_hidden_out = self.lru_layer(x, last_hidden_in)
        x = self.feed_forward(x)
        return x, last_hidden_out

class ConvLRULayer(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = args.emb_ch
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        # init
        u1 = torch.rand(self.emb_ch, self.hidden_size[0])
        u2 = torch.rand(self.emb_ch, self.hidden_size[0])
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        # define layers
        self.proj_B = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same', bias=self.use_bias).to(torch.cfloat)
        self.proj_C = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same', bias=self.use_bias).to(torch.cfloat)
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
        self.gate_conv = None
        if args.use_gate:
            self.gate_conv = nn.Sequential(nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same'), nn.Sigmoid())
        # 
        self.pscan = PScan.apply
    def convlru(self, x, last_hidden_in):
        B, L, _, _, W = x.size()
        nu, theta, gamma = torch.exp(self.params_log).split((self.emb_ch, self.emb_ch, self.emb_ch))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = torch.fft.fft2(x.to(torch.cfloat), dim=(-3, -2, -1), norm='ortho')
        h = self.proj_B(h.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        h = h * gamma.reshape(1, 1, *gamma.shape, 1).expand(B, L, *gamma.shape, W)
        C, S = lamb.size()
        if last_hidden_in is not None:
            h = torch.concat([last_hidden_in[:, -1:], h], dim=1)
            B, L, _, _, W = h.size()
        else:
            pass
        h = self.pscan(lamb.reshape(1, 1, C, S, 1).expand(B, L, C, S, 1), h)
        last_hidden_out = h[:, -1:]
        h = torch.fft.ifft2(h, dim=(-3, -2, -1), norm='ortho')
        h = self.proj_C(h.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        h = self.layer_norm(h.real)
        if last_hidden_in is not None:
            h = h[:, 1:]
        if self.gate_conv is not None:
            gate = self.gate_conv(h.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x = (1 - gate) * x + gate * h
        if self.gate_conv is None:
            x = x + h
        return x, last_hidden_out
    def forward(self, x, last_hidden_in):
        x, last_hidden_out = self.convlru(x, last_hidden_in)
        return x, last_hidden_out

class FeedForward(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.emb_ch = args.emb_ch
        self.ffn_hidden_ch = args.ffn_hidden_ch
        self.ffn_hidden_layers_num = args.ffn_hidden_layers_num
        self.use_mhsa = args.use_mhsa
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 7, 7), padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.ffn_hidden_ch, self.hidden_size, args.hidden_activation, use_mhsa=self.use_mhsa, sa_dim=128) for _ in range(self.ffn_hidden_layers_num)])
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding='same')
        self.activation = getattr(nn, args.hidden_activation)()
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
    def forward(self, x):
        x_update = self.c_in(x.permute(0, 2, 1, 3, 4))
        x_update = self.activation(x_update)
        for layer in self.c_hidden:
            x_update = layer(x_update)
        x_update = self.c_out(x_update)
        x_update = self.layer_norm(x_update.permute(0, 2, 1, 3, 4))
        x = x_update + x
        return x
