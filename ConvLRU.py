import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class IterativeConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.out_frames = args.out_frames
        self.model = ConvLRU(self.args)
    def forward(self, x):
        out = []
        for i in range(self.out_frames):
            out_ = self.model(x)[:, -1:, :, :, :]
            out.append(out_)
            x = torch.cat((x[:, 1:, :, :, :], out_.detach()), 1)
        return torch.cat(out, 1)

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.model = ConvLRUModel(self.args)
        self.decoder = Decoder(self.args)
        self.truncated_normal_init()
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
    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        x = self.decoder(x)
        return x

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = args.input_ch
        self.emb_ch = args.emb_ch
        self.embedding = nn.Conv2d(self.input_ch, self.emb_ch, kernel_size=3, padding='same')
    def forward(self, x):
        B, L, C, H, W = x.size()
        x = self.embedding(x.reshape(B*L, C, H, W)).reshape(B, L, -1, H, W)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = args.input_ch
        self.emb_ch = args.emb_ch
        self.embedding = nn.Conv2d(self.emb_ch, self.input_ch, kernel_size=3, padding='same')
    def forward(self, x):
        B, L, C, H, W = x.size()
        x = self.embedding(x.reshape(B*L, C, H, W)).reshape(B, L, -1, H, W)
        return x

class ConvLRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layers = args.convlru_num_blocks
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args) for _ in range(layers)])
    def forward(self, x):
        for lru_block in self.convlru_blocks:
            x = lru_block.forward(x)
        return x 
        
class ConvLRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        emb_ch = args.emb_ch
        hidden_ch = args.hidden_ch
        self.lru_layer = ConvLRULayer(emb_ch = emb_ch, 
                                      hidden_ch = hidden_ch, 
                                      input_size = args.input_size,
                                      dropout = args.convlru_dropout)
        self.feed_forward = PositionwiseFeedForward(emb_ch=emb_ch, 
                                                    hidden_ch=hidden_ch, 
                                                    input_size=args.input_size,
                                                    dropout=args.ffn_dropout)
    def forward(self, x):
        x = self.lru_layer(x)
        x = self.feed_forward(x)
        return x
    
class ConvLRULayer(nn.Module):
    def __init__(self,
                 emb_ch, 
                 hidden_ch, 
                 input_size,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.emb_ch = emb_ch
        self.hidden_ch = hidden_ch
        self.use_bias = use_bias
        self.input_size = input_size
        # init 
        u1 = torch.rand(hidden_ch, self.input_size)
        u2 = torch.rand(hidden_ch, self.input_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        # define layers
        self.in_proj_B = nn.Conv2d(self.emb_ch, self.hidden_ch, kernel_size=1, padding='same', bias=use_bias).to(torch.cfloat)
        self.in_proj_P_ = nn.Conv2d(self.emb_ch, self.hidden_ch, kernel_size=1, padding='same', bias=use_bias).to(torch.cfloat)
        self.out_proj_C = nn.Conv2d(self.hidden_ch, self.emb_ch, kernel_size=1, padding='same', bias=use_bias).to(torch.cfloat)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm([self.emb_ch, self.input_size, self.input_size])
    def lru_parallel(self, i, h, lamb, B, L, C, H, W):
        if i == 1:
            lamb = lamb.unsqueeze(0)
        l = 2 ** i
        h = h.reshape(B * L // l, l, C, H, W)  
        h1, h2 = h[:, :l // 2, :, :, :], h[:, l // 2:, :, :, :]  
        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        lamb = torch.diag_embed(lamb)
        h2 = h2 + lamb.unsqueeze(0) * h1[:, -1:, :, :, :]
        h = torch.cat([h1, h2], axis=1)
        lamb = torch.diagonal(lamb, dim1=-2, dim2=-1)
        return h, lamb
    def forward(self, x):
        B, L, _, H, W = x.size()
        nu, theta, gamma = torch.exp(self.params_log).split((self.hidden_ch, self.hidden_ch, self.hidden_ch))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj_B(x.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.hidden_ch, H, W)
        h = torch.fft.fft2(h)
        h = self.in_proj_P_(x.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.hidden_ch, H, W)
        h = h * torch.diag_embed(gamma)
        log2_L = int(np.ceil(np.log2(L)))
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, B, L,  self.hidden_ch, H, W)
        x_ = self.out_proj_C(h.reshape(B*L, self.hidden_ch, H, W )).reshape(B, L, self.emb_ch, H, W).real
        x_ = self.dropout(x_)
        x_ = torch.fft.ifft2(x_)
        x_ = x_.real
        x_ = self.layer_norm(x_.reshape(B*L, self.emb_ch, H, W )).reshape(B, L, self.emb_ch, H, W)
        x = x_ + x
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_ch, hidden_ch, input_size, dropout):
        super().__init__()
        self.emb_ch = emb_ch
        self.hidden_ch = hidden_ch
        self.w_1 = nn.Conv2d(emb_ch, hidden_ch, kernel_size=3, padding='same')
        self.w_2 = nn.Conv2d(hidden_ch, emb_ch, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([emb_ch, input_size, input_size])
    def forward(self, x):
        B, L, _, H, W = x.size()
        x_ = self.w_1(x.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.hidden_ch, H, W)
        x_ = self.activation(x_)
        x_ = self.dropout(x_)
        x_ = self.w_2(x_.reshape(B*L, self.hidden_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        x_ = self.dropout(x_)
        x_ = self.layer_norm(x_)
        x = x_ + x
        return x
    