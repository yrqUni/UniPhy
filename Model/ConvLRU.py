import torch
import torch.nn as nn
import math
import numpy as np
from .pscan import pscan

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
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
    def forward(self, x = None, out_frames = None, mode = None):
        if mode == 'p':
            x = self.embedding(x)
            x = self.model(x)
            x = self.decoder(x)
        elif mode == 'i':
            x = self.embedding(x)
            x = self.model(x)
            x = self.decoder(x)
            for _ in range(out_frames):
                _x = x[:, -1:, :, :, :]
                _x = self.embedding(_x)
                _x = self.model(_x)
                _x = self.decoder(_x)
                x = torch.cat((x, _x), dim=1)
            x = x[:, -out_frames:, :, :, :]
        return x

class Conv_hidden(nn.Module):
    def __init__(self, ch, dropout, hidden_size):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([self.ch, *hidden_size])
    def forward(self, x):
        B, L, _, H, W = x.size()
        x_ = self.conv(x.reshape(B*L, self.ch, H, W)).reshape(B, L, self.ch, H, W)
        x_ = self.activation(x_)
        x_ = self.dropout(x_)
        x_ = self.layer_norm(x_)
        x = x_ + x
        return x

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = args.input_ch
        self.input_size = args.input_size
        self.emb_ch = args.emb_ch
        self.emb_hidden_ch = args.emb_hidden_ch
        self.emb_hidden_layers_num = args.emb_hidden_layers_num
        self.dropout_rate = args.emb_dropout
        self.downsp = nn.Conv2d(in_channels=self.input_ch, out_channels=self.input_ch, kernel_size=args.hidden_factor, stride=args.hidden_factor)
        with torch.no_grad():
            x = torch.zeros(1, self.input_ch, *self.input_size)
            x = self.downsp(x)
            _, C, H, W = x.size()
            self.input_downsp_shape = (C, H, W)
        self.hidden_size = (self.input_downsp_shape[1], self.input_downsp_shape[2])
        self.c_in = nn.Conv2d(C, self.emb_hidden_ch, kernel_size=3, padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.emb_hidden_ch, self.dropout_rate, self.hidden_size) for _ in range(self.emb_hidden_layers_num)])
        self.c_out = nn.Conv2d(self.emb_hidden_ch, self.emb_ch, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm([self.emb_ch, *self.hidden_size])
    def forward(self, x):
        B, L, C, H, W = x.size()
        x = self.downsp(x.reshape(B*L, C, H, W))
        _, C, H, W = x.size()
        x = x.reshape(B, L, C, H, W)
        B, L, _, H, W = x.size()
        x = self.c_in(x.reshape(B*L, -1, H, W))
        x = self.activation(x)
        x = self.dropout(x).reshape(B, L, -1, H, W)
        for layer in self.c_hidden:
            x = layer(x)
        x = self.c_out(x.reshape(B*L, -1, H, W))
        x = self.dropout(x)
        x = self.layer_norm(x).reshape(B, L, -1, H, W)
        return x

class Decoder(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.input_ch = args.input_ch
        self.emb_ch = args.emb_ch
        self.dec_hidden_ch = args.dec_hidden_ch
        self.dec_hidden_layers_num = args.dec_hidden_layers_num
        self.hidden_size = ([input_downsp_shape[1], input_downsp_shape[2]])
        self.dropout_rate = args.dec_dropout
        self.c_in_1 = nn.Conv2d(self.emb_ch, input_downsp_shape[0], kernel_size=3, padding='same')
        self.upsp = nn.ConvTranspose2d(in_channels=input_downsp_shape[0], out_channels=input_downsp_shape[0], kernel_size=args.hidden_factor, stride=args.hidden_factor)
        with torch.no_grad():
            x = torch.zeros(1, input_downsp_shape[0], input_downsp_shape[1], input_downsp_shape[2])
            x = self.upsp(x)
            _, C, H, W = x.size()
        self.c_in_2 = nn.Conv2d(C, self.dec_hidden_ch, kernel_size=3, padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.dec_hidden_ch, self.dropout_rate, (H, W)) for _ in range(self.dec_hidden_layers_num)])
        self.c_out = nn.Conv2d(self.dec_hidden_ch, self.input_ch, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self, x):
        B, L, _, H, W = x.size()
        x = self.c_in_1(x.reshape(B*L, self.emb_ch, H, W))
        x = self.upsp(x)
        _, _, H, W = x.size()
        x = self.c_in_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.reshape(B, L, self.dec_hidden_ch, H, W)
        for layer in self.c_hidden:
            x = layer(x)
        x = self.c_out(x.reshape(B*L, self.dec_hidden_ch, H, W))
        x = self.dropout(x).reshape(B, L, self.input_ch, H, W)
        return x

class ConvLRUModel(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        layers = args.convlru_num_blocks
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, input_downsp_shape) for _ in range(layers)])
    def forward(self, x):
        for lru_block in self.convlru_blocks:
            x = lru_block.forward(x)
        return x 

class ConvLRUBlock(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)
    def forward(self, x):
        x = self.lru_layer(x)
        x = self.feed_forward(x)
        return x
    
class ConvLRULayer(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = args.emb_ch 
        self.emb_ch = args.emb_ch
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.dropout = args.convlru_dropout
        # init 
        u1 = torch.rand(self.emb_ch, self.hidden_size[0])
        u2 = torch.rand(self.emb_ch, self.hidden_size[0])
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        # define layers
        self.proj_B = nn.Conv2d(self.emb_ch, self.emb_ch, kernel_size=1, padding='same', bias=self.use_bias).to(torch.cfloat)
        self.proj_C = nn.Conv2d(self.emb_ch, self.emb_ch, kernel_size=1, padding='same', bias=self.use_bias).to(torch.cfloat)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm([self.emb_ch, *self.hidden_size])
    def convlru(self, x):
        B, L, _, H, W = x.size()
        nu, theta, gamma = torch.exp(self.params_log).split((self.emb_ch, self.emb_ch, self.emb_ch))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = torch.fft.fft2(x.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.emb_ch, H, W)
        h = self.proj_B(h.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.emb_ch, H, W)
        h = h * gamma.reshape(1, 1, *gamma.shape, 1).expand(B, L, *gamma.shape, W)
        C, S = lamb.size()
        h = pscan(lamb.reshape(1, 1, C, S, 1).expand(1, 1, C, S, 1), h)
        h = torch.fft.ifft2(h)
        h = self.proj_C(h.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        h = h.real
        h = self.dropout(h)
        h = self.layer_norm(h.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        x = h + x
        return x
    def forward(self, x):
        x = self.convlru(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.emb_ch = args.emb_ch
        self.ffn_hidden_ch = args.ffn_hidden_ch
        self.ffn_hidden_layers_num = args.ffn_hidden_layers_num
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.dropout_rate = args.ffn_dropout
        self.c_in = nn.Conv2d(self.emb_ch, self.ffn_hidden_ch, kernel_size=3, padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.ffn_hidden_ch, self.dropout_rate, self.hidden_size) for _ in range(self.ffn_hidden_layers_num)])
        self.c_out = nn.Conv2d(self.ffn_hidden_ch, self.emb_ch, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm([self.emb_ch, *self.hidden_size])
    def forward(self, x):
        B, L, _, H, W = x.size()
        x_ = self.c_in(x.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.ffn_hidden_ch, H, W)
        x_ = self.activation(x_)
        x_ = self.dropout(x_)
        for layer in self.c_hidden:
            x_ = layer(x_)
        x_ = self.c_out(x_.reshape(B*L, self.ffn_hidden_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        x_ = self.dropout(x_)
        x_ = self.layer_norm(x_)
        x = x_ + x
        return x
    