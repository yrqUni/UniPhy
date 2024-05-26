import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
try:
    from .pscan import pscan
except:
    from pscan import pscan
# torch.autograd.set_detect_anomaly(True)

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
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
    def forward(self, x, mode, out_frames_num=None):
        assert mode in ['p_sigmoid', 'p_logits', 'i_sigmoid', 'i_logits'], f'mode should be either p_sigmoid, p_logits, i_sigmoid or i_logits, but got {mode}'
        if mode == 'p_logits':
            x = self.embedding(x)
            x, _ = self.convlru_model(x, last_hidden_ins=None)
            x = self.decoder(x, mode='p', condition=None)
            return x
        elif mode == 'p_sigmoid':
            x = self.embedding(x)
            x, _ = self.convlru_model(x, last_hidden_ins=None)
            x = self.decoder(x, mode='p', condition=None)
            x = torch.sigmoid(x)
            return x
        elif mode == 'i_logits':
            out = []
            x = self.embedding(x)
            condition = x.detach().clone()
            x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=None)
            x = x[:, -1:]
            x = self.decoder(x, mode='p', condition=None)
            out.append(x)
            for i in range(out_frames_num-1):
                x = self.embedding(torch.sigmoid(out[i]))
                x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=last_hidden_outs)
                x = x[:, -1:]
                x = self.decoder(x, mode='i', condition=condition)
                out.append(x)
            out = torch.concat(out, dim=1)
            return out
        elif mode == 'i_sigmoid':
            out = []
            x = self.embedding(x)
            condition = x.detach().clone()
            x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=None)
            x = x[:, -1:]
            x = self.decoder(x, mode='p', condition=None)
            out.append(torch.sigmoid(x))
            for i in range(out_frames_num-1):
                x = self.embedding(out[i])
                x, last_hidden_outs = self.convlru_model(x, last_hidden_ins=last_hidden_outs)
                x = x[:, -1:]
                x = self.decoder(x, mode='i', condition=condition)
                out.append(torch.sigmoid(x))
            out = torch.concat(out, dim=1)
            return out

class Conv_hidden(nn.Module):
    def __init__(self, ch, dropout, hidden_size):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding='same')
        self.activation = nn.LeakyReLU()
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
        self.activation = nn.LeakyReLU()
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
        self.activation = nn.LeakyReLU()
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
            if last_hidden_ins is not None: x, last_hidden_out = convlru_block.forward(x, last_hidden_ins[convlru_block_num])
            else: x, last_hidden_out = convlru_block.forward(x, None)
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
    def convlru(self, x, last_hidden_in):
        B, L, _, H, W = x.size()
        nu, theta, gamma = torch.exp(self.params_log).split((self.emb_ch, self.emb_ch, self.emb_ch))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = torch.fft.fft2(x.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.emb_ch, H, W)
        h = self.proj_B(h.reshape(B*L, self.emb_ch, H, W).to(torch.cfloat)).reshape(B, L, self.emb_ch, H, W)
        h = h * gamma.reshape(1, 1, *gamma.shape, 1).expand(B, L, *gamma.shape, W)
        C, S = lamb.size()
        if last_hidden_in is not None: 
            h = torch.concat([last_hidden_in, h[:, -1:]], dim=1)
            B, L, _, H, W = h.size()
        else:
            pass
        h = pscan(lamb.reshape(1, 1, C, S, 1).expand(B, L, C, S, 1), h)
        last_hidden_out = h[:, -1:]
        h = torch.fft.ifft2(h)
        h = self.proj_C(h.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        h = h.real
        h = self.dropout(h)
        h = self.layer_norm(h.reshape(B*L, self.emb_ch, H, W)).reshape(B, L, self.emb_ch, H, W)
        x = h + x
        return x, last_hidden_out
    def forward(self, x, last_hidden_in):
        x, last_hidden_out = self.convlru(x, last_hidden_in)
        return x, last_hidden_out

# class RoPEPositionEncoding(nn.Module):
#     def __init__(self, dim):
#         super(RoPEPositionEncoding, self).__init__()
#         self.dim = dim
#     def forward(self, x):
#         seq_len = x.size(1)
#         position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(math.log(10000.0) / self.dim))
#         pe = torch.zeros(seq_len, self.dim, device=x.device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         x = x + pe.unsqueeze(0)
#         return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x)  # B, L, 3 * D
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3, B, num_heads, L, head_dim
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B, num_heads, L, L
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)  # B, 1, 1, L
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # B, num_heads, L, head_dim
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # B, L, D
        attn_output = self.proj_dropout(self.out(attn_output))
        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        # self.position_encoding = RoPEPositionEncoding(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        # x = self.position_encoding(x)
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, attn_layers_num, attn_dim, num_heads, ffn_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(attn_dim, num_heads, ffn_dim, dropout)
            for _ in range(attn_layers_num)
        ])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
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
        self.remember_mixer = Transformer(args.dec_attn_layers_num, 
                                          self.emb_ch*input_downsp_shape[1]*input_downsp_shape[2], 
                                          args.dec_attn_num_heads, 
                                          args.dec_attn_ffn_dim_factor*self.emb_ch*input_downsp_shape[1]*input_downsp_shape[2], 
                                          args.dec_attn_dropout)
        self.c_in_1 = nn.Conv2d(self.emb_ch, input_downsp_shape[0], kernel_size=3, padding='same')
        self.upsp = nn.ConvTranspose2d(in_channels=input_downsp_shape[0], out_channels=input_downsp_shape[0], kernel_size=args.hidden_factor, stride=args.hidden_factor)
        with torch.no_grad():
            x = torch.zeros(1, input_downsp_shape[0], input_downsp_shape[1], input_downsp_shape[2])
            x = self.upsp(x)
            _, C, H, W = x.size()
        self.c_in_2 = nn.Conv2d(C, self.dec_hidden_ch, kernel_size=3, padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.dec_hidden_ch, self.dropout_rate, (H, W)) for _ in range(self.dec_hidden_layers_num)])
        self.c_out = nn.Conv2d(self.dec_hidden_ch, self.input_ch, kernel_size=3, padding='same')
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self, x, mode, condition):
        if mode == 'p':
            B, L, C, H, W = x.size()
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
        elif mode == 'i':
            B, L, C, H, W = x.size()
            x = self.remember_mixer(torch.concat([condition, x], dim=1).reshape(B, -1, C*H*W)).reshape(B, -1, C, H, W)[:, -1:]
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