import torch
import torch.nn as nn
import numpy as np
from pscan import pscan
# torch.autograd.set_detect_anomaly(True)

class ConvLRUModel(nn.Module):
    def __init__(self, convlru_num_blocks, d_model, convlru_dropout, d_ffn, ffn_hidden_layers_num, ffn_dropout, input_shape):
        super().__init__()
        layers = convlru_num_blocks
        self.convlru_blocks = nn.ModuleList([ConvLRUBlock(d_model, convlru_dropout, d_ffn, ffn_hidden_layers_num, ffn_dropout, input_shape) for _ in range(layers)])
    def forward(self, x):
        for block in self.convlru_blocks:
            x = block(x)
        return x

class ConvLRUBlock(nn.Module):
    def __init__(self, d_model, convlru_dropout, d_ffn, ffn_hidden_layers_num, ffn_dropout, input_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(d_model, convlru_dropout, input_shape)
        self.feed_forward = FeedForward(d_model, d_ffn, ffn_hidden_layers_num, ffn_dropout, input_shape)
    def forward(self, x):
        x = self.lru_layer(x)
        x = self.feed_forward(x)
        return x
    
class ConvLRULayer(nn.Module):
    def __init__(self, d_model, convlru_dropout, input_shape):
        super().__init__()
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.d_model = d_model 
        self.hidden_size = [input_shape[1], input_shape[2]]
        self.dropout = convlru_dropout
        # init 
        u1 = torch.rand(self.d_model, self.hidden_size[0])
        u2 = torch.rand(self.d_model, self.hidden_size[0])
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        # define layers
        self.proj_B = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, padding='same', bias=self.use_bias).to(torch.cfloat)
        self.proj_C = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, padding='same', bias=self.use_bias).to(torch.cfloat)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm([self.d_model, *self.hidden_size])
    def convlru(self, x):
        B, L, _, H, W = x.size()
        nu, theta, gamma = torch.exp(self.params_log).split((self.d_model, self.d_model, self.d_model))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = torch.fft.fft2(x.reshape(B*L, self.d_model, H, W).to(torch.cfloat)).reshape(B, L, self.d_model, H, W)
        h = self.proj_B(h.reshape(B*L, self.d_model, H, W).to(torch.cfloat)).reshape(B, L, self.d_model, H, W)
        h = h * gamma.reshape(1, 1, *gamma.shape, 1).expand(B, L, *gamma.shape, W)
        C, S = lamb.size()
        h = pscan(lamb.reshape(1, 1, C, S, 1).expand(1, 1, C, S, 1), h)
        h = torch.fft.ifft2(h)
        h = self.proj_C(h.reshape(B*L, self.d_model, H, W)).reshape(B, L, self.d_model, H, W)
        h = h.real
        h = self.dropout(h)
        h = self.layer_norm(h.reshape(B*L, self.d_model, H, W)).reshape(B, L, self.d_model, H, W)
        x = h + x
        return x
    def forward(self, x):
        x = self.convlru(x)
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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, ffn_hidden_layers_num, ffn_dropout, input_shape):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.ffn_hidden_layers_num = ffn_hidden_layers_num
        self.hidden_size = [input_shape[1], input_shape[2]]
        self.dropout_rate = ffn_dropout
        self.c_in = nn.Conv2d(self.d_model, self.d_ffn, kernel_size=3, padding='same')
        self.c_hidden = nn.ModuleList([Conv_hidden(self.d_ffn, self.dropout_rate, self.hidden_size) for _ in range(self.ffn_hidden_layers_num)])
        self.c_out = nn.Conv2d(self.d_ffn, self.d_model, kernel_size=3, padding='same')
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm([self.d_model, *self.hidden_size])
    def forward(self, x):
        B, L, _, H, W = x.size()
        x_ = self.c_in(x.reshape(B*L, self.d_model, H, W)).reshape(B, L, self.d_ffn, H, W)
        x_ = self.activation(x_)
        x_ = self.dropout(x_)
        for layer in self.c_hidden:
            x_ = layer(x_)
        x_ = self.c_out(x_.reshape(B*L, self.d_ffn, H, W)).reshape(B, L, self.d_model, H, W)
        x_ = self.dropout(x_)
        x_ = self.layer_norm(x_)
        x = x_ + x
        return x

############################################
# batch_size = 2
# sequence_length = 10
# d_model = 8
# height = 16
# width = 16
# input_shape = (batch_size, sequence_length, d_model, height, width)

# x = torch.rand(input_shape)

# convlru_num_blocks = 2
# convlru_dropout = 0.1
# d_ffn = 16
# ffn_hidden_layers_num = 2
# ffn_dropout = 0.1

# model = ConvLRUModel(convlru_num_blocks, d_model, convlru_dropout, d_ffn, ffn_hidden_layers_num, ffn_dropout, (d_model, height, width))

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model.train()

# output = model(x)
# target = torch.rand_like(output)

# loss = criterion(output, target)
# print("Loss:", loss.item())

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(x.shape)
# print(output.shape)