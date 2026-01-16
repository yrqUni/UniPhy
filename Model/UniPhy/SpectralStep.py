import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralStep(nn.Module):
    def __init__(self, in_ch, rank=32, w_freq=64):
        super().__init__()
        self.in_ch = in_ch
        self.rank = rank
        self.w_freq = w_freq
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, w_freq)),
            nn.Conv2d(in_ch, rank * 2, 1)
        )
        nn.init.uniform_(self.estimator[-1].weight, -0.01, 0.01)

    def forward(self, x, dt):
        B, C, H, W = x.shape
        x_spec = torch.fft.rfft2(x, norm="ortho")
        
        params = self.estimator(x)
        nu, theta = torch.chunk(params, 2, dim=1)
        nu = F.softplus(nu)
        theta = torch.tanh(theta) * math.pi
        
        if dt.dim() < nu.dim():
            dt_view = dt.view(B, 1, 1, 1)
        else:
            dt_view = dt
            
        decay = torch.exp(-nu * dt_view)
        angle = theta * dt_view
        
        operator = torch.complex(decay * torch.cos(angle), decay * torch.sin(angle))
        
        operator_sum = operator.sum(dim=1, keepdim=True)
        
        feat_spec = x_spec[:, :, :, :self.w_freq]
        
        if operator_sum.shape[2] == 1:
             feat_spec = feat_spec * operator_sum
        else:
             feat_spec = feat_spec * operator_sum
             
        x_out_spec = x_spec.clone()
        x_out_spec[:, :, :, :self.w_freq] = feat_spec
        
        return torch.fft.irfft2(x_out_spec, s=(H, W), norm="ortho")

