import torch
import torch.nn as nn
import torch.nn.functional as F

class CliffordConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dim = dim // 4
        if self.dim * 4 != dim:
            raise ValueError("Clifford channels must be divisible by 4")
        
        self.w_s = nn.Parameter(torch.empty(self.dim, self.dim, kernel_size, kernel_size))
        self.w_x = nn.Parameter(torch.empty(self.dim, self.dim, kernel_size, kernel_size))
        self.w_y = nn.Parameter(torch.empty(self.dim, self.dim, kernel_size, kernel_size))
        self.w_b = nn.Parameter(torch.empty(self.dim, self.dim, kernel_size, kernel_size))
        
        self.bias = nn.Parameter(torch.zeros(dim))
        
        nn.init.kaiming_normal_(self.w_s)
        nn.init.kaiming_normal_(self.w_x)
        nn.init.kaiming_normal_(self.w_y)
        nn.init.kaiming_normal_(self.w_b)
        
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        row_s = torch.cat([self.w_s, -self.w_x, -self.w_y, -self.w_b], dim=1)
        row_x = torch.cat([self.w_x,  self.w_s, -self.w_b,  self.w_y], dim=1)
        row_y = torch.cat([self.w_y,  self.w_b,  self.w_s, -self.w_x], dim=1)
        row_b = torch.cat([self.w_b,  self.w_y, -self.w_x,  self.w_s], dim=1)
        
        W_fused = torch.cat([row_s, row_x, row_y, row_b], dim=0)
        
        return F.conv2d(x, W_fused, bias=self.bias, padding=self.padding)

