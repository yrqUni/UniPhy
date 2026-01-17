import torch
import torch.nn as nn
import torch.nn.functional as F

class CliffordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        assert in_channels % 4 == 0
        assert out_channels % 4 == 0
        
        self.dim_in = in_channels // 4
        self.dim_out = out_channels // 4
        
        self.conv_sb = nn.Conv2d(self.dim_in * 2, self.dim_out * 2, kernel_size, stride, padding)
        self.conv_v = nn.Conv2d(self.dim_in * 2, self.dim_out * 2, kernel_size, stride, padding)
        self.conv_v2sb = nn.Conv2d(self.dim_in * 2, self.dim_out * 2, kernel_size, stride, padding)
        self.conv_sb2v = nn.Conv2d(self.dim_in * 2, self.dim_out * 2, kernel_size, stride, padding)

    def forward(self, x):
        B, C, H, W = x.shape
        D = C // 4
        
        x_reshaped = x.view(B, 4, D, H, W)
        s = x_reshaped[:, 0]
        vx = x_reshaped[:, 1]
        vy = x_reshaped[:, 2]
        b = x_reshaped[:, 3]
        
        in_sb = torch.cat([s, b], dim=1)
        in_v = torch.cat([vx, vy], dim=1)
        
        out_sb_main = self.conv_sb(in_sb)
        out_v_main = self.conv_v(in_v)
        
        out_sb_cross = self.conv_v2sb(in_v)
        out_v_cross = self.conv_sb2v(in_sb)
        
        out_sb = out_sb_main + out_sb_cross
        out_v = out_v_main + out_v_cross
        
        s_out, b_out = torch.chunk(out_sb, 2, dim=1)
        vx_out, vy_out = torch.chunk(out_v, 2, dim=1)
        
        out = torch.stack([s_out, vx_out, vy_out, b_out], dim=1)
        out = out.view(B, 4 * self.dim_out, H, W)
        
        return out

