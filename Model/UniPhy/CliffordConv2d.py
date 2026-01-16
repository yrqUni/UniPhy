import torch
import torch.nn as nn

class CliffordConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dim = dim // 4
        if self.dim * 4 != dim:
            raise ValueError("Clifford channels must be divisible by 4")
        self.conv_s = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_x = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_y = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)
        self.conv_b = nn.Conv2d(self.dim, self.dim, kernel_size, padding=padding)

    def forward(self, x):
        s, vx, vy, b = torch.chunk(x, 4, dim=1)
        out_s = self.conv_s(s) - self.conv_x(vx) - self.conv_y(vy) - self.conv_b(b)
        out_x = self.conv_x(s) + self.conv_s(vx) - self.conv_b(vy) + self.conv_y(b)
        out_y = self.conv_y(s) + self.conv_b(vx) + self.conv_s(vy) - self.conv_x(b)
        out_b = self.conv_b(s) + self.conv_y(vx) - self.conv_x(vy) + self.conv_s(b)
        return torch.cat([out_s, out_x, out_y, out_b], dim=1)

