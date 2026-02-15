import torch
import torch.nn as nn
import torch.nn.functional as F


class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, img_height, img_width):
        super().__init__()
        self.in_ch = int(in_ch)
        self.embed_dim = int(embed_dim)
        if isinstance(patch_size, (tuple, list)):
            self.ph = int(patch_size[0])
            self.pw = int(patch_size[1])
        else:
            self.ph = int(patch_size)
            self.pw = int(patch_size)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.pad_h = (self.ph - self.img_height % self.ph) % self.ph
        self.pad_w = (self.pw - self.img_width % self.pw) % self.pw
        self.h_patches = (self.img_height + self.pad_h) // self.ph
        self.w_patches = (self.img_width + self.pad_w) // self.pw
        self.proj = nn.Conv2d(self.in_ch, self.embed_dim, kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw))
        self.pos = nn.Parameter(torch.randn(1, 1, self.embed_dim, self.h_patches, self.w_patches) * 0.02)

    def forward(self, x):
        bsz, t_len, channels, height, width = x.shape
        x = x.reshape(bsz * t_len, channels, height, width)
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        z = self.proj(x).reshape(bsz, t_len, self.embed_dim, self.h_patches, self.w_patches)
        z = z + self.pos
        return torch.complex(z, torch.zeros_like(z))


class UniPhyEnsembleDecoder(nn.Module):
    def __init__(self, out_ch, embed_dim, patch_size, img_height, img_width, ensemble_size=1):
        super().__init__()
        self.out_ch = int(out_ch)
        self.embed_dim = int(embed_dim)
        if isinstance(patch_size, (tuple, list)):
            self.ph = int(patch_size[0])
            self.pw = int(patch_size[1])
        else:
            self.ph = int(patch_size)
            self.pw = int(patch_size)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.ensemble_size = int(ensemble_size)
        self.h_patches = (self.img_height + (self.ph - self.img_height % self.ph) % self.ph) // self.ph
        self.w_patches = (self.img_width + (self.pw - self.img_width % self.pw) % self.pw) // self.pw
        self.member_emb = nn.Embedding(self.ensemble_size, self.embed_dim)
        self.proj = nn.Conv2d(self.embed_dim * 2, self.out_ch, kernel_size=1)

    def forward(self, z, member_idx=None):
        bsz, t_len, dim, h_p, w_p = z.shape
        z_real = torch.cat([z.real, z.imag], dim=2).reshape(bsz * t_len, dim * 2, h_p, w_p)
        if member_idx is None:
            member_idx = torch.zeros((bsz,), device=z.device, dtype=torch.long)
        emb = self.member_emb(member_idx).view(bsz, 1, dim, 1, 1).expand(bsz, t_len, dim, h_p, w_p)
        emb = torch.cat([emb, torch.zeros_like(emb)], dim=2).reshape(bsz * t_len, dim * 2, h_p, w_p)
        y = self.proj(z_real + emb)
        y = F.interpolate(y, size=(self.img_height, self.img_width), mode="bilinear", align_corners=False)
        return y.reshape(bsz, t_len, self.out_ch, self.img_height, self.img_width)
