import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexiblePadder(nn.Module):
    def __init__(self, patch_size, mode="replicate"):
        super().__init__()
        self.patch_size = patch_size
        self.mode = mode
        self.pad_h = 0
        self.pad_w = 0

    def set_padding(self, h, w):
        self.pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        self.pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

    def pad(self, x):
        _, _, h, w = x.shape
        self.set_padding(h, w)
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode=self.mode)
        return x

    def unpad(self, x):
        _, _, h, w = x.shape
        if self.pad_h > 0 or self.pad_w > 0:
            h_end = h - self.pad_h if self.pad_h > 0 else h
            w_end = w - self.pad_w if self.pad_w > 0 else w
            x = x[:, :, :h_end, :w_end]
        return x


class LearnableSphericalPosEmb(nn.Module):
    def __init__(self, dim, h_dim, w_dim):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.lat_emb = nn.Parameter(torch.randn(1, dim // 2, h_dim, 1) * 0.02)
        self.lon_emb = nn.Parameter(torch.randn(1, dim // 2, 1, w_dim) * 0.02)

        lat = torch.linspace(-math.pi / 2, math.pi / 2, h_dim)
        lon = torch.linspace(0, 2 * math.pi, w_dim)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        self.register_buffer("cos_lat", torch.cos(lat_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_lat", torch.sin(lat_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("cos_lon", torch.cos(lon_grid).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_lon", torch.sin(lon_grid).unsqueeze(0).unsqueeze(0))

        self.geo_proj = nn.Conv2d(4, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        lat_emb = self.lat_emb
        lon_emb = self.lon_emb

        if H != self.h_dim or W != self.w_dim:
            lat_emb = F.interpolate(lat_emb, size=(H, 1), mode="bilinear", align_corners=False)
            lon_emb = F.interpolate(lon_emb, size=(1, W), mode="bilinear", align_corners=False)

        pos_lat = lat_emb.expand(B, -1, H, W)
        pos_lon = lon_emb.expand(B, -1, H, W)
        pos_emb = torch.cat([pos_lat, pos_lon], dim=1)

        cos_lat = self.cos_lat
        sin_lat = self.sin_lat
        cos_lon = self.cos_lon
        sin_lon = self.sin_lon

        if H != self.h_dim or W != self.w_dim:
            cos_lat = F.interpolate(cos_lat, size=(H, W), mode="bilinear", align_corners=False)
            sin_lat = F.interpolate(sin_lat, size=(H, W), mode="bilinear", align_corners=False)
            cos_lon = F.interpolate(cos_lon, size=(H, W), mode="bilinear", align_corners=False)
            sin_lon = F.interpolate(sin_lon, size=(H, W), mode="bilinear", align_corners=False)

        geo_feat = torch.cat([cos_lat, sin_lat, cos_lon, sin_lon], dim=1)
        geo_feat = geo_feat.expand(B, -1, -1, -1)
        geo_emb = self.geo_proj(geo_feat)

        return x + pos_emb + geo_emb


class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, img_height=64, img_width=64):
        super().__init__()
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width

        self.padder = FlexiblePadder(patch_size, mode="replicate")

        self.unshuffle_dim = in_ch * (patch_size ** 2)
        self.proj = nn.Conv2d(self.unshuffle_dim, embed_dim * 2, kernel_size=1)

        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim = (img_height + pad_h) // patch_size
        w_dim = (img_width + pad_w) // patch_size

        self.pos_emb = LearnableSphericalPosEmb(embed_dim * 2, h_dim, w_dim)

    def forward(self, x):
        is_5d = x.ndim == 5

        if is_5d:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        x = self.padder.pad(x)
        x = F.pixel_unshuffle(x, self.patch_size)
        x_emb = self.proj(x)
        x_emb = self.pos_emb(x_emb)

        x_real, x_imag = torch.chunk(x_emb, 2, dim=1)
        out = torch.complex(x_real, x_imag)

        if is_5d:
            _, D, H_p, W_p = out.shape
            out = out.reshape(B, T, D, H_p, W_p)

        return out


class UniPhyEnsembleDecoder(nn.Module):
    def __init__(
        self,
        out_ch,
        latent_dim,
        patch_size=16,
        model_channels=128,
        ensemble_size=10,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.ensemble_size = ensemble_size

        self.padder = FlexiblePadder(patch_size, mode="replicate")

        self.latent_proj = nn.Conv2d(
            latent_dim * 2, model_channels, kernel_size=3, padding=1
        )

        self.member_emb = nn.Embedding(ensemble_size, model_channels)

        self.block = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
        )

        self.final_proj = nn.Conv2d(
            model_channels, out_ch * (patch_size ** 2), kernel_size=1
        )

    def forward(self, z_latent, member_idx=None):
        is_5d = z_latent.ndim == 5

        if is_5d:
            B, T, D, H_p, W_p = z_latent.shape
            z_flat = z_latent.reshape(B * T, D, H_p, W_p)
        else:
            B, D, H_p, W_p = z_latent.shape
            T = 1
            z_flat = z_latent

        if z_flat.is_complex():
            z_cat = torch.cat([z_flat.real, z_flat.imag], dim=1)
        else:
            z_cat = z_flat

        x_feat = self.latent_proj(z_cat)

        if member_idx is None:
            member_idx = torch.zeros(
                (x_feat.shape[0],), dtype=torch.long, device=x_feat.device
            )
        elif member_idx.numel() == B and is_5d:
            member_idx = member_idx.repeat_interleave(T)

        m_emb = self.member_emb(member_idx).unsqueeze(-1).unsqueeze(-1)

        h = x_feat + m_emb
        h = self.block(h)

        out = self.final_proj(h)
        out = F.pixel_shuffle(out, self.patch_size)

        self.padder.set_padding(self.img_height, self.img_width)
        out = self.padder.unpad(out)

        _, C_actual, H_actual, W_actual = out.shape

        if is_5d:
            out = out.reshape(B, T, C_actual, H_actual, W_actual)

        return out
    