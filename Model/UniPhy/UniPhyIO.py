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

        nn.init.xavier_uniform_(self.geo_proj.weight)
        nn.init.zeros_(self.geo_proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        lat_emb = self.lat_emb
        lon_emb = self.lon_emb

        if H != self.h_dim or W != self.w_dim:
            lat_emb = F.interpolate(
                lat_emb, size=(H, 1), mode="bilinear", align_corners=False
            )
            lon_emb = F.interpolate(
                lon_emb, size=(1, W), mode="bilinear", align_corners=False
            )

        pos_emb = torch.cat([
            lat_emb.expand(B, -1, H, W),
            lon_emb.expand(B, -1, H, W),
        ], dim=1)

        cos_lat = self.cos_lat
        sin_lat = self.sin_lat
        cos_lon = self.cos_lon
        sin_lon = self.sin_lon

        if H != self.h_dim or W != self.w_dim:
            cos_lat = F.interpolate(
                cos_lat, size=(H, W), mode="bilinear", align_corners=False
            )
            sin_lat = F.interpolate(
                sin_lat, size=(H, W), mode="bilinear", align_corners=False
            )
            cos_lon = F.interpolate(
                cos_lon, size=(H, W), mode="bilinear", align_corners=False
            )
            sin_lon = F.interpolate(
                sin_lon, size=(H, W), mode="bilinear", align_corners=False
            )

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


class UniPhyDecoder(nn.Module):
    def __init__(
        self,
        out_ch,
        latent_dim,
        patch_size=16,
        model_channels=128,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.out_ch = out_ch

        self.padder = FlexiblePadder(patch_size, mode="replicate")

        self.latent_proj = nn.Conv2d(
            latent_dim * 2, model_channels, kernel_size=3, padding=1
        )

        self.block = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
        )

        self.to_pixel = nn.Conv2d(
            model_channels, out_ch * (patch_size ** 2), kernel_size=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, member_idx=None):
        is_5d = z.ndim == 5

        if is_5d:
            B, T, D, H_p, W_p = z.shape
            z = z.reshape(B * T, D, H_p, W_p)

        if z.is_complex():
            z_real = torch.cat([z.real, z.imag], dim=1)
        else:
            z_real = z

        h = self.latent_proj(z_real)
        h = h + self.block(h)
        out = self.to_pixel(h)
        out = F.pixel_shuffle(out, self.patch_size)
        out = self.padder.unpad(out)

        if is_5d:
            _, C, H, W = out.shape
            out = out.reshape(B, T, C, H, W)

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
        self.out_ch = out_ch

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

        self.to_pixel = nn.Conv2d(
            model_channels, out_ch * (patch_size ** 2), kernel_size=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.member_emb.weight, mean=0.0, std=0.02)

    def forward(self, z, member_idx=None):
        is_5d = z.ndim == 5

        if is_5d:
            B, T, D, H_p, W_p = z.shape
            z = z.reshape(B * T, D, H_p, W_p)
        else:
            B = z.shape[0]
            T = 1

        BT = z.shape[0]

        if z.is_complex():
            z_real = torch.cat([z.real, z.imag], dim=1)
        else:
            z_real = z

        h = self.latent_proj(z_real)

        if member_idx is not None:
            if member_idx.ndim == 0:
                member_idx = member_idx.unsqueeze(0).expand(BT)
            elif member_idx.numel() == B and is_5d:
                member_idx = member_idx.repeat_interleave(T)
            elif member_idx.numel() != BT:
                member_idx = member_idx.view(-1)[:BT]

            member_idx = member_idx.to(z.device)
            member_feat = self.member_emb(member_idx)
            member_feat = member_feat.view(BT, -1, 1, 1)
            h = h + member_feat

        h = h + self.block(h)
        out = self.to_pixel(h)
        out = F.pixel_shuffle(out, self.patch_size)
        out = self.padder.unpad(out)

        if is_5d:
            _, C, H, W = out.shape
            out = out.reshape(B, T, C, H, W)

        return out


class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        embed_dim,
        patch_size,
        img_height=64,
        img_width=64,
        num_scales=3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_scales = num_scales

        self.padder = FlexiblePadder(patch_size, mode="replicate")

        self.scale_convs = nn.ModuleList()
        for i in range(num_scales):
            scale_factor = 2 ** i
            kernel_size = 3 + 2 * i
            padding = kernel_size // 2
            self.scale_convs.append(
                nn.Conv2d(in_ch, embed_dim // num_scales, kernel_size, padding=padding)
            )

        self.fusion = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1)

        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim = (img_height + pad_h) // patch_size
        w_dim = (img_width + pad_w) // patch_size

        self.pos_emb = LearnableSphericalPosEmb(embed_dim * 2, h_dim, w_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        is_5d = x.ndim == 5

        if is_5d:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        x = self.padder.pad(x)

        scale_features = []
        for i, conv in enumerate(self.scale_convs):
            feat = conv(x)
            scale_features.append(feat)

        x_cat = torch.cat(scale_features, dim=1)
        x_emb = self.fusion(x_cat)

        _, _, H_pad, W_pad = x_emb.shape
        x_emb = F.avg_pool2d(x_emb, self.patch_size)
        x_emb = self.pos_emb(x_emb)

        x_real, x_imag = torch.chunk(x_emb, 2, dim=1)
        out = torch.complex(x_real, x_imag)

        if is_5d:
            _, D, H_p, W_p = out.shape
            out = out.reshape(B, T, D, H_p, W_p)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_ch,
        latent_dim,
        patch_size=16,
        model_channels=128,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.out_ch = out_ch

        self.padder = FlexiblePadder(patch_size, mode="replicate")

        self.latent_proj = nn.Conv2d(
            latent_dim * 2, model_channels, kernel_size=1
        )

        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        num_ups = int(math.log2(patch_size))
        current_ch = model_channels

        for i in range(num_ups):
            out_ch_block = current_ch // 2 if i < num_ups - 1 else current_ch // 2
            out_ch_block = max(out_ch_block, 32)
            self.up_blocks.append(
                nn.ConvTranspose2d(current_ch, out_ch_block, 4, 2, 1)
            )
            self.res_blocks.append(ResidualBlock(out_ch_block))
            current_ch = out_ch_block

        self.final_conv = nn.Conv2d(current_ch, out_ch, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, member_idx=None):
        is_5d = z.ndim == 5

        if is_5d:
            B, T, D, H_p, W_p = z.shape
            z = z.reshape(B * T, D, H_p, W_p)

        if z.is_complex():
            z_real = torch.cat([z.real, z.imag], dim=1)
        else:
            z_real = z

        h = self.latent_proj(z_real)

        for up, res in zip(self.up_blocks, self.res_blocks):
            h = up(h)
            h = res(h)

        out = self.final_conv(h)
        out = self.padder.unpad(out)

        if is_5d:
            _, C, H, W = out.shape
            out = out.reshape(B, T, C, H, W)

        return out


class LatentNormalizer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight_re = nn.Parameter(torch.ones(dim))
        self.weight_im = nn.Parameter(torch.ones(dim))
        self.bias_re = nn.Parameter(torch.zeros(dim))
        self.bias_im = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if not x.is_complex():
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            weight = self.weight_re.view(1, -1, 1, 1)
            bias = self.bias_re.view(1, -1, 1, 1)
            return x_norm * weight + bias

        x_re = x.real
        x_im = x.imag

        mean_re = x_re.mean(dim=1, keepdim=True)
        mean_im = x_im.mean(dim=1, keepdim=True)

        var_re = x_re.var(dim=1, keepdim=True, unbiased=False)
        var_im = x_im.var(dim=1, keepdim=True, unbiased=False)

        x_re_norm = (x_re - mean_re) / torch.sqrt(var_re + self.eps)
        x_im_norm = (x_im - mean_im) / torch.sqrt(var_im + self.eps)

        weight_re = self.weight_re.view(1, -1, 1, 1)
        weight_im = self.weight_im.view(1, -1, 1, 1)
        bias_re = self.bias_re.view(1, -1, 1, 1)
        bias_im = self.bias_im.view(1, -1, 1, 1)

        out_re = x_re_norm * weight_re + bias_re
        out_im = x_im_norm * weight_im + bias_im

        return torch.complex(out_re, out_im)


class TemporalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        self.dim = dim

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, t_idx=None):
        if t_idx is None:
            T = x.shape[1] if x.ndim == 5 else 1
            t_idx = torch.arange(T, device=x.device)

        if t_idx.ndim == 0:
            t_idx = t_idx.unsqueeze(0)

        pos_emb = self.pe[:, t_idx, :]

        if x.ndim == 5:
            B, T, D, H, W = x.shape
            pos_emb = pos_emb.view(1, T, D, 1, 1).expand(B, -1, -1, H, W)
            if x.is_complex():
                pos_emb = torch.complex(pos_emb, torch.zeros_like(pos_emb))
            return x + pos_emb
        else:
            B, D, H, W = x.shape
            pos_emb = pos_emb.view(1, D, 1, 1).expand(B, -1, H, W)
            if x.is_complex():
                pos_emb = torch.complex(pos_emb, torch.zeros_like(pos_emb))
            return x + pos_emb
        