import torch
import torch.nn as nn
import torch.nn.functional as F

from .UniPhyFFN import build_activation


class FlexiblePadder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.ph, self.pw = patch_size

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (self.ph - h % self.ph) % self.ph
        pad_w = (self.pw - w % self.pw) % self.pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x


class UniPhyEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        embed_dim,
        patch_size,
        img_height,
        img_width,
    ):
        super().__init__()
        self.ph, self.pw = patch_size
        self.padder = FlexiblePadder(patch_size)
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            build_activation("silu"),
        )
        h_patches = (
            img_height + (self.ph - img_height % self.ph) % self.ph
        ) // self.ph
        w_patches = (
            img_width + (self.pw - img_width % self.pw) % self.pw
        ) // self.pw
        self.pos_emb = nn.Parameter(
            torch.zeros(1, embed_dim, h_patches, w_patches)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _encode_4d(self, x):
        x = self.padder(x)
        x = self.proj(x)
        x = self.stem(x)
        if x.shape[-2:] != self.pos_emb.shape[-2:]:
            pos_emb = F.interpolate(
                self.pos_emb,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos_emb = self.pos_emb
        x = x + pos_emb
        return torch.complex(x, torch.zeros_like(x))

    def forward(self, x):
        batch_size, steps, channels, height, width = x.shape
        x_flat = x.reshape(batch_size * steps, channels, height, width)
        z = self._encode_4d(x_flat)
        _, dim, h_patches, w_patches = z.shape
        return z.reshape(batch_size, steps, dim, h_patches, w_patches)


class _PixelShuffleStage(nn.Module):
    def __init__(self, in_ch, out_ch, scale_h, scale_w):
        super().__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.expand = nn.Conv2d(
            in_ch,
            out_ch * scale_h * scale_w,
            kernel_size=1,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            build_activation("silu"),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        batch_tokens = x.shape[0]
        h = self.expand(x)
        out_channels = h.shape[1] // (self.scale_h * self.scale_w)
        h_patches, w_patches = h.shape[2], h.shape[3]
        h = h.view(
            batch_tokens,
            out_channels,
            self.scale_h,
            self.scale_w,
            h_patches,
            w_patches,
        )
        h = h.permute(0, 1, 4, 2, 5, 3)
        h = h.reshape(
            batch_tokens,
            out_channels,
            h_patches * self.scale_h,
            w_patches * self.scale_w,
        )
        return h + self.refine(h)


class UniPhyEnsembleDecoder(nn.Module):
    def __init__(
        self,
        out_ch,
        latent_dim,
        patch_size,
        model_channels,
        img_height,
        img_width,
    ):
        super().__init__()
        self.ph, self.pw = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.latent_proj = nn.Conv2d(
            latent_dim * 2,
            model_channels,
            kernel_size=3,
            padding=1,
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
            build_activation("silu"),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
        )
        self.block2 = nn.Sequential(
            build_activation("silu"),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
        )
        mid_ch = max(out_ch, model_channels // 4)
        self.stage1 = _PixelShuffleStage(model_channels, mid_ch, self.ph, 1)
        self.stage2 = _PixelShuffleStage(mid_ch, out_ch, 1, self.pw)
        self.out_smooth = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _decode_4d(self, z):
        z_real = torch.cat([z.real, z.imag], dim=1)
        hidden = self.latent_proj(z_real)
        hidden = hidden + self.block1(hidden)
        hidden = hidden + self.block2(hidden)
        out = self.stage1(hidden)
        out = self.stage2(out)
        out = self.out_smooth(out)
        return out[..., : self.img_height, : self.img_width]

    def forward(self, z):
        batch_size, steps, dim, h_patches, w_patches = z.shape
        z_flat = z.contiguous().view(batch_size * steps, dim, h_patches, w_patches)
        out = self._decode_4d(z_flat)
        _, out_channels, height, width = out.shape
        return out.view(batch_size, steps, out_channels, height, width)
