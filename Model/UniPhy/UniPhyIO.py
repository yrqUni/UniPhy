import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexiblePadder(nn.Module):
    def __init__(self, patch_size, mode="replicate"):
        super().__init__()
        if isinstance(patch_size, int):
            self.ph = self.pw = patch_size
        else:
            self.ph, self.pw = patch_size
        self.mode = mode

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (self.ph - h % self.ph) % self.ph
        pad_w = (self.pw - w % self.pw) % self.pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.mode)
        return x


class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, img_height, img_width):
        super().__init__()
        if isinstance(patch_size, (tuple, list)):
            self.ph, self.pw = patch_size
        else:
            self.ph = self.pw = patch_size
        self.padder = FlexiblePadder((self.ph, self.pw))
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )
        h_patches = (
            (img_height + (self.ph - img_height % self.ph) % self.ph)
            // self.ph
        )
        w_patches = (
            (img_width + (self.pw - img_width % self.pw) % self.pw)
            // self.pw
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, embed_dim, h_patches, w_patches)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _encode_4d(self, x):
        x = self.padder(x)
        x = self.proj(x)
        if x.shape[-2:] != self.pos_emb.shape[-2:]:
            pos_emb = F.interpolate(
                self.pos_emb, size=x.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        else:
            pos_emb = self.pos_emb
        x = x + pos_emb
        return torch.complex(x, torch.zeros_like(x))

    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            z = self._encode_4d(x_flat)
            _, D, Hp, Wp = z.shape
            return z.reshape(B, T, D, Hp, Wp)
        return self._encode_4d(x)


class _PixelShuffleStage(nn.Module):
    def __init__(self, in_ch, out_ch, scale_h, scale_w):
        super().__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.expand = nn.Conv2d(
            in_ch, out_ch * scale_h * scale_w, kernel_size=1,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        BT = x.shape[0]
        h = self.expand(x)
        C_out = h.shape[1] // (self.scale_h * self.scale_w)
        Hp, Wp = h.shape[2], h.shape[3]
        h = h.view(BT, C_out, self.scale_h, self.scale_w, Hp, Wp)
        h = h.permute(0, 1, 4, 2, 5, 3)
        h = h.reshape(BT, C_out, Hp * self.scale_h, Wp * self.scale_w)
        h = h + self.refine(h)
        return h


class UniPhyEnsembleDecoder(nn.Module):
    def __init__(self, out_ch, latent_dim, patch_size, model_channels,
                 ensemble_size, img_height, img_width):
        super().__init__()
        if isinstance(patch_size, (tuple, list)):
            self.ph, self.pw = patch_size
        else:
            self.ph = self.pw = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.latent_proj = nn.Conv2d(
            latent_dim * 2, model_channels, kernel_size=3, padding=1,
        )
        self.member_emb = nn.Embedding(ensemble_size, model_channels)
        self.block = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
        )
        mid_ch = max(out_ch, model_channels // 4)
        self.stage1 = _PixelShuffleStage(
            model_channels, mid_ch, self.ph, 1,
        )
        self.stage2 = _PixelShuffleStage(
            mid_ch, out_ch, 1, self.pw,
        )
        self.out_smooth = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.member_emb.weight, mean=0.0, std=0.02)

    def _decode_4d(self, z, member_idx):
        BT, D, H_p, W_p = z.shape
        z_real = torch.cat([z.real, z.imag], dim=1)
        h = self.latent_proj(z_real)
        if member_idx is not None:
            member_idx = member_idx.to(z.device)
            member_feat = self.member_emb(member_idx)
            member_feat = member_feat.view(BT, -1, 1, 1)
            h = h + member_feat
        h = h + self.block(h)
        out = self.stage1(h)
        out = self.stage2(out)
        out = self.out_smooth(out)
        out = out[..., :self.img_height, :self.img_width]
        return out

    def forward(self, z, member_idx=None):
        if z.ndim == 5:
            B, T, D, H_p, W_p = z.shape
            z_flat = z.contiguous().view(B * T, D, H_p, W_p)
            midx = None
            if member_idx is not None:
                midx = member_idx.repeat_interleave(T)
            out = self._decode_4d(z_flat, midx)
            _, C_out, H_out, W_out = out.shape
            return out.view(B, T, C_out, H_out, W_out)
        return self._decode_4d(z, member_idx)
    