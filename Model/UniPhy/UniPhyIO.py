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
        self.pad_h = 0
        self.pad_w = 0

    def forward(self, x):
        h, w = x.shape[-2:]
        self.pad_h = (self.ph - h % self.ph) % self.ph
        self.pad_w = (self.pw - w % self.pw) % self.pw
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode=self.mode)
        return x

    def unpad(self, x):
        h, w = x.shape[-2:]
        return x[..., :h - self.pad_h, :w - self.pad_w]


class UniPhyEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        embed_dim,
        patch_size=16,
        img_height=64,
        img_width=64,
    ):
        super().__init__()
        if isinstance(patch_size, (tuple, list)):
            self.ph, self.pw = patch_size
        else:
            self.ph = self.pw = patch_size

        self.padder = FlexiblePadder((self.ph, self.pw))

        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )

        h_patches = (img_height + (self.ph - img_height % self.ph) % self.ph) // self.ph
        w_patches = (img_width + (self.pw - img_width % self.pw) % self.pw) // self.pw

        self.pos_emb = nn.Parameter(torch.zeros(1, embed_dim, h_patches, w_patches))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1

        x = self.padder(x)
        x = self.proj(x)

        if x.shape[-2:] != self.pos_emb.shape[-2:]:
            pos_emb = F.interpolate(
                self.pos_emb, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            pos_emb = self.pos_emb

        x = x + pos_emb

        if is_5d:
            _, D, H_new, W_new = x.shape
            x = x.view(B, T, D, H_new, W_new)

        return torch.complex(x, torch.zeros_like(x))


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
        if isinstance(patch_size, (tuple, list)):
            self.ph, self.pw = patch_size
        else:
            self.ph = self.pw = patch_size

        self.img_height = img_height
        self.img_width = img_width
        self.ensemble_size = ensemble_size
        self.out_ch = out_ch

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
            model_channels, out_ch * self.ph * self.pw, kernel_size=1
        )

        self.out_smooth = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

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
            z = z.contiguous().view(B * T, D, H_p, W_p)
        else:
            B, D, H_p, W_p = z.shape
            T = 1

        z_real = torch.cat([z.real, z.imag], dim=1)

        h = self.latent_proj(z_real)

        if member_idx is not None:
            BT = h.shape[0]
            if member_idx.ndim == 0:
                member_idx = member_idx.unsqueeze(0).expand(BT)
            elif member_idx.numel() == B and is_5d:
                member_idx = member_idx.repeat_interleave(T)
            elif member_idx.numel() != BT:
                member_idx = member_idx.view(-1)
                if member_idx.shape[0] >= BT:
                    member_idx = member_idx[:BT]
                else:
                    raise ValueError(f"member_idx size {member_idx.shape} mismatch {BT}")

            member_idx = member_idx.to(z.device)
            member_feat = self.member_emb(member_idx)
            member_feat = member_feat.view(BT, -1, 1, 1)
            h = h + member_feat

        h = h + self.block(h)
        out = self.to_pixel(h)

        BT, _, H_grid, W_grid = out.shape
        out = out.view(BT, self.out_ch, self.ph, self.pw, H_grid, W_grid)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(BT, self.out_ch, H_grid * self.ph, W_grid * self.pw)

        out = self.out_smooth(out)

        out = out[..., :self.img_height, :self.img_width]

        if is_5d:
            out = out.view(B, T, self.out_ch, self.img_height, self.img_width)

        return out

