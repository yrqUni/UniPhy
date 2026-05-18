import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv2d(in_channels, dim, self.patch_size, stride=self.patch_size)

    def forward(self, x):
        ph, pw = self.patch_size
        h, w = x.shape[-2:]
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return self.proj(x)


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, heads, window_size, mlp_ratio):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, int(heads), batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _partition(self, x):
        b, h, w, c = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(
                x.permute(0, 3, 1, 2),
                (0, pad_w, 0, pad_h),
            ).permute(0, 2, 3, 1)
        hp, wp = x.shape[1], x.shape[2]
        windows = (
            x.reshape(b, hp // ws, ws, wp // ws, ws, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, ws * ws, c)
        )
        return windows, h, w, hp, wp

    def _merge(self, windows, h, w, hp, wp, batch_size):
        ws = self.window_size
        x = (
            windows.reshape(batch_size, hp // ws, wp // ws, ws, ws, self.dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, hp, wp, self.dim)
        )
        return x[:, :h, :w]

    def forward(self, x):
        batch_size = x.shape[0]
        windows, h, w, hp, wp = self._partition(x)
        y = self.norm1(windows)
        y, _ = self.attn(y, y, y, need_weights=False)
        windows = windows + y
        windows = windows + self.mlp(self.norm2(windows))
        return self._merge(windows, h, w, hp, wp, batch_size)


class SwinTransModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        depth,
        patch_size,
        img_height,
        img_width,
        heads=4,
        window_size=6,
        mlp_ratio=4,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.patch_size = tuple(patch_size)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.encoder = PatchEmbed(in_channels, embed_dim, self.patch_size)
        self.blocks = nn.ModuleList(
            [
                WindowAttentionBlock(embed_dim, heads, window_size, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.decoder = nn.ConvTranspose2d(
            embed_dim,
            out_channels,
            self.patch_size,
            stride=self.patch_size,
        )

    def sample_noise(self, x):
        return None

    def sample_block_noises(self, x):
        return [None] * self.depth

    def sample_rollout_noise(self, batch_size, steps, device, dtype=torch.float32):
        del batch_size, steps, device, dtype
        return None

    def forward_frame(self, x):
        z = self.encoder(x).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            z = block(z)
        z = z.permute(0, 3, 1, 2).contiguous()
        y = self.decoder(z)
        return y[..., : self.img_height, : self.img_width]

    def forward(self, x, dt, z=None, return_latent=False):
        del dt, z
        b, t, c, h, w = x.shape
        out = self.forward_frame(x.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        if return_latent:
            return out, None
        return out

    def forward_rollout(self, x_context, dt_context, dt_list, **kwargs):
        del dt_context, kwargs
        x_curr = x_context[:, -1]
        preds = []
        for _ in dt_list:
            x_curr = self.forward_frame(x_curr)
            preds.append(x_curr)
        return torch.stack(preds, dim=1)
