import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_safe_groups(channels: int, target: int = 8) -> int:
    if channels % target == 0:
        return target
    for g in [4, 2, 1]:
        if channels % g == 0:
            return g
    return 1

class Padder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.pad_h = 0
        self.pad_w = 0

    def pad(self, x):
        H, W = x.shape[-2:]
        self.pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        self.pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='replicate')
        return x

    def unpad(self, x):
        H, W = x.shape[-2:]
        target_h = H - self.pad_h
        target_w = W - self.pad_w
        return x[..., :target_h, :target_w]

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeAwareResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        g1 = get_safe_groups(dim_out, groups)
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(g1, dim_out),
            nn.SiLU(),
        )
        g2 = get_safe_groups(dim_out, groups)
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(g2, dim_out),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        scale_shift = self.mlp(time_emb)
        scale_shift = scale_shift.unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.block1(x)
        h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res_conv(x)

class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        self.unshuffle_dim = in_ch * (patch_size ** 2)
        self.proj = nn.Conv2d(self.unshuffle_dim, embed_dim * 2, kernel_size=1)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        x = self.padder.pad(x)
        x = F.pixel_unshuffle(x, self.patch_size)
        x_emb = self.proj(x)
        x_real, x_imag = torch.chunk(x_emb, 2, dim=1)
        out = torch.complex(x_real, x_imag)
        if is_5d:
            _, D, H_p, W_p = out.shape
            out = out.view(B, T, D, H_p, W_p)
        return out

class UniPhyDiffusionDecoder(nn.Module):
    def __init__(self, out_ch, latent_dim, patch_size=16, model_channels=128):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        noisy_in_dim = out_ch * (patch_size ** 2)
        self.noisy_proj = nn.Conv2d(noisy_in_dim, model_channels, kernel_size=1)
        self.latent_proj = nn.Conv2d(latent_dim * 2, model_channels, kernel_size=3, padding=1)
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.block1 = TimeAwareResBlock(model_channels * 2, model_channels, time_dim)
        self.block2 = TimeAwareResBlock(model_channels, model_channels, time_dim)
        self.block3 = TimeAwareResBlock(model_channels, model_channels, time_dim)
        self.final_proj = nn.Conv2d(model_channels, out_ch * (patch_size ** 2), kernel_size=1)
        nn.init.uniform_(self.final_proj.weight, -1e-5, 1e-5)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, z_latent, x_noisy, t):
        is_5d = x_noisy.ndim == 5
        if is_5d:
            B, T, C, H, W = x_noisy.shape
            x_noisy = x_noisy.reshape(B * T, C, H, W)
            if z_latent.ndim == 5:
                z_latent = z_latent.reshape(B * T, *z_latent.shape[2:])
            if t.numel() == B:
                t = t.repeat_interleave(T)
            elif t.numel() == B * T:
                t = t.view(-1)
        x_noisy = self.padder.pad(x_noisy)
        x = F.pixel_unshuffle(x_noisy, self.patch_size)
        x = self.noisy_proj(x)
        if z_latent.is_complex():
            z_cat = torch.cat([z_latent.real, z_latent.imag], dim=1)
        else:
            z_cat = z_latent
        h_lr, w_lr = x.shape[-2:]
        z_cat = z_cat[..., :h_lr, :w_lr]
        cond_feat = self.latent_proj(z_cat)
        t_emb = self.time_mlp(t)
        h = torch.cat([x, cond_feat], dim=1)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        out = self.final_proj(h)
        out = F.pixel_shuffle(out, self.patch_size)
        out = self.padder.unpad(out)
        if is_5d:
            _, C_o, H_o, W_o = out.shape
            out = out.view(B, T, C_o, H_o, W_o)
        return out

    @torch.no_grad()
    def sample(self, z_latent, shape, device, steps=20):
        img = torch.randn(shape, device=device)
        is_5d = len(shape) == 5
        B = shape[0]
        T = shape[1] if is_5d else 1
        betas = torch.linspace(1e-4, 0.02, 1000, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        times = torch.linspace(999, 0, steps, device=device).long()
        for i in range(steps):
            t_idx = times[i]
            t_val = torch.full((B,), t_idx, device=device, dtype=torch.long)
            t_input = t_val.unsqueeze(1).repeat(1, T) if is_5d else t_val
            beta_t = betas[t_idx]
            s1 = torch.sqrt(1 - alphas_cumprod[t_idx])
            s2 = 1.0 / torch.sqrt(alphas[t_idx])
            pred_noise = self.forward(z_latent, img, t_input)
            img = s2 * (img - beta_t / s1 * pred_noise)
            if i < steps - 1:
                img = img + torch.sqrt(beta_t) * torch.randn_like(img)
        return img

