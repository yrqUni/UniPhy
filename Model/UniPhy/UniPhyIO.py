import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        
        self.proj = nn.Conv2d(in_ch, embed_dim * 2, kernel_size=patch_size, stride=patch_size)
        
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.padder.pad(x)
        x_emb = self.proj(x)
        x_real, x_imag = torch.chunk(x_emb, 2, dim=1)
        return torch.complex(x_real, x_imag)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionResBlock(nn.Module):
    def __init__(self, channels, cond_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.cond_proj = nn.Conv2d(cond_channels, channels, 1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.act = nn.SiLU()

    def forward(self, x, t_emb, condition):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = h + self.cond_proj(condition)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

class UniPhyDiffusionDecoder(nn.Module):
    def __init__(self, out_ch, latent_dim, patch_size=4, model_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        
        self.latent_proj = nn.ConvTranspose2d(
            latent_dim * 2, model_channels, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        self.input_conv = nn.Conv2d(out_ch, model_channels, 3, padding=1)
        
        self.res1 = DiffusionResBlock(model_channels, model_channels, model_channels)
        self.res2 = DiffusionResBlock(model_channels, model_channels, model_channels)
        self.res3 = DiffusionResBlock(model_channels, model_channels, model_channels)
        
        self.out_conv = nn.Conv2d(model_channels, out_ch, 3, padding=1)
        
        nn.init.uniform_(self.out_conv.weight, -1e-5, 1e-5)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_latent, x_noisy, t):
        z_cat = torch.cat([z_latent.real, z_latent.imag], dim=1)
        cond_feat = self.latent_proj(z_cat)
        
        target_h, target_w = x_noisy.shape[-2:]
        cond_feat = cond_feat[..., :target_h, :target_w]
        
        t_emb = self.time_mlp(t)
        
        x = self.input_conv(x_noisy)
        
        x = self.res1(x, t_emb, cond_feat)
        x = self.res2(x, t_emb, cond_feat)
        x = self.res3(x, t_emb, cond_feat)
        
        out = self.out_conv(x)
        return out

    @torch.no_grad()
    def sample(self, z_latent, shape, device, steps=20):
        return torch.randn(shape, device=device)

