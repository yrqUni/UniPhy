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

    def set_padding(self, h, w):
        self.pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        self.pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

    def pad(self, x):
        H, W = x.shape[-2:]
        self.set_padding(H, W)
        
        if self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, 0), mode='circular')
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, 0, self.pad_h), mode='replicate')
        return x

    def unpad(self, x):
        H, W = x.shape[-2:]
        target_h = H - self.pad_h
        target_w = W - self.pad_w
        return x[..., :target_h, :target_w]

class SphericalPosEmb(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        
        assert dim % 2 == 0
        dim_h = dim // 2
        dim_w = dim - dim_h

        inv_freq_h = 1.0 / (10000 ** (torch.arange(0, dim_h, 2).float() / dim_h))
        pos_h = torch.arange(h, dtype=torch.float)
        sin_inp_h = torch.einsum("i,j->ij", pos_h, inv_freq_h)
        emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)

        inv_freq_w = 1.0 / (10000 ** (torch.arange(0, dim_w, 2).float() / dim_w))
        pos_w = torch.arange(w, dtype=torch.float)
        
        pos_w = pos_w / w * 2 * math.pi
        sin_inp_w = torch.einsum("i,j->ij", pos_w, inv_freq_w)
        emb_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=-1)

        emb_h = emb_h.unsqueeze(1).repeat(1, w, 1) 
        emb_w = emb_w.unsqueeze(0).repeat(h, 1, 1) 
        
        self.register_buffer('emb', torch.cat([emb_h, emb_w], dim=-1).permute(2, 0, 1).unsqueeze(0))

    def forward(self, x):
        return x + self.emb

class MassCorrector(nn.Module):
    def __init__(self, height, mass_idx=0):
        super().__init__()
        self.mass_idx = mass_idx
        lat_indices = torch.arange(height)
        lat_rad = (lat_indices / (height - 1)) * math.pi - (math.pi / 2)
        weights = torch.cos(lat_rad)
        weights = weights / weights.mean()
        self.register_buffer('weights', weights.view(1, 1, 1, -1, 1))

    def forward(self, pred, ref):
        if ref.ndim == 5:
            ref_slice = ref[:, -1:, ...]
        else:
            ref_slice = ref.unsqueeze(1)
            
        ref_mass = (ref_slice[:, :, self.mass_idx:self.mass_idx+1] * self.weights).mean(dim=(-2, -1), keepdim=True)
        pred_mass = (pred[:, :, self.mass_idx:self.mass_idx+1] * self.weights).mean(dim=(-2, -1), keepdim=True)
        
        diff = pred_mass - ref_mass
        
        correction = torch.zeros_like(pred)
        correction[:, :, self.mass_idx:self.mass_idx+1] = diff
        
        return pred - correction

class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=16, img_height=64, img_width=64):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        self.unshuffle_dim = in_ch * (patch_size ** 2)
        
        self.proj = nn.Conv2d(self.unshuffle_dim, embed_dim * 2, kernel_size=1)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        pad_h = (patch_size - img_height % patch_size) % patch_size
        pad_w = (patch_size - img_width % patch_size) % patch_size
        h_dim = (img_height + pad_h) // patch_size
        w_dim = (img_width + pad_w) // patch_size
        
        self.pos_emb = SphericalPosEmb(embed_dim * 2, h_dim, w_dim)

    def forward(self, x):
        is_5d = x.ndim == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            
        x = self.padder.pad(x)
        x = F.pixel_unshuffle(x, self.patch_size)
        x_emb = self.proj(x)
        
        x_emb = self.pos_emb(x_emb)
        
        x_real, x_imag = torch.chunk(x_emb, 2, dim=1)
        out = torch.complex(x_real, x_imag)
        
        if is_5d:
            _, D, H_p, W_p = out.shape
            out = out.view(B, T, D, H_p, W_p)
        return out

class UniPhyEnsembleDecoder(nn.Module):
    def __init__(self, out_ch, latent_dim, patch_size=16, model_channels=128, ensemble_size=10, img_height=64):
        super().__init__()
        self.patch_size = patch_size
        self.padder = Padder(patch_size)
        self.ensemble_size = ensemble_size
        self.mass_corrector = MassCorrector(img_height)
        
        self.latent_proj = nn.Conv2d(latent_dim * 2, model_channels, kernel_size=3, padding=1)
        self.member_emb = nn.Embedding(ensemble_size, model_channels)
        
        self.block = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, 1, 1)
        )
        self.final_proj = nn.Conv2d(model_channels, out_ch * (patch_size ** 2), kernel_size=1)

    def forward(self, z_latent, x_ref, member_idx=None):
        is_5d = x_ref.ndim == 5
        if is_5d:
            B, T, C, H, W = x_ref.shape
            z_flat = z_latent.view(B * T, *z_latent.shape[2:])
        else:
            B, C, H, W = x_ref.shape
            T = 1
            z_flat = z_latent
            
        self.padder.set_padding(H, W)

        if z_flat.is_complex():
            z_cat = torch.cat([z_flat.real, z_flat.imag], dim=1)
        else:
            z_cat = z_flat
            
        x_feat = self.latent_proj(z_cat)
        
        if member_idx is None:
            member_idx = torch.zeros((x_feat.shape[0],), dtype=torch.long, device=x_feat.device)
        elif member_idx.numel() == B and is_5d:
            member_idx = member_idx.repeat_interleave(T)
            
        m_emb = self.member_emb(member_idx).unsqueeze(-1).unsqueeze(-1)
        h = x_feat + m_emb
        h = self.block(h)
        
        out = self.final_proj(h)
        out = F.pixel_shuffle(out, self.patch_size)
        
        out = self.padder.unpad(out)
        
        if is_5d:
            out = out.view(B, T, C, H, W)
            
        out = self.mass_corrector(out, x_ref)
        
        return out

