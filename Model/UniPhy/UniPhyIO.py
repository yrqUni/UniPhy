import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexiblePadder(nn.Module):
    def __init__(self, patch_size, mode='replicate'):
        super().__init__()
        self.patch_size = patch_size
        self.pad_h = 0
        self.pad_w = 0
        self.mode = mode

    def set_padding(self, h, w):
        self.pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        self.pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

    def pad(self, x):
        H, W = x.shape[-2:]
        self.set_padding(H, W)
        if self.pad_w > 0 or self.pad_h > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode=self.mode)
        return x

    def unpad(self, x):
        H, W = x.shape[-2:]
        target_h = H - self.pad_h
        target_w = W - self.pad_w
        return x[..., :target_h, :target_w]

class LearnableSphericalPosEmb(nn.Module):
    def __init__(self, dim, h, w, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2
        phi = torch.linspace(0, 2 * math.pi, w)
        theta = torch.linspace(0, math.pi, h)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')
        x_coord = torch.sin(theta_grid) * torch.cos(phi_grid)
        y_coord = torch.sin(theta_grid) * torch.sin(phi_grid)
        z_coord = torch.cos(theta_grid)
        coords = torch.stack([x_coord, y_coord, z_coord], dim=-1)
        self.register_buffer('coords', coords)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim) 
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, x):
        emb = self.mlp(self.coords)
        emb = emb.permute(2, 0, 1).unsqueeze(0)
        return x + emb

class GlobalConservationConstraint(nn.Module):
    def __init__(self, conserved_indices=[0], conserve_energy=True, epsilon=1e-6):
        super().__init__()
        self.conserved_indices = conserved_indices
        self.conserve_energy = conserve_energy
        self.epsilon = epsilon

    def forward(self, pred, ref):
        if ref.ndim == 5:
            ref_slice = ref[:, -1:, ...]
        else:
            ref_slice = ref.unsqueeze(1)
        B, T, C, H, W = pred.shape
        mask = torch.zeros(1, 1, C, 1, 1, device=pred.device, dtype=pred.dtype)
        for idx in self.conserved_indices:
            if idx < C:
                mask[:, :, idx, :, :] = 1.0
        ref_mean = ref_slice.mean(dim=(-2, -1), keepdim=True)
        pred_mean = pred.mean(dim=(-2, -1), keepdim=True)
        diff_mean = pred_mean - ref_mean
        pred_centered = pred - (diff_mean * mask)
        if self.conserve_energy:
            ref_var = torch.var(ref_slice, dim=(-2, -1), keepdim=True, unbiased=False)
            pred_var = torch.var(pred_centered, dim=(-2, -1), keepdim=True, unbiased=False)
            scale = torch.sqrt((ref_var + self.epsilon) / (pred_var + self.epsilon))
            ref_mean_target = ref_mean * mask + pred_mean * (1 - mask)
            pred_scaled = (pred_centered - ref_mean_target) * (scale * mask + (1 - mask)) + ref_mean_target
            return pred_scaled
        return pred_centered

class UniPhyEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=16, img_height=64, img_width=64):
        super().__init__()
        self.patch_size = patch_size
        self.padder = FlexiblePadder(patch_size, mode='replicate')
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
    def __init__(self, out_ch, latent_dim, patch_size=16, model_channels=128, ensemble_size=10, img_height=64):
        super().__init__()
        self.patch_size = patch_size
        self.padder = FlexiblePadder(patch_size, mode='replicate')
        self.ensemble_size = ensemble_size
        self.conservation_constraint = GlobalConservationConstraint(conserved_indices=[0])
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
            z_flat = z_latent.reshape(B * T, *z_latent.shape[2:])
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
            out = out.reshape(B, T, C, H, W)
        out = self.conservation_constraint(out, x_ref)
        return out
    