import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pscanTriton import pscan

# =============================================================================
# Utils & Init
# =============================================================================

def _kaiming_like_(tensor):
    nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="relu")
    return tensor

def icnr_conv3d_weight_(weight, rH: int, rW: int):
    out_ch, in_ch, kD, kH, kW = weight.shape
    base_out = out_ch // (rH * rW)
    base = weight.new_zeros((base_out, in_ch, 1, kH, kW))
    _kaiming_like_(base)
    base = base.repeat_interleave(rH * rW, dim=0)
    with torch.no_grad():
        weight.copy_(base)
    return weight

class FlashFFTConvInterface(nn.Module):
    """
    Wrapper for FlashFFTConv. 
    Falls back to high-performance PyTorch FFT if flash-fft-conv is not installed.
    """
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.H, self.W = size
        
        # Determine padding for PyTorch implementation
        # Pad to power of 2 for generic FFT efficiency if not using Flash
        self.padded_H = 1 << (self.H - 1).bit_length()
        if self.padded_H < self.H: self.padded_H *= 2
        self.padded_W = 1 << (self.W - 1).bit_length()
        if self.padded_W < self.W: self.padded_W *= 2
        
        # Try importing real FlashFFTConv
        self.use_flash = False
        try:
            from flash_fft_conv import FlashFFTConv
            self.flash_conv = FlashFFTConv(size, dtype=torch.bfloat16) 
            self.use_flash = True
        except ImportError:
            pass

    def forward(self, u, k):
        # u: [B, C, H, W] (Signal)
        # k: [C, H, W] (Kernel - Frequency Domain or Spatial depending on impl)
        # Here we assume k comes in spatial, or we handle FFT internally
        
        if self.use_flash:
            # Placeholder: Real FlashFFTConv usage depends on specific library API
            # return self.flash_conv(u, k)
            pass

        # High-performance PyTorch fallback
        B, C, H, W = u.shape
        # Pad
        u_pad = F.pad(u, (0, self.padded_W - W, 0, self.padded_H - H))
        k_pad = F.pad(k, (0, self.padded_W - W, 0, self.padded_H - H))
        
        # FFT
        u_f = torch.fft.rfft2(u_pad.float())
        k_f = torch.fft.rfft2(k_pad.float())
        
        # Mul
        y_f = u_f * k_f
        
        # IFFT
        y = torch.fft.irfft2(y_f, s=(self.padded_H, self.padded_W))
        
        # Crop
        return y[..., :H, :W].type_as(u)

# =============================================================================
# Core Components (Attention, Priors, Gating)
# =============================================================================

class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        BL, C, H, W = x.shape
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)
        max_pool, _ = torch.max(x.reshape(BL, C, -1), dim=-1)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(BL, C, 1, 1)
        return x * attn

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn

class CBAM2DPerStep(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(spatial_kernel)

    def forward(self, x):
        B, C, L, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * L, C, H, W)
        x_flat = self.ca(x_flat)
        x_flat = self.sa(x_flat)
        x = x_flat.view(B, L, C, H, W).permute(0, 2, 1, 3, 4)
        return x

class GatedConvBlock(nn.Module):
    def __init__(self, channels, hidden_size, use_cbam=False, cond_channels=None):
        super().__init__()
        self.use_cbam = use_cbam
        self.dw_conv = nn.Conv3d(channels, channels, kernel_size=(1, 7, 7), padding="same", groups=channels)
        self.norm = nn.LayerNorm([*hidden_size])
        
        self.cond_channels = cond_channels
        if self.cond_channels is not None and self.cond_channels > 0:
            self.cond_proj = nn.Conv3d(self.cond_channels, channels * 2, kernel_size=1)
        else:
            self.cond_proj = None

        self.pw_conv_in = nn.Conv3d(channels, channels * 2, kernel_size=1)
        self.act = nn.SiLU()
        self.pw_conv_out = nn.Conv3d(channels, channels, kernel_size=1)
        
        if self.use_cbam:
            self.cbam = CBAM2DPerStep(channels, reduction=16)

    def forward(self, x, cond=None):
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4)
        
        if self.cond_proj is not None and cond is not None:
            cond_in = cond.unsqueeze(2) if cond.dim() == 4 else cond
            affine = self.cond_proj(cond_in)
            gamma, beta = torch.chunk(affine, 2, dim=1)
            x = x * (1 + gamma) + beta

        x = self.pw_conv_in(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.act(x) * gate
        
        if self.use_cbam:
            x = self.cbam(x)
            
        x = self.pw_conv_out(x)
        return residual + x

# =============================================================================
# ConvLRU Layer (Core: Selective, Bidirectional, FlashFFT)
# =============================================================================

class ConvLRULayer(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        self.use_bias = True
        self.r_min = 0.8
        self.r_max = 0.99
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        S, W = self.hidden_size
        self.rank = int(getattr(args, "lru_rank", min(S, W, 32)))
        
        # A. Selective Mechanism (Mamba-style)
        self.is_selective = getattr(args, "use_selective", False)
        
        # B. Bidirectional Scan
        self.bidirectional = getattr(args, "bidirectional", False)
        
        # C. FlashFFTConv (Interface)
        self.flash_fft = FlashFFTConvInterface(self.emb_ch, (S, W))
        
        # Parameter Initialization
        u1 = torch.rand(self.emb_ch, self.rank)
        u2 = torch.rand(self.emb_ch, self.rank)
        nu_log = torch.log(-0.5 * torch.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        theta_log = torch.log(u2 * (2 * torch.tensor(np.pi)))
        self.params_log_base = nn.Parameter(torch.stack([nu_log, theta_log], dim=0))
        self.dispersion_mod = nn.Parameter(torch.zeros(2, self.emb_ch, self.rank) * 0.01)
        
        # Forcing / Selective Projection
        self.mod_hidden = 32
        # If Selective, input -> dt, nu, theta
        in_dim = self.emb_ch if self.is_selective else (self.emb_ch + 1)
        self.forcing_mlp = nn.Sequential(
            nn.Linear(in_dim, self.mod_hidden),
            nn.Tanh(),
            nn.Linear(self.mod_hidden, self.emb_ch * self.rank * 2)
        )
        self.forcing_scale = nn.Parameter(torch.tensor(0.1))

        # Projections
        self.U_row = nn.Parameter(torch.randn(self.emb_ch, S, self.rank, dtype=torch.cfloat) / math.sqrt(S))
        self.V_col = nn.Parameter(torch.randn(self.emb_ch, W, self.rank, dtype=torch.cfloat) / math.sqrt(W))
        C = self.emb_ch
        self.proj_W = nn.Parameter(torch.randn(C, C, dtype=torch.cfloat) / math.sqrt(C))
        self.proj_b = nn.Parameter(torch.zeros(C, dtype=torch.cfloat)) if self.use_bias else None
        
        self.post_ifft_conv_real = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.post_ifft_conv_imag = nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        out_dim_fusion = self.emb_ch * 2 if not self.bidirectional else self.emb_ch * 4
        self.post_ifft_proj = nn.Conv3d(out_dim_fusion, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.layer_norm = nn.LayerNorm([*self.hidden_size])
        
        self.noise_level = nn.Parameter(torch.tensor(0.01))
        if getattr(args, "use_gate", False):
            self.gate_conv = nn.Sequential(
                nn.Conv3d(self.emb_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same"),
                nn.Sigmoid()
            )
        self.pscan = pscan

    def _apply_forcing(self, x, dt):
        # x: [B, L, C, S, W]
        ctx = x.mean(dim=(-2, -1)) # [B, L, C]
        
        if self.is_selective:
            # Mamba-style: Dynamics depend on Input Content
            inp = ctx 
        else:
            # Standard: Dynamics depend on Context + TimeDelta
            dt_feat = dt.view(x.size(0), x.size(1), 1)
            inp = torch.cat([ctx, dt_feat], dim=-1)
            
        mod = self.forcing_mlp(inp)
        mod = mod.view(x.size(0), x.size(1), self.emb_ch, self.rank, 2)
        dnu = self.forcing_scale * torch.tanh(mod[..., 0])
        dth = self.forcing_scale * torch.tanh(mod[..., 1])
        return dnu.unsqueeze(-1), dth.unsqueeze(-1)

    def _fft_impl(self, x):
        # Using FlashFFTConv logic (currently fallback)
        B, L, C, S, W = x.shape
        pad_size = S // 4
        x_reshaped = x.view(B * L, C, S, W)
        x_pad = F.pad(x_reshaped, (0, 0, pad_size, pad_size), mode='reflect')
        x_pad = x_pad.view(B, L, C, S + 2 * pad_size, W)
        h = torch.fft.fft2(x_pad.to(torch.cfloat), dim=(-2, -1), norm="ortho")
        return h, pad_size

    def forward(self, x, last_hidden_in, listT=None):
        B, L, C, S, W = x.size()
        if listT is None:
            dt = torch.ones(B, L, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            dt = listT.view(B, L, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        
        h, pad_size = self._fft_impl(x)
        S_pad = h.shape[-2]
        
        h_perm = h.permute(0, 1, 3, 4, 2).contiguous().view(B * L * S_pad * W, C)
        h_proj = torch.matmul(h_perm, self.proj_W)
        h = h_proj.view(B, L, S_pad, W, C).permute(0, 1, 4, 2, 3).contiguous()
        if self.proj_b is not None:
            h = h + self.proj_b.view(1, 1, C, 1, 1)
            
        h_spatial = torch.fft.ifft2(h, dim=(-2, -1), norm="ortho")
        h_spatial = h_spatial[..., pad_size:-pad_size, :]
        h = torch.fft.fft2(h_spatial, dim=(-2, -1), norm="ortho")
        
        Uc = self.U_row.conj()
        t = torch.matmul(h.permute(0, 1, 2, 4, 3), Uc)
        t = t.permute(0, 1, 2, 4, 3)
        zq = torch.matmul(t, self.V_col)
        
        nu_log, theta_log = self.params_log_base.unbind(dim=0)
        disp_nu, disp_th = self.dispersion_mod.unbind(dim=0)
        nu_base = torch.exp(nu_log + disp_nu).view(1, 1, C, self.rank, 1)
        th_base = torch.exp(theta_log + disp_th).view(1, 1, C, self.rank, 1)
        
        dnu_force, dth_force = self._apply_forcing(x, dt)
        
        nu_t = torch.clamp(nu_base * dt + dnu_force, min=1e-6)
        th_t = th_base * dt + dth_force
        lamb = torch.exp(torch.complex(-nu_t, th_t))
        
        if self.training:
            noise_std = self.noise_level * torch.sqrt(dt + 1e-6)
            noise = torch.randn_like(zq) * noise_std
            x_in = zq + noise
        else:
            x_in = zq
            
        gamma_t = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * nu_t.real), min=1e-12))
        x_in = x_in * gamma_t
        
        # --- Recurrence ---
        
        # 1. Forward Scan
        # Handle state passing
        if last_hidden_in is not None:
            prev_state = last_hidden_in
            x_in_fwd = torch.concat([prev_state, x_in], dim=1)
            lamb_fwd = torch.concat([lamb[:, :1], lamb], dim=1)
        else:
            if L == 1:
                zero_prev = torch.zeros_like(x_in[:, :1])
                x_in_fwd = torch.cat([zero_prev, x_in], dim=1)
                lamb_fwd = torch.cat([lamb[:, :1], lamb], dim=1)
            else:
                x_in_fwd = x_in
                lamb_fwd = lamb
                
        L_eff = x_in_fwd.size(1)
        lamb_in_fwd = lamb_fwd[:, :L_eff].expand_as(x_in_fwd)
        z_out = self.pscan(lamb_in_fwd.contiguous(), x_in_fwd.contiguous())
        
        if last_hidden_in is not None or (last_hidden_in is None and L == 1):
            z_out = z_out[:, 1:]
        
        last_hidden_out = z_out[:, -1:]
        
        # 2. Bidirectional Scan (Backward)
        if self.bidirectional:
            # Flip time dimension for inputs
            x_in_bwd = x_in.flip(1)
            lamb_bwd = lamb.flip(1) # Decay for backward pass
            
            # Usually backward pass starts with 0 state or learns a backward state
            # Here we assume 0 init for simplicity for backward stream
            if L == 1:
                 x_in_bwd = torch.cat([torch.zeros_like(x_in_bwd[:,:1]), x_in_bwd], dim=1)
                 lamb_bwd = torch.cat([lamb_bwd[:,:1], lamb_bwd], dim=1)
            
            L_eff_b = x_in_bwd.size(1)
            lamb_in_bwd = lamb_bwd[:, :L_eff_b].expand_as(x_in_bwd)
            z_out_bwd = self.pscan(lamb_in_bwd.contiguous(), x_in_bwd.contiguous())
            
            if L == 1:
                z_out_bwd = z_out_bwd[:, 1:]
            
            # Flip back to align with forward time
            z_out_bwd = z_out_bwd.flip(1)
            
            # Fuse: We will concat in channel dim later
        
        # Projection Back
        def project_back(z):
            t = torch.matmul(z, self.V_col.conj().transpose(1, 2))
            t = t.permute(0, 1, 2, 4, 3)
            return torch.matmul(t, self.U_row.transpose(1, 2)).permute(0, 1, 2, 4, 3)
        
        h_rec_fwd = project_back(z_out)
        
        # IFFT Recovery
        def recover_spatial(h_rec):
            h_sp = torch.fft.ifft2(h_rec, dim=(-2, -1), norm="ortho")
            hr = self.post_ifft_conv_real(h_sp.real.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            hi = self.post_ifft_conv_imag(h_sp.imag.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            return torch.cat([hr, hi], dim=2)

        feat_fwd = recover_spatial(h_rec_fwd)
        
        if self.bidirectional:
            h_rec_bwd = project_back(z_out_bwd)
            feat_bwd = recover_spatial(h_rec_bwd)
            feat_final = torch.cat([feat_fwd, feat_bwd], dim=2) # [B, L, 4*C, H, W]
        else:
            feat_final = feat_fwd # [B, L, 2*C, H, W]
            
        h_final = self.post_ifft_proj(feat_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        h_final = self.layer_norm(h_final)
        
        if hasattr(self, 'gate_conv'):
            gate = self.gate_conv(h_final.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            x = (1 - gate) * x + gate * h_final
        else:
            x = x + h_final
            
        return x, last_hidden_out

# =============================================================================
# High-Level Architecture (U-Net & Heads)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.ffn_hidden_ch = getattr(args, "ffn_hidden_ch", 32)
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.layers_num = getattr(args, "ffn_hidden_layers_num", 1)
        self.use_cbam = bool(getattr(args, "use_cbam", False))
        self.static_ch = int(getattr(args, "static_ch", 0))
        
        self.c_in = nn.Conv3d(self.emb_ch, self.ffn_hidden_ch, kernel_size=(1, 1, 1), padding="same")
        self.blocks = nn.ModuleList([
            GatedConvBlock(self.ffn_hidden_ch, self.hidden_size, use_cbam=self.use_cbam, cond_channels=self.emb_ch if self.static_ch > 0 else None)
            for _ in range(self.layers_num)
        ])
        self.c_out = nn.Conv3d(self.ffn_hidden_ch, self.emb_ch, kernel_size=(1, 1, 1), padding="same")
        self.act = nn.SiLU()

    def forward(self, x, cond=None):
        residual = x
        x = self.c_in(x.permute(0, 2, 1, 3, 4))
        x = self.act(x)
        for block in self.blocks:
            x = block(x, cond=cond)
        x = self.c_out(x)
        x = x.permute(0, 2, 1, 3, 4)
        return residual + x

class ConvLRUBlock(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.lru_layer = ConvLRULayer(args, input_downsp_shape)
        self.feed_forward = FeedForward(args, input_downsp_shape)

    def forward(self, x, last_hidden_in, listT=None, cond=None):
        x, last_hidden_out = self.lru_layer(x, last_hidden_in, listT=listT)
        x = self.feed_forward(x, cond=cond)
        return x, last_hidden_out

class VectorQuantizer(nn.Module):
    """ Discrete Token Modeling Head """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, L, H, W] -> flatten to [N, C]
        B, C, L, H, W = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        
        # Distances
        dist = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(B, L, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

class DiffusionHead(nn.Module):
    """ Simple Time-Embedding for Diffusion Head """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # Sinusoidal embedding
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class Decoder(nn.Module):
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        self.head_mode = getattr(args, "head_mode", "gaussian") # 'gaussian', 'diffusion', 'token'
        
        # Upsampling Logic (Shared)
        self.output_ch = getattr(args, "out_ch", 1)
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.dec_hidden_ch = getattr(args, "dec_hidden_ch", 0)
        self.dec_hidden_layers_num = getattr(args, "dec_hidden_layers_num", 0)
        self.static_ch = int(getattr(args, "static_ch", 0))
        self.hidden_size = [input_downsp_shape[1], input_downsp_shape[2]]
        self.dec_strategy = getattr(args, "dec_strategy", "pxsf")
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        if self.dec_hidden_layers_num != 0:
            self.dec_hidden_ch = getattr(args, "dec_hidden_ch", self.emb_ch)
        out_ch_after_up = self.dec_hidden_ch if self.dec_hidden_layers_num != 0 else self.emb_ch
        
        if self.dec_strategy == "deconv":
            self.upsp = nn.ConvTranspose3d(self.emb_ch, out_ch_after_up, kernel_size=(1, self.rH, self.rW), stride=(1, self.rH, self.rW))
            deconv3d_bilinear_init_(self.upsp.weight)
        else:
            self.pre_shuffle_conv = nn.Conv3d(self.emb_ch, out_ch_after_up * self.rH * self.rW, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            icnr_conv3d_weight_(self.pre_shuffle_conv.weight, self.rH, self.rW)
            
        if self.dec_hidden_layers_num != 0:
            H = self.hidden_size[0] * self.rH
            W = self.hidden_size[1] * self.rW
            self.c_hidden = nn.ModuleList([
                GatedConvBlock(out_ch_after_up, (H, W), use_cbam=False, cond_channels=self.emb_ch if self.static_ch > 0 else None)
                for _ in range(self.dec_hidden_layers_num)
            ])
        else:
            self.c_hidden = None
            
        # Specific Heads
        if self.head_mode == "gaussian":
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch * 2, kernel_size=(1, 1, 1), padding="same")
        elif self.head_mode == "diffusion":
            # Diffusion predicts noise: [B, Out, L, H, W]
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.time_embed = DiffusionHead(out_ch_after_up)
        elif self.head_mode == "token":
            # VQ-VAE Style
            self.c_out = nn.Conv3d(out_ch_after_up, self.output_ch, kernel_size=(1, 1, 1), padding="same")
            self.vq = VectorQuantizer(num_embeddings=1024, embedding_dim=self.output_ch)
        
        self.activation = nn.SiLU()

    def forward(self, x, cond=None, timestep=None):
        # x: [B, L, C, H, W] -> [B, C, L, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        if self.dec_strategy == "deconv":
            x = self.upsp(x)
        else:
            x = self.pre_shuffle_conv(x)
            x = pixel_shuffle_hw_3d(x, self.rH, self.rW)
        x = self.activation(x)
        
        if self.head_mode == "diffusion" and timestep is not None:
            # Inject time embedding
            t_emb = self.time_embed(timestep) # [B, C]
            x = x + t_emb.view(x.size(0), x.size(1), 1, 1, 1)

        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=None)
                
        x = self.c_out(x) # [B, C_out, L, H, W]
        
        if self.head_mode == "gaussian":
            mu, log_sigma = torch.chunk(x, 2, dim=1)
            sigma = F.softplus(log_sigma) + 1e-6
            return torch.cat([mu, sigma], dim=1).permute(0, 2, 1, 3, 4)
            
        elif self.head_mode == "token":
            # Quantize
            quantized, loss, indices = self.vq(x)
            return quantized.permute(0, 2, 1, 3, 4), loss, indices
            
        else: # diffusion or standard regression
            return x.permute(0, 2, 1, 3, 4)

class ConvLRUModel(nn.Module):
    """ U-Net Structure enabled by args.unet = True """
    def __init__(self, args, input_downsp_shape):
        super().__init__()
        self.args = args
        self.use_unet = getattr(args, "unet", False)
        layers = getattr(args, "convlru_num_blocks", 2)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        C = args.emb_ch
        H, W = input_downsp_shape[1], input_downsp_shape[2]
        
        if not self.use_unet:
            # Sequential Stack (Original)
            self.convlru_blocks = nn.ModuleList([ConvLRUBlock(self.args, (C, H, W)) for _ in range(layers)])
        else:
            # U-Net Encoder
            curr_H, curr_W = H, W
            for i in range(layers):
                self.down_blocks.append(ConvLRUBlock(self.args, (C, curr_H, curr_W)))
                # Simple strided conv for downsample
                if i < layers - 1:
                    self.skip_convs.append(nn.Identity()) # Store feature for skip
                    # Note: Real downsampling should change C, here we keep C constant for simplicity
                    # or you can increase C. Let's keep C constant to match Embedding output.
                    curr_H //= 2
                    curr_W //= 2
            
            # U-Net Decoder
            for i in range(layers - 1):
                self.up_blocks.append(ConvLRUBlock(self.args, (C, curr_H, curr_W)))
                curr_H *= 2
                curr_W *= 2
            
            # Bottom layer is the last of down_blocks
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            self.fusion = nn.Conv3d(C*2, C, 1)

    def forward(self, x, last_hidden_ins=None, listT=None, cond=None):
        # x: [B, L, C, H, W]
        if not self.use_unet:
            # Original Sequential
            last_hidden_outs = []
            for idx, convlru_block in enumerate(self.convlru_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None) else None
                x, last_hidden_out = convlru_block(x, h_in, listT=listT, cond=cond)
                last_hidden_outs.append(last_hidden_out)
            return x, last_hidden_outs
        
        else:
            # U-Net Forward
            skips = []
            last_hidden_outs = []
            
            # Encoder
            for idx, block in enumerate(self.down_blocks):
                h_in = last_hidden_ins[idx] if (last_hidden_ins is not None) else None
                x, last_hidden_out = block(x, h_in, listT=listT, cond=cond)
                last_hidden_outs.append(last_hidden_out)
                
                if idx < len(self.down_blocks) - 1:
                    skips.append(x)
                    # Downsample spatial only [B, L, C, H, W] -> permute -> [B, C, L, H, W]
                    x_s = x.permute(0, 2, 1, 3, 4)
                    x_s = F.avg_pool3d(x_s, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                    x = x_s.permute(0, 2, 1, 3, 4)

            # Decoder
            # Reverse iterate up blocks
            # Note: last_hidden_ins logic for UNet is complex for streaming, 
            # here we assume simpler hidden passing or just return bottom layer hidden
            
            for idx, block in enumerate(self.up_blocks):
                # Upsample
                x_s = x.permute(0, 2, 1, 3, 4)
                x_s = self.upsample(x_s)
                x = x_s.permute(0, 2, 1, 3, 4)
                
                # Concat Skip
                skip = skips.pop()
                x = torch.cat([x, skip], dim=2) # Concat Channel
                x = self.fusion(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                
                # ConvLRU Refine
                # We skip hidden state passing for decoder layers in this simplified impl
                x, _ = block(x, None, listT=listT, cond=cond)

            return x, last_hidden_outs

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_ch = getattr(args, "input_ch", 1)
        self.input_size = getattr(args, "input_size", (64, 64))
        self.emb_ch = getattr(args, "emb_ch", 32)
        self.emb_hidden_ch = getattr(args, "emb_hidden_ch", 32)
        self.emb_hidden_layers_num = getattr(args, "emb_hidden_layers_num", 0)
        self.static_ch = int(getattr(args, "static_ch", 0))
        
        hf = getattr(args, "hidden_factor", (2, 2))
        self.rH, self.rW = int(hf[0]), int(hf[1])
        
        self.patch_embed = nn.Conv3d(
            self.input_ch,
            self.emb_hidden_ch,
            kernel_size=(1, self.rH + 2, self.rW + 2),
            stride=(1, self.rH, self.rW),
            padding=(0, 1, 1)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_ch, 1, *self.input_size)
            out_dummy = self.patch_embed(dummy)
            _, C_hidden, _, H, W = out_dummy.shape
            self.input_downsp_shape = (self.emb_ch, H, W)
        
        self.hidden_size = (H, W)
        
        if self.static_ch > 0:
            self.static_embed = nn.Sequential(
                nn.Conv2d(self.static_ch, self.emb_ch, kernel_size=(self.rH+2, self.rW+2), stride=(self.rH, self.rW), padding=(1, 1)),
                nn.SiLU()
            )
            
        if self.emb_hidden_layers_num > 0:
            self.c_hidden = nn.ModuleList([
                GatedConvBlock(self.emb_hidden_ch, self.hidden_size, use_cbam=False, cond_channels=self.emb_ch if self.static_ch > 0 else None)
                for _ in range(self.emb_hidden_layers_num)
            ])
        else:
            self.c_hidden = None
            
        self.c_out = nn.Conv3d(self.emb_hidden_ch, self.emb_ch, kernel_size=1)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm([*self.hidden_size])

    def forward(self, x, static_feats=None):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x)
        x = self.activation(x)
        
        cond = None
        if self.static_ch > 0 and static_feats is not None:
            cond = self.static_embed(static_feats)
        
        if self.c_hidden is not None:
            for layer in self.c_hidden:
                x = layer(x, cond=cond)
                
        x = self.c_out(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.layer_norm(x)
        return x, cond

class ConvLRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(self.args)
        self.convlru_model = ConvLRUModel(self.args, self.embedding.input_downsp_shape)
        self.decoder = Decoder(self.args, self.embedding.input_downsp_shape)
        
        skip_contains = ["layer_norm", "params_log", "prior", "post_ifft", "spatial_mod", "forcing", "dispersion"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                if any(tok in n for tok in skip_contains): continue
                if n.endswith(".bias"): p.zero_(); continue
                if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x, mode="p", out_gen_num=None, listT=None, listT_future=None, static_feats=None, timestep=None):
        cond = None
        if self.embedding.static_ch > 0 and static_feats is not None:
            cond = self.embedding.static_embed(static_feats)

        if mode == "p":
            x, _ = self.embedding(x, static_feats=static_feats)
            x, _ = self.convlru_model(x, listT=listT, cond=cond)
            # Decoder handles Diffusion/Gaussian/Token switch
            return self.decoder(x, cond=cond, timestep=timestep)

        # Autoregressive / Streaming Mode
        out = []
        x_emb, _ = self.embedding(x, static_feats=None)
        x_hidden, last_hidden_outs = self.convlru_model(x_emb, listT=listT, cond=cond)
        x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
        
        # Handle different output formats
        if isinstance(x_dec, tuple): # Token mode
            x_dec = x_dec[0]
            
        x_step_dist = x_dec[:, -1:]
        
        if self.decoder.head_mode == "gaussian":
            x_step_mean = x_step_dist[..., :self.args.out_ch, :, :]
        else:
            x_step_mean = x_step_dist # Raw pred for diffusion/token
            
        out.append(x_step_dist)
        
        for t in range(out_gen_num - 1):
            dt = listT_future[:, t:t+1] if listT_future is not None else torch.ones_like(listT[:, 0:1])
            x_in, _ = self.embedding(x_step_mean, static_feats=None)
            x_hidden, last_hidden_outs = self.convlru_model(x_in, last_hidden_ins=last_hidden_outs, listT=dt, cond=cond)
            x_dec = self.decoder(x_hidden, cond=cond, timestep=timestep)
            
            if isinstance(x_dec, tuple): x_dec = x_dec[0]
            
            x_step_dist = x_dec[:, -1:]
            if self.decoder.head_mode == "gaussian":
                x_step_mean = x_step_dist[..., :self.args.out_ch, :, :]
            else:
                x_step_mean = x_step_dist
                
            out.append(x_step_dist)
            
        return torch.concat(out, dim=1)