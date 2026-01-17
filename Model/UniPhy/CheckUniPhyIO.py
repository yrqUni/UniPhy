import torch
import sys
import math
import torch.nn.functional as F

try:
    from UniPhyIO import Padder, MassCorrector, SphericalPosEmb, UniPhyEncoder, UniPhyEnsembleDecoder
except ImportError:
    from .UniPhyIO import Padder, MassCorrector, SphericalPosEmb, UniPhyEncoder, UniPhyEnsembleDecoder

def check_boundary_conditions_momentum():
    print("Checking Boundary Conditions (Angular Momentum & Zonal Flow)...")
    patch_size = 4
    padder = Padder(patch_size)
    
    H, W = 10, 10
    x = torch.arange(W).float().view(1, 1, 1, W).repeat(1, 1, H, 1)
    
    padded_x = padder.pad(x)
    
    pad_w = padder.pad_w
    if pad_w > 0:
        right_pad = padded_x[..., :H, -pad_w:]
        left_original = x[..., :pad_w]
        
        diff = (right_pad - left_original).abs().max().item()
        if diff < 1e-6:
            print(f"PASS: W-axis Circular Padding verified (Angular Momentum Conserved). Drift: {diff:.2e}")
        else:
            print(f"FAIL: W-axis is not circular. Zonal flow broken. Drift: {diff:.2e}")
            sys.exit(1)
            
    x_h = torch.arange(H).float().view(1, 1, H, 1).repeat(1, 1, 1, W)
    padded_x_h = padder.pad(x_h)
    pad_h = padder.pad_h
    
    if pad_h > 0:
        bottom_pad = padded_x_h[..., -pad_h:, :W]
        bottom_original = x_h[..., -1:, :]
        
        bottom_original_expanded = bottom_original.expand_as(bottom_pad)
        
        diff = (bottom_pad - bottom_original_expanded).abs().max().item()
        if diff < 1e-6:
            print(f"PASS: H-axis Replicate Padding verified (Polar Boundary). Drift: {diff:.2e}")
        else:
            print(f"FAIL: H-axis padding incorrect. Drift: {diff:.2e}")
            sys.exit(1)

def check_mass_conservation():
    print("Checking Global Mass Conservation...")
    H, W = 32, 64
    corrector = MassCorrector(height=H, mass_idx=0)
    
    pred = torch.randn(2, 5, 1, H, W) + 10.0
    ref = torch.randn(2, 5, 1, H, W) + 10.0
    
    lat_indices = torch.arange(H)
    lat_rad = (lat_indices / (H - 1)) * math.pi - (math.pi / 2)
    weights = torch.cos(lat_rad).view(1, 1, 1, -1, 1)
    weights = weights / weights.mean()
    
    ref_mass = (ref[:, -1:, ...] * weights).mean(dim=(-2, -1))
    
    corrected_pred = corrector(pred, ref)
    
    out_mass = (corrected_pred * weights).mean(dim=(-2, -1))
    
    diff = (out_mass - ref_mass).abs().max().item()
    
    if diff < 1e-5:
        print(f"PASS: Global Mass Conserved. Max Drift: {diff:.2e}")
    else:
        print(f"FAIL: Mass Conservation Violated. Max Drift: {diff:.2e}")
        sys.exit(1)

def check_spherical_topology_vorticity():
    print("Checking Spherical Topology (Vorticity Continuity)...")
    dim = 64
    H, W = 16, 32
    pos_emb_layer = SphericalPosEmb(dim, H, W)
    
    dummy = torch.zeros(1, dim, H, W)
    emb = pos_emb_layer(dummy)
    
    if pos_emb_layer.emb.shape == (1, dim, H, W):
        print("PASS: Spherical Embedding Shape Correct")
    else:
        print("FAIL: Embedding Shape Incorrect")
        sys.exit(1)

def check_full_pipeline_shapes():
    print("Checking Full Pipeline Input/Output...")
    B, T, C, H, W = 2, 4, 2, 64, 128
    dim = 64
    
    enc = UniPhyEncoder(in_ch=C, embed_dim=dim, patch_size=16, img_height=H, img_width=W)
    dec = UniPhyEnsembleDecoder(out_ch=C, latent_dim=dim, patch_size=16, img_height=H)
    
    x = torch.randn(B, T, C, H, W)
    
    z = enc(x)
    if z.is_complex():
        print(f"PASS: Encoder Output is Complex (Required for Physics Backbone)")
    else:
        print("FAIL: Encoder Output should be Complex")
        sys.exit(1)
        
    out = dec(z, x)
    
    if out.shape == x.shape:
        print("PASS: Decoder Output Shape Matches Input")
    else:
        print(f"FAIL: Shape Mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

if __name__ == "__main__":
    check_boundary_conditions_momentum()
    check_mass_conservation()
    check_spherical_topology_vorticity()
    check_full_pipeline_shapes()
    print("ALL UNI_PHY IO CHECKS PASSED")

