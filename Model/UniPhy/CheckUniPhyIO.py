import torch
import sys
import math
import torch.nn.functional as F

try:
    from UniPhyIO import Padder, MassCorrector, SphericalPosEmb, UniPhyEncoder, UniPhyEnsembleDecoder
except ImportError:
    from .UniPhyIO import Padder, MassCorrector, SphericalPosEmb, UniPhyEncoder, UniPhyEnsembleDecoder

def check_padder_odd_shapes():
    print("Checking Padder with Odd Shapes (H=721)...")
    patch_size = 16
    padder = Padder(patch_size)
    
    H, W = 721, 1440
    x = torch.randn(1, 2, H, W)
    
    padded_x = padder.pad(x)
    
    expected_h = math.ceil(H / patch_size) * patch_size
    expected_w = math.ceil(W / patch_size) * patch_size
    
    if padded_x.shape[-2:] != (expected_h, expected_w):
        print(f"FAIL: Padded shape mismatch. Got {padded_x.shape[-2:]}, expected {(expected_h, expected_w)}")
        sys.exit(1)
        
    unpadded_x = padder.unpad(padded_x)
    
    if unpadded_x.shape != x.shape:
        print(f"FAIL: Unpadded shape mismatch. Got {unpadded_x.shape}, expected {x.shape}")
        sys.exit(1)
    
    diff = (x - unpadded_x).abs().max().item()
    if diff == 0:
        print("PASS: Padder Reversibility verified for odd shapes.")
    else:
        print(f"FAIL: Data corruption during pad/unpad. Diff: {diff}")
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
    print("Checking Spherical Embedding Topology...")
    dim = 64
    H, W = 16, 32
    pos_emb_layer = SphericalPosEmb(dim, H, W)
    
    dummy = torch.zeros(1, dim, H, W)
    emb = pos_emb_layer(dummy)
    
    if pos_emb_layer.emb.shape == (1, dim, H, W):
        print("PASS: Spherical Embedding Shape Correct")
    else:
        print(f"FAIL: Embedding Shape Incorrect {pos_emb_layer.emb.shape}")
        sys.exit(1)

def check_encoder_decoder_integration():
    print("Checking Full Encoder/Decoder Integration with Odd Shapes...")
    B, T, C = 2, 4, 2
    H, W = 721, 1440
    dim = 64
    patch_size = 16
    
    enc = UniPhyEncoder(in_ch=C, embed_dim=dim, patch_size=patch_size, img_height=H, img_width=W)
    dec = UniPhyEnsembleDecoder(out_ch=C, latent_dim=dim, patch_size=patch_size, img_height=H)
    
    x = torch.randn(B, T, C, H, W)
    
    z = enc(x)
    
    expected_h_dim = math.ceil(H / patch_size)
    expected_w_dim = math.ceil(W / patch_size)
    
    if z.shape[-2:] != (expected_h_dim, expected_w_dim):
        print(f"FAIL: Encoder latent spatial dim mismatch. Got {z.shape[-2:]}, expected {(expected_h_dim, expected_w_dim)}")
        sys.exit(1)
        
    out = dec(z, x)
    
    if out.shape == x.shape:
        print(f"PASS: Decoder Output Shape Matches Input ({H}x{W})")
    else:
        print(f"FAIL: Shape Mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

if __name__ == "__main__":
    check_padder_odd_shapes()
    check_mass_conservation()
    check_spherical_topology_vorticity()
    check_encoder_decoder_integration()
    print("ALL UNI_PHY IO CHECKS PASSED")

