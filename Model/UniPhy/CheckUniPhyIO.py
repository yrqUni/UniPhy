import torch
import torch.nn as nn
from UniPhyIO import UniPhyEncoder, UniPhyDiffusionDecoder

def report(name, val, threshold=1e-5):
    passed = val < threshold
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"[{name}] Error/Metric: {val:.2e} -> {color}{status}{reset}")
    if not passed:
        raise ValueError(f"{name} Failed")

def check_geometric_conservation():
    print("\n--- Checking Geometric Conservation (Padding/Unpadding) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 2, 30, 721, 1440
    patch_size = 4
    latent_dim = 64
    
    encoder = UniPhyEncoder(in_ch=C, embed_dim=latent_dim, patch_size=patch_size).to(device)
    decoder = UniPhyDiffusionDecoder(out_ch=C, latent_dim=latent_dim, patch_size=patch_size).to(device)
    
    x = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device).float()
    
    with torch.no_grad():
        z = encoder(x)
        
        expected_h = (H + (patch_size - H % patch_size) % patch_size) // patch_size
        expected_w = (W + (patch_size - W % patch_size) % patch_size) // patch_size
        
        if z.shape[-2] != expected_h or z.shape[-1] != expected_w:
            print(f"Latent Shape: {z.shape}, Expected: ({expected_h}, {expected_w})")
            raise ValueError("Encoder Padding Logic Failed")
            
        x_pred = decoder(z, x, t)
        
    shape_diff = sum([abs(s1 - s2) for s1, s2 in zip(x_pred.shape, x.shape)])
    
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {x_pred.shape}")
    report("Shape Restoration", shape_diff, threshold=0.1)

def check_encoder_isometry():
    print("\n--- Checking Encoder Isometry (Information Preservation) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 2, 30, 64, 64
    encoder = UniPhyEncoder(in_ch=C, embed_dim=64).to(device)
    
    x1 = torch.randn(B, C, H, W, device=device)
    x2 = x1 * 2.0 
    
    with torch.no_grad():
        z1 = encoder(x1)
        z2 = encoder(x2)
        
    norm_x = torch.norm(x1)
    norm_z = torch.norm(z1)
    
    energy_ratio = norm_z / (norm_x + 1e-6)
    print(f"Energy Scale Ratio: {energy_ratio:.4f}")
    
    norm_z2 = torch.norm(z2)
    expected_norm_z2 = norm_z * 2.0
    
    linearity_error = abs(norm_z2 - expected_norm_z2) / (expected_norm_z2 + 1e-6)
    report("Linearity Conservation", linearity_error, threshold=1e-4)

def check_diffusion_pipeline():
    print("\n--- Checking Full IO Pipeline Gradient Flow ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 1, 4, 32, 32
    encoder = UniPhyEncoder(in_ch=C, embed_dim=16).to(device)
    decoder = UniPhyDiffusionDecoder(out_ch=C, latent_dim=16).to(device)
    
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    t = torch.ones(B, device=device)
    
    z = encoder(x)
    
    noise = torch.randn_like(x)
    pred_noise = decoder(z, x + noise, t)
    
    loss = pred_noise.sum()
    loss.backward()
    
    grad_exists = x.grad is not None and torch.norm(x.grad) > 0
    grad_val = torch.norm(x.grad).item() if grad_exists else 0.0
    
    report("Gradient Connectivity", 0.0 if grad_exists else 1.0, threshold=0.1)
    print(f"Input Gradient Norm: {grad_val:.6f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    try:
        check_geometric_conservation()
        check_encoder_isometry()
        check_diffusion_pipeline()
        print("\nAll IO Checks Passed.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")

