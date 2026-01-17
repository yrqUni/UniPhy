import torch
import sys
from UniPhyOps import MetricAwareCliffordConv2d, SymplecticPropagator, SpectralStep

def check_metric_aware_clifford_shapes():
    print("Checking MetricAwareCliffordConv2d Shapes...")
    B, C, H, W = 2, 16, 32, 32
    layer = MetricAwareCliffordConv2d(in_channels=C, out_channels=C, kernel_size=3, padding=1, img_height=H)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    if out.shape == x.shape:
        print("PASS: Shape match")
    else:
        print(f"FAIL: Shape mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

def check_symplectic_properties():
    print("Checking SymplecticPropagator Properties...")
    dim = 16
    dt = torch.ones(1) * 0.1
    prop = SymplecticPropagator(dim=dim, dt_ref=1.0)
    
    V, V_inv, evo_diag = prop.get_operators(dt)
    
    recon = torch.matmul(V, V_inv)
    eye = torch.eye(dim, device=dt.device)
    if V.is_complex():
        eye = eye.to(dtype=V.dtype)
        
    diff = (recon - eye).abs().max().item()
    
    if diff < 1e-4:
        print(f"PASS: Eigenbasis inversion verified. Error: {diff:.2e}")
    else:
        print(f"FAIL: Eigenbasis inversion error {diff:.2e}")
        sys.exit(1)

def check_spectral_mass_conservation():
    print("Checking SpectralStep Mass Conservation...")
    B, C, H, W = 2, 4, 32, 32
    layer = SpectralStep(dim=C, h=H, w=W, viscosity=1e-3)
    
    x = torch.randn(B, C, H, W, dtype=torch.cfloat)
    input_mass = x.mean(dim=(-1, -2))
    
    out = layer(x)
    output_mass = out.mean(dim=(-1, -2))
    
    diff = (input_mass - output_mass).abs().max().item()
    if diff < 1e-5:
        print(f"PASS: Mass drift {diff:.2e}")
    else:
        print(f"FAIL: Mass drift {diff:.2e}")
        sys.exit(1)

if __name__ == "__main__":
    check_metric_aware_clifford_shapes()
    check_symplectic_properties()
    check_spectral_mass_conservation()
    print("ALL CHECKS PASSED")

