import torch
import sys
import math

from UniPhyOps import MetricAwareCliffordConv2d, SymplecticPropagator, SpectralStep

def check_metric_aware_clifford():
    print("Checking MetricAwareCliffordConv2d...")
    B, C, H, W = 2, 16, 32, 32
    layer = MetricAwareCliffordConv2d(in_channels=C, out_channels=C, kernel_size=3, padding=1, img_height=H)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    if out.shape == x.shape:
        print("PASS: Shape match")
    else:
        print(f"FAIL: Shape mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

def check_symplectic_cayley():
    print("Checking SymplecticPropagator (Cayley Transform)...")
    dim = 16
    prop = SymplecticPropagator(dim=dim, dt_ref=1.0, stochastic=False)
    
    Q = prop.get_orthogonal_basis()
    I = torch.eye(dim, device=Q.device)
    QTQ = torch.matmul(Q.T, Q)
    
    diff = (QTQ - I).abs().max().item()
    if diff < 1e-4:
        print(f"PASS: Orthogonality verified via Cayley. Error: {diff:.2e}")
    else:
        print(f"FAIL: Orthogonality broken. Error: {diff:.2e}")
        sys.exit(1)

def check_noise_injector():
    print("Checking Neural SDE Noise Injector...")
    dim = 16
    prop = SymplecticPropagator(dim=dim, stochastic=True)
    
    B, T, D, H, W = 2, 5, dim, 16, 16
    x = torch.randn(B, T, D, H, W)
    dt = torch.ones(B, T) * 0.1
    
    noise1 = prop.inject_noise(x, dt)
    noise2 = prop.inject_noise(x, dt)
    
    diff = (noise1 - noise2).abs().mean().item()
    if diff > 0:
        print(f"PASS: Stochasticity verified. Diff: {diff:.2e}")
    else:
        print("FAIL: Output is deterministic.")
        sys.exit(1)

    dt_zero = torch.zeros(B, T)
    noise_zero = prop.inject_noise(x, dt_zero)
    if noise_zero.abs().max().item() < 1e-6:
        print("PASS: Wiener process scaling verified.")
    else:
        print("FAIL: Noise not zero at dt=0.")
        sys.exit(1)

def check_spectral_step():
    print("Checking SpectralStep...")
    B, C, H, W = 2, 4, 32, 32
    layer = SpectralStep(dim=C, h=H, w=W, viscosity=1e-3)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    if out.shape == x.shape:
        print("PASS: SpectralStep shape match")
    else:
        print("FAIL: SpectralStep shape mismatch")
        sys.exit(1)

if __name__ == "__main__":
    check_metric_aware_clifford()
    check_symplectic_cayley()
    check_noise_injector()
    check_spectral_step()
    print("ALL OPS CHECKS PASSED")

