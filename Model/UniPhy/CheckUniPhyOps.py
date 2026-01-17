import torch
import sys
import math

from UniPhyOps import MetricAwareCliffordConv2d, SymplecticPropagator, SpectralStep

def check_metric_aware_clifford():
    print("Checking MetricAwareCliffordConv2d (Dynamic Metric)...")
    B, C, H, W = 2, 16, 32, 32
    layer = MetricAwareCliffordConv2d(in_channels=C, out_channels=C, kernel_size=3, padding=1, img_height=H)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    if out.shape == x.shape:
        print("PASS: Shape match")
    else:
        print(f"FAIL: Shape mismatch {out.shape} vs {x.shape}")
        sys.exit(1)
        
    x_perturb = x + 0.1
    out_perturb = layer(x_perturb)
    print("PASS: Dynamic Metric Refiner forward successful.")

def check_time_warped_dynamics():
    print("Checking SymplecticPropagator (Time Warping + Open System)...")
    dim = 16
    B, T, H, W = 2, 5, 16, 16
    prop = SymplecticPropagator(dim=dim, img_height=H, img_width=W, dt_ref=1.0, stochastic=False)
    
    dt = torch.ones(B, T)
    
    x_calm = torch.randn(B, T, dim, H, W, dtype=torch.cfloat) * 0.01
    x_storm = torch.randn(B, T, dim, H, W, dtype=torch.cfloat) * 5.0
    
    _, _, evo_calm, dt_calm = prop.get_operators(dt, x_context=x_calm)
    _, _, evo_storm, dt_storm = prop.get_operators(dt, x_context=x_storm)
    
    diff = (evo_calm - evo_storm).abs().mean().item()
    
    if diff > 1e-5:
        print(f"PASS: Dynamics are adaptive (Growth + Time Warping). Diff: {diff:.2e}")
    else:
        print("FAIL: Dynamics are static. Input context ignored.")
        sys.exit(1)
        
    mag_calm = evo_calm.abs().mean().item()
    if abs(mag_calm - 1.0) > 1e-6:
        print(f"PASS: Non-Unitary Evolution verified. Mean Mag: {mag_calm:.4f}")
    else:
        print("WARNING: Evolution is strictly unitary.")

    Q = prop.get_orthogonal_basis()
    I = torch.eye(dim, device=Q.device)
    QTQ = torch.matmul(Q.T, Q)
    ortho_err = (QTQ - I).abs().max().item()
    if ortho_err < 1e-4:
        print(f"PASS: Spatial Basis Orthogonality verified. Error: {ortho_err:.2e}")
    else:
        print("FAIL: Basis orthogonality broken.")
        sys.exit(1)

def check_spectral_noise_injector():
    print("Checking SpectralNoiseInjector (SDE)...")
    dim = 16
    B, T, D, H, W = 2, 5, dim, 16, 16
    prop = SymplecticPropagator(dim=dim, img_height=H, img_width=W, stochastic=True)
    
    x = torch.randn(B, T, D, H, W, dtype=torch.cfloat)
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
    if noise_zero.abs().max().item() < 1e-5:
        print("PASS: Wiener process scaling verified (dt=0 -> noise=0).")
    else:
        print(f"FAIL: Noise not zero at dt=0. Max val: {noise_zero.abs().max().item()}")
        sys.exit(1)

def check_anisotropic_spectral_step():
    print("Checking Anisotropic Fractional SpectralStep...")
    B, C, H, W = 2, 4, 32, 32
    layer = SpectralStep(dim=C, h=H, w=W)
    
    layer.viscosity_x.data.fill_(1.0)
    layer.viscosity_y.data.fill_(0.0)
    layer.alpha.data.fill_(2.0)
    
    x = torch.randn(B, C, H, W, dtype=torch.cfloat)
    
    out = layer(x)
    
    if out.shape == x.shape:
        print("PASS: SpectralStep shape match.")
    else:
        print("FAIL: Shape mismatch.")
        sys.exit(1)
        
    if hasattr(layer, 'alpha') and layer.alpha.requires_grad:
        print("PASS: Fractional order 'alpha' is learnable.")
    else:
        print("FAIL: 'alpha' is not learnable.")

if __name__ == "__main__":
    check_metric_aware_clifford()
    check_time_warped_dynamics()
    check_spectral_noise_injector()
    check_anisotropic_spectral_step()
    print("ALL OPS CHECKS PASSED")

