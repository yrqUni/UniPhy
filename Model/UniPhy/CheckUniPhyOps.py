import torch
import sys
import os
from SpectralStep import SpectralStep
from CliffordConv2d import CliffordConv2d
from HamiltonianPropagator import HamiltonianPropagator
from UniPhyOps import UniPhyTransformerBlock

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

def check_clifford_shapes():
    print("Checking CliffordConv2d Shapes...")
    B, C, H, W = 2, 16, 32, 32
    layer = CliffordConv2d(in_channels=C, out_channels=C, kernel_size=3, padding=1)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    if out.shape == x.shape:
        print("PASS: Shape match")
    else:
        print(f"FAIL: Shape mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

def check_hamiltonian_properties():
    print("Checking HamiltonianPropagator Properties...")
    dim = 16
    dt = torch.ones(1)
    prop = HamiltonianPropagator(dim=dim, dt_ref=1.0, conserve_energy=False)
    
    V, V_inv, evo_diag = prop.get_operators(dt)
    lambda_modes = torch.log(evo_diag.squeeze())
    
    mass_check = (lambda_modes[0].abs().item() < 1e-6) and (lambda_modes[1].abs().item() < 1e-6)
    
    if mass_check:
        print("PASS: Mass anchor verified")
    else:
        print("FAIL: Mass anchor broken")
        sys.exit(1)

def check_block_consistency():
    print("Checking UniPhyTransformerBlock Serial vs Parallel Consistency...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 32
    B, T, H, W = 2, 5, 16, 16
    
    block = UniPhyTransformerBlock(dim=dim, state_dim=dim, img_height=H, img_width=W).to(device)
    block.eval()
    
    x = torch.randn(B, T, dim, H, W, dtype=torch.cfloat).to(device)
    dt = torch.rand(B, T).to(device)
    
    with torch.no_grad():
        out_parallel = block.forward_parallel(x, dt)
        
        out_serial_list = []
        state = None
        for t in range(T):
            step_in = x[:, t]
            step_dt = dt[:, t]
            step_out, state = block.step_serial(step_in, state, step_dt)
            out_serial_list.append(step_out)
        
        out_serial = torch.stack(out_serial_list, dim=1)
    
    diff = (out_parallel - out_serial).abs().max().item()
    if diff < 1e-4:
        print(f"PASS: Serial/Parallel mismatch {diff:.2e}")
    else:
        print(f"FAIL: Serial/Parallel mismatch {diff:.2e}")
        sys.exit(1)

if __name__ == "__main__":
    check_spectral_mass_conservation()
    check_clifford_shapes()
    check_hamiltonian_properties()
    check_block_consistency()
    print("ALL CHECKS PASSED")

