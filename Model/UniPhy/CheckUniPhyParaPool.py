import torch
import sys
import torch.nn.functional as F

from UniPhyParaPool import MassConservingSwiGLU, ThermodynamicVectorMLP, LieAlgebraRotation, UniPhyParaPool

def check_swiglu_learned_distribution():
    print("Checking MassConservingSwiGLU (Learned Heat Distribution)...")
    B, C, H, W = 2, 8, 16, 16
    dim = C
    hidden = C * 4
    layer = MassConservingSwiGLU(dim, hidden)
    
    x = torch.randn(B, C, H, W)
    
    out_cold = layer(x, heat_gain=None)
    drift_cold = (out_cold - x).mean(dim=(2, 3))
    err_cold = drift_cold.abs().max().item()
    
    if err_cold < 1e-6:
        print(f"PASS: Zero-Mean Update verified (No Heat). Max Drift: {err_cold:.2e}")
    else:
        print(f"FAIL: Zero-Mean Update violated. Max Drift: {err_cold:.2e}")
        sys.exit(1)
        
    heat_input = torch.ones(B, 1, H, W) * 10.0
    out_heated = layer(x, heat_gain=heat_input)
    
    actual_delta = out_heated - out_cold
    
    total_added_mass = actual_delta.sum(dim=1, keepdim=True)
    
    gate_val = torch.sigmoid(layer.heat_gate).item()
    expected_mass = heat_input * gate_val
    
    mass_diff = (total_added_mass - expected_mass).abs().mean().item()
    
    if mass_diff < 1e-5:
        print(f"PASS: Heat injection conserved globally via Gate/Softmax. Diff: {mass_diff:.2e}")
    else:
        print(f"FAIL: Heat injection logic mismatch. Diff: {mass_diff:.2e}")
        sys.exit(1)

    dist_variance = actual_delta.var(dim=1).mean().item()
    if dist_variance > 0:
        print(f"PASS: Heat is distributed unevenly (Learnable). Var: {dist_variance:.2e}")
    else:
        print("WARNING: Heat distribution is perfectly uniform (Expected random init variance).")

def check_thermo_mlp_conservation():
    print("Checking ThermodynamicVectorMLP (Energy Conservation)...")
    dim = 16 
    hidden = 32
    layer = ThermodynamicVectorMLP(dim, hidden)
    
    B, C, H, W = 2, dim, 16, 16
    x = torch.randn(B, C, H, W)
    
    out_vec, energy_dissipated = layer(x)
    
    n_vectors = dim // 2
    vec_in = x.view(B, n_vectors, 2, H, W)
    ke_in = (vec_in ** 2).sum(dim=2)
    
    vec_out = out_vec.view(B, n_vectors, 2, H, W)
    ke_out = (vec_out ** 2).sum(dim=2)
    
    total_energy_in = ke_in.sum()
    total_energy_out = ke_out.sum() + energy_dissipated.sum()
    
    err = abs(total_energy_in - total_energy_out).item() / (total_energy_in.item() + 1e-8)
    
    if err < 1e-5:
        print(f"PASS: Energy Conservation (Kinetic + Heat). Rel Error: {err:.2e}")
    else:
        print(f"FAIL: Energy Leakage detected. Rel Error: {err:.2e}")
        sys.exit(1)

def check_lie_rotation():
    print("Checking LieAlgebraRotation (Orthogonality & Stability)...")
    dim = 16
    layer = LieAlgebraRotation(dim)
    
    A = layer.skew_generator.triu(1)
    A = A - A.t()
    R = torch.linalg.matrix_exp(A)
    
    I = torch.eye(dim, device=R.device)
    RTR = torch.matmul(R.T, R)
    
    err = (RTR - I).abs().max().item()
    if err < 1e-5:
        print(f"PASS: Rotation Matrix is Orthogonal. Error: {err:.2e}")
    else:
        print(f"FAIL: Rotation Matrix not Orthogonal. Error: {err:.2e}")
        sys.exit(1)

    gen_mag = layer.skew_generator.abs().mean().item()
    if gen_mag < 0.01:
         print(f"PASS: Initialization is conservative (small). Mag: {gen_mag:.4f}")
    else:
         print(f"WARNING: Initialization might be too large. Mag: {gen_mag:.4f}")

def check_para_pool_integration():
    print("Checking UniPhyParaPool Pipeline...")
    dim = 32
    layer = UniPhyParaPool(dim, expand=4)
    
    B, C, H, W = 2, dim, 16, 16
    x = torch.randn(B, C, H, W)
    
    out = layer(x)
    
    if out.shape == x.shape:
        print("PASS: Output shape matches input.")
    else:
        print(f"FAIL: Shape mismatch. {out.shape} vs {x.shape}")
        sys.exit(1)

if __name__ == "__main__":
    check_swiglu_learned_distribution()
    check_thermo_mlp_conservation()
    check_lie_rotation()
    check_para_pool_integration()
    print("ALL PARAPOOL CHECKS PASSED")

