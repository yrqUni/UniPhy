import torch
import sys
import math
import torch.nn.functional as F

try:
    from UniPhyParaPool import UniPhyParaPool, MassConservingSwiGLU, LieAlgebraRotation, ThermodynamicVectorMLP
except ImportError:
    from .UniPhyParaPool import UniPhyParaPool, MassConservingSwiGLU, LieAlgebraRotation, ThermodynamicVectorMLP

def check_scalar_mass_mechanics():
    print("Checking Scalar Stream Mass Constraints...")
    dim = 16
    layer = MassConservingSwiGLU(dim, dim * 2)
    x = torch.randn(2, dim, 32, 32)
    
    out = layer(x, heat_gain=None)
    
    diff_ch1 = (out[:, 1:] - x[:, 1:]).mean(dim=(2, 3)).abs().max().item()
    
    if diff_ch1 < 1e-6:
        print(f"PASS: Passive Scalar Mass Conserved. Drift: {diff_ch1:.2e}")
    else:
        print(f"FAIL: Passive Scalar Mass Drift: {diff_ch1:.2e}")
        sys.exit(1)

def check_thermodynamic_consistency():
    print("Checking Total Energy Conservation (Thermodynamics)...")
    dim = 32
    pool = UniPhyParaPool(dim, expand=2)
    
    scalar_dim = dim // 4
    vector_dim = dim - scalar_dim
    n_vectors = vector_dim // 2
    
    x = torch.randn(1, dim, 16, 16)
    
    x_scalar = x[:, :scalar_dim, :, :]
    x_vector = x[:, scalar_dim:, :, :]
    x_vec_reshaped = x_vector.view(1, n_vectors, 2, 16, 16)
    ke_in = torch.sum(x_vec_reshaped ** 2)
    
    internal_in = x_scalar[:, 0, :, :].sum()
    
    total_energy_in = ke_in + internal_in
    
    out = pool(x)
    
    out_scalar = out[:, :scalar_dim, :, :]
    out_vector = out[:, scalar_dim:, :, :]
    
    out_vec_reshaped = out_vector.view(1, n_vectors, 2, 16, 16)
    ke_out = torch.sum(out_vec_reshaped ** 2)
    
    internal_out = out_scalar[:, 0, :, :].sum()
    
    total_energy_out = ke_out + internal_out
    
    diff = (total_energy_in - total_energy_out).abs().item()
    rel_diff = diff / total_energy_in.abs().item()
    
    if rel_diff < 1e-5:
        print(f"PASS: Total Energy (KE + Internal) Conserved. Rel Error: {rel_diff:.2e}")
    else:
        print(f"FAIL: Thermodynamics Violated. Rel Error: {rel_diff:.2e}")
        sys.exit(1)

def check_lie_orthogonality():
    print("Checking Lie Algebra Rotation Orthogonality...")
    dim = 16
    layer = LieAlgebraRotation(dim)
    
    A = layer.skew_generator.triu(diagonal=1)
    A = A - A.t()
    R = torch.linalg.matrix_exp(A)
    
    I = torch.eye(dim, device=R.device)
    RTR = torch.matmul(R.T, R)
    
    diff = (RTR - I).abs().max().item()
    
    if diff < 1e-5:
        print(f"PASS: Orthogonality verified. Error: {diff:.2e}")
    else:
        print(f"FAIL: Orthogonality broken. Error: {diff:.2e}")
        sys.exit(1)

def check_isotropy():
    print("Checking Vector MLP Isotropy (Rotational Equivariance)...")
    dim = 16 
    layer = ThermodynamicVectorMLP(dim, dim * 2)
    
    B, H, W = 1, 4, 4
    x = torch.randn(B, dim, H, W)
    
    theta = 0.5
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_mat = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=x.device)
    
    x_reshaped = x.view(B, dim // 2, 2, H, W).permute(0, 1, 3, 4, 2)
    x_rot = torch.matmul(x_reshaped, rot_mat.T) 
    x_rot = x_rot.permute(0, 1, 4, 2, 3).reshape(B, dim, H, W)
    
    out_original, _ = layer(x)
    out_rotated_input, _ = layer(x_rot)
    
    out_original_reshaped = out_original.view(B, dim // 2, 2, H, W).permute(0, 1, 3, 4, 2)
    expected_rotated_output = torch.matmul(out_original_reshaped, rot_mat.T)
    expected_rotated_output = expected_rotated_output.permute(0, 1, 4, 2, 3).reshape(B, dim, H, W)
    
    diff = (out_rotated_input - expected_rotated_output).abs().max().item()
    
    if diff < 1e-5:
        print(f"PASS: Isotropy verified. Error: {diff:.2e}")
    else:
        print(f"FAIL: Isotropy broken. Error: {diff:.2e}")
        sys.exit(1)

if __name__ == "__main__":
    check_scalar_mass_mechanics()
    check_thermodynamic_consistency()
    check_lie_orthogonality()
    check_isotropy()
    print("ALL PHYSICS CHECKS PASSED")

