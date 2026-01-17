import torch
import sys
import math

try:
    from UniPhyParaPool import UniPhyParaPool, MassConservingSwiGLU, LieAlgebraRotation, HighCapacityVectorMLP
except ImportError:
    from .UniPhyParaPool import UniPhyParaPool, MassConservingSwiGLU, LieAlgebraRotation, HighCapacityVectorMLP

def check_mass_conservation():
    print("Checking Mass Conservation (Scalar Stream)...")
    dim = 16
    layer = MassConservingSwiGLU(dim, dim * 2)
    x = torch.randn(2, dim, 32, 32)
    
    input_mass = x.mean(dim=(2, 3))
    out = layer(x)
    output_mass = out.mean(dim=(2, 3))
    
    diff = (input_mass - output_mass).abs().max().item()
    if diff < 1e-6:
        print(f"PASS: Mass drift {diff:.2e}")
    else:
        print(f"FAIL: Mass drift {diff:.2e}")
        sys.exit(1)

def check_orthogonality_enstrophy():
    print("Checking Orthogonality/Enstrophy Stability (Mixing Stream)...")
    dim = 16
    layer = LieAlgebraRotation(dim)
    
    A = layer.skew_generator.triu(diagonal=1)
    A = A - A.t()
    R = torch.linalg.matrix_exp(A)
    
    I = torch.eye(dim, device=x.device if 'x' in locals() else 'cpu')
    RTR = torch.matmul(R.T, R)
    
    diff = (RTR - I).abs().max().item()
    
    x = torch.randn(2, dim, 8, 8)
    out = layer(x)
    norm_in = torch.norm(x, p=2)
    norm_out = torch.norm(out, p=2)
    norm_diff = (norm_in - norm_out).abs().item()
    
    if diff < 1e-5 and norm_diff < 1e-5:
        print(f"PASS: Orthogonality error {diff:.2e}, Norm drift {norm_diff:.2e}")
    else:
        print(f"FAIL: Orthogonality error {diff:.2e}, Norm drift {norm_diff:.2e}")
        sys.exit(1)

def check_isotropy_angular_momentum():
    print("Checking Isotropy/Angular Momentum (Vector MLP)...")
    dim = 16 
    layer = HighCapacityVectorMLP(dim, dim * 2)
    
    B, H, W = 1, 4, 4
    x = torch.randn(B, dim, H, W)
    
    theta = 0.5
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_mat = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=x.device)
    
    x_reshaped = x.view(B, dim // 2, 2, H, W).permute(0, 1, 3, 4, 2)
    x_rot = torch.matmul(x_reshaped, rot_mat.T) 
    x_rot = x_rot.permute(0, 1, 4, 2, 3).reshape(B, dim, H, W)
    
    out_original = layer(x)
    out_rotated_input = layer(x_rot)
    
    out_original_reshaped = out_original.view(B, dim // 2, 2, H, W).permute(0, 1, 3, 4, 2)
    expected_rotated_output = torch.matmul(out_original_reshaped, rot_mat.T)
    expected_rotated_output = expected_rotated_output.permute(0, 1, 4, 2, 3).reshape(B, dim, H, W)
    
    diff = (out_rotated_input - expected_rotated_output).abs().max().item()
    
    if diff < 1e-5:
        print(f"PASS: Rotational Equivariance Error {diff:.2e}")
    else:
        print(f"FAIL: Rotational Equivariance Error {diff:.2e}")
        sys.exit(1)

def check_full_pool_structure():
    print("Checking Full UniPhyParaPool Structure...")
    dim = 32
    layer = UniPhyParaPool(dim, expand=4)
    x = torch.randn(2, dim, 16, 16)
    out = layer(x)
    
    if out.shape == x.shape:
        print("PASS: Input/Output shape match")
    else:
        print(f"FAIL: Shape mismatch {out.shape} vs {x.shape}")
        sys.exit(1)

if __name__ == "__main__":
    check_mass_conservation()
    check_orthogonality_enstrophy()
    check_isotropy_angular_momentum()
    check_full_pool_structure()
    print("ALL PHYSICS CHECKS PASSED")

