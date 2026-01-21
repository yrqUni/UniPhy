import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import MetriplecticPropagator, RiemannianCliffordConv2d, SpectralStep, GatedChannelMixer
from UniPhyIO import GlobalConservationConstraint
from PScan import PScanTriton
from ModelUniPhy import UniPhyModel, UniPhyBlock

def check_propagator_stability():
    print("\n--- Checking Propagator Stability & SSM Logic ---")
    dim = 16
    prop = MetriplecticPropagator(dim, 32, 32, stochastic=False)
    
    dt = torch.tensor(1.0)
    C_op, B_op, A_op, dt_eff = prop.get_operators(dt)
    
    if C_op is None:
        print("[PASS] C_op is None (Identity Output).")
    else:
        print(f"[INFO] C_op shape: {C_op.shape}")
        
    if B_op is not None:
        print(f"[PASS] B_op (Input Valve) generated with shape: {B_op.shape}")
        if B_op.max() > 2.1:
            print("[WARN] Valve gain unusually high.")
        else:
            print("[PASS] Valve gain within expected range.")
            
    max_amp = A_op.abs().max().item()
    min_amp = A_op.abs().min().item()
    
    print(f"Evolution Eigenvalues Max Amp: {max_amp:.6f}")
    print(f"Evolution Eigenvalues Min Amp: {min_amp:.6f}")
    
    if max_amp < 1.05:
        print("[PASS] Evolution is stable (non-exploding).")
    else:
        print("[FAIL] Evolution eigenvalues imply explosion.")

def check_riemannian_metric():
    print("\n--- Checking Riemannian Metric Positivity ---")
    layer = RiemannianCliffordConv2d(16, 16, 3, 1, 32, 32)
    
    x = torch.randn(2, 16, 32, 32)
    _ = layer(x)
    
    base_metric = torch.exp(layer.log_metric_param)
    min_metric = base_metric.min().item()
    
    print(f"Min Metric Value: {min_metric:.6e}")
    
    if min_metric > 0:
        print("[PASS] Metric is strictly positive definite.")
    else:
        print("[FAIL] Metric contains non-positive values.")

def check_pscan_unit_test():
    print("\n--- Checking PScan vs Sequential Equivalence (Unit Test) ---")
    device = torch.device("cuda")
    pscan = PScanTriton().to(device)
    
    B, T, D = 2, 16, 8
    
    A = torch.randn(B, T, D, dtype=torch.cfloat, device=device)
    A = A / A.abs() * 0.99 
    X = torch.randn(B, T, D, dtype=torch.cfloat, device=device)
    
    try:
        out_parallel = pscan(A, X)
        
        out_sequential = torch.zeros_like(X)
        h = torch.zeros(B, D, dtype=torch.cfloat, device=device)
        for t in range(T):
            h = A[:, t] * h + X[:, t]
            out_sequential[:, t] = h
            
        diff = (out_parallel - out_sequential).abs().max().item()
        print(f"Max Diff (Parallel vs Sequential): {diff:.6e}")
        
        if diff < 1e-4:
            print("[PASS] PScan matches sequential logic.")
        else:
            print("[FAIL] PScan output diverges from sequential.")
            
    except Exception as e:
        print(f"[FAIL] PScan check error: {e}")

def check_global_constraint():
    print("\n--- Checking Global Conservation Constraint ---")
    constraint = GlobalConservationConstraint(conserved_indices=[0])
    
    B, T, C, H, W = 1, 5, 2, 16, 16
    pred = torch.randn(B, T, C, H, W)
    ref = torch.randn(B, T, C, H, W)
    
    fixed_pred = constraint(pred, ref)
    
    ref_slice_mass = ref[:, -1:, 0, ...].mean()
    pred_mass = fixed_pred[:, :, 0, ...].mean()
    
    diff = abs(pred_mass - ref_slice_mass).item()
    print(f"Ref Slice Mass: {ref_slice_mass:.6f}")
    print(f"Pred Mass (Corrected): {pred_mass:.6f}")
    print(f"Difference: {diff:.6e}")
    
    if diff < 1e-5:
        print("[PASS] Global constraint enforced successfully.")
    else:
        print("[FAIL] Global constraint failed.")

def check_model_stability_simulation():
    print("\n--- Checking Whole Model Long-term Stability ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("[INFO] Skipping simulation on CPU.")
        return

    B, T, C, H, W = 1, 10, 2, 16, 16
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=1, img_height=H, img_width=W).to(device)
    model.eval()
    
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.tensor(0.1, device=device)
    
    try:
        with torch.no_grad():
            out = model(x, dt)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[FAIL] Model output contains NaN or Inf.")
        else:
            print("[PASS] Model forward pass is numerically stable.")
            
    except Exception as e:
        print(f"[FAIL] Simulation error: {e}")
        import traceback
        traceback.print_exc()

def manual_sequential_forward(model, x, dt):
    B, T, C, H, W = x.shape
    
    z = model.encoder(x)
    
    for block in model.blocks:
        residual = z
        
        B_z, T_z, D_z, H_z, W_z = z.shape
        z_flat = z.view(B_z * T_z, D_z, H_z, W_z)
        
        z_perm = z_flat.permute(0, 2, 3, 1)
        z_cat = torch.cat([z_perm.real, z_perm.imag], dim=-1)
        z_norm = block.norm_spatial(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        z_norm_c = torch.complex(r, i).permute(0, 3, 1, 2)
        
        z_cat_conv = torch.cat([z_norm_c.real, z_norm_c.imag], dim=1)
        out_cliff = block.spatial_cliff(z_cat_conv)
        r_c, i_c = torch.chunk(out_cliff, 2, dim=1)
        out_cliff_c = torch.complex(r_c, i_c)
        
        out_spec = block.spatial_spec(z_norm_c)
        
        z_s = block.spatial_gate * out_cliff_c + (1.0 - block.spatial_gate) * out_spec
        z = z_s.view(B_z, T_z, D_z, H_z, W_z) + residual
        
        residual = z
        
        z_perm = z.permute(0, 1, 3, 4, 2)
        z_cat = torch.cat([z_perm.real, z_perm.imag], dim=-1)
        z_norm = block.norm_temporal(z_cat)
        r, i = torch.chunk(z_norm, 2, dim=-1)
        z_in = torch.complex(r, i).permute(0, 1, 4, 2, 3)
        
        C_op, B_op, A_op, dt_eff = block.prop.get_operators(dt, x_context=z_in)
        
        if B_op is None:
            u = z_in
        elif B_op.ndim >= 2 and B_op.shape[-1] == D_z and B_op.shape[-2] == D_z:
             z_flat_t = z_in.permute(0, 3, 4, 1, 2).reshape(B_z*H_z*W_z, T_z, D_z)
             if not B_op.is_complex():
                 B_op = B_op.to(dtype=z_in.dtype)
             u_flat = torch.matmul(z_flat_t, B_op.transpose(-1, -2)) 
             u = u_flat.view(B_z, H_z, W_z, T_z, D_z).permute(0, 3, 4, 1, 2)
        else:
             if B_op.ndim == 3: 
                 B_op_cast = B_op.unsqueeze(-1).unsqueeze(-1)
             elif B_op.ndim == z_in.ndim:
                 B_op_cast = B_op
             else:
                 B_op_cast = B_op
                 
             if not B_op_cast.is_complex() and z_in.is_complex():
                 B_op_cast = B_op_cast.to(dtype=z_in.dtype)
                 
             u = z_in * B_op_cast

        if A_op.ndim == 5:
            A_expanded = A_op.expand(B_z, H_z, W_z, T_z, D_z)
        elif A_op.ndim == 3:
            A_expanded = A_op.unsqueeze(1).unsqueeze(1).expand(B_z, H_z, W_z, T_z, D_z)
        else:
            A_expanded = A_op.view(B_z, 1, 1, -1, D_z).expand(B_z, H_z, W_z, T_z, D_z)
            
        A_flat = A_expanded.reshape(B_z * H_z * W_z, T_z, D_z)
        u_t = u.permute(0, 3, 4, 1, 2).reshape(B_z * H_z * W_z, T_z, D_z)
        
        h_eigen_list = []
        h_state = torch.zeros(B_z * H_z * W_z, D_z, dtype=u_t.dtype, device=u_t.device)
        
        for t in range(T_z):
            A_val = A_flat[:, t, :]
            u_val = u_t[:, t, :] 
            h_state = A_val * h_state + u_val
            h_eigen_list.append(h_state)
            
        h = torch.stack(h_eigen_list, dim=1)
        
        if C_op is None:
            y = h
        else:
            if not C_op.is_complex():
                 C_op = C_op.to(dtype=y.dtype)
            y = torch.matmul(h, C_op.transpose(-1, -2))
            
        x_drift = y.view(B_z, H_z, W_z, T_z, D_z).permute(0, 3, 4, 1, 2)
        noise = block.prop.inject_noise(z, dt_eff)
        
        z = x_drift + noise + residual
        
        residual = z
        
        if z.shape[2] == block.dim * 2 or z.shape[2] == block.dim:
             z_in = z.permute(0, 1, 3, 4, 2)
        else:
             z_in = z
             
        z_cat = torch.cat([z_in.real, z_in.imag], dim=-1)
        z_norm = block.mlp_norm(z_cat)
        z_mlp = block.mlp(z_norm)
        
        z_r, z_i = torch.chunk(z_mlp, 2, dim=-1)
        z_out = torch.complex(z_r, z_i)
        
        if z_out.ndim == 5 and z_out.shape[1] == T_z and z_out.shape[2] == H_z:
             z_out = z_out.permute(0, 1, 4, 2, 3)
             
        z = z + z_out
        
    out = model.decoder(z, x)
    return out

def check_model_equivalence():
    print("\n--- Checking Parallel vs Sequential Equivalence (Whole Model) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("[WARNING] Triton PScan requires CUDA. Check might fail or skip PScan.")
    
    B, T, C, H, W = 1, 8, 2, 32, 32
    model = UniPhyModel(
        in_channels=C, 
        out_channels=C, 
        embed_dim=16, 
        depth=2, 
        patch_size=16, 
        img_height=H, 
        img_width=W
    ).to(device)
    
    model.eval()
    
    for block in model.blocks:
        block.prop.stochastic = False
    
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.tensor(0.5, device=device)
    
    with torch.no_grad():
        try:
            out_parallel = model(x, dt)
        except Exception as e:
            print(f"[FAIL] Parallel execution failed: {e}")
            import traceback
            traceback.print_exc()
            return

        out_sequential = manual_sequential_forward(model, x, dt)
        
        diff = (out_parallel - out_sequential).abs().max().item()
        
        print(f"Max Absolute Difference: {diff:.9f}")
        
        if diff < 1e-3:
            print("[PASS] Model logic is equivalent in both modes.")
        else:
            print("[FAIL] Divergence detected between parallel and sequential modes.")

if __name__ == "__main__":
    check_propagator_stability()
    check_riemannian_metric()
    check_global_constraint()
    if torch.cuda.is_available():
        check_model_stability_simulation()
        check_pscan_unit_test()
        check_model_equivalence()
    else:
        print("\n[INFO] Skipping checks requiring CUDA")

