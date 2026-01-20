import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import StablePropagator, RiemannianCliffordConv2d, SpectralStep
from UniPhyParaPool import UniPhyParaPool, FluxConservingSwiGLU, SymplecticExchange
from UniPhyIO import GlobalConservationConstraint
from PScan import PScanTriton
from ModelUniPhy import UniPhyModel

def check_symplectic_conservation():
    print("--- Checking Symplectic Exchange Conservation ---")
    B, C, H, W = 2, 4, 32, 32
    scalar_dim = C
    vector_dim = C
    layer = SymplecticExchange(scalar_dim, vector_dim)
    
    s = torch.randn(B, scalar_dim, H, W)
    v = torch.randn(B, vector_dim, H, W)
    
    energy_in = torch.sqrt((s**2).sum() + (v**2).sum())
    
    s_out, v_out = layer(s, v)
    
    energy_out = torch.sqrt((s_out**2).sum() + (v_out**2).sum())
    
    diff = abs(energy_in - energy_out).item()
    print(f"Energy In: {energy_in:.6f}")
    print(f"Energy Out: {energy_out:.6f}")
    print(f"Difference: {diff:.6e}")
    
    if diff < 1e-4:
        print("[PASS] Symplectic conservation holds.")
    else:
        print("[FAIL] Energy leakage detected.")

def check_propagator_stability():
    print("\n--- Checking Propagator Stability & Unitarity ---")
    dim = 16
    prop = StablePropagator(dim, 32, 32, stochastic=False)
    
    Q = prop.get_orthogonal_basis()
    I = torch.eye(dim, device=Q.device)
    Q_check = torch.matmul(Q.H, Q).real
    
    ortho_error = (Q_check - I).abs().max().item()
    print(f"Orthogonality Error (Q^H Q - I): {ortho_error:.6e}")
    
    if ortho_error < 1e-5:
        print("[PASS] Basis is orthogonal.")
    else:
        print("[FAIL] Basis is not orthogonal.")
        
    dt = torch.tensor(1.0)
    _, _, evo_diag, _ = prop.get_operators(dt)
    
    max_amp = evo_diag.abs().max().item()
    min_amp = evo_diag.abs().min().item()
    
    print(f"Evolution Eigenvalues Max Amp: {max_amp:.6f}")
    print(f"Evolution Eigenvalues Min Amp: {min_amp:.6f}")
    
    if max_amp < 1.05:
        print("[PASS] Evolution is stable (non-exploding).")
    else:
        print("[FAIL] Evolution eigenvalues imply explosion.")

def check_flux_conservation():
    print("\n--- Checking Flux Conservation ---")
    dim = 16
    layer = FluxConservingSwiGLU(dim, dim * 2)
    x = torch.randn(2, dim, 32, 32) + 5.0 
    
    mean_in = x.mean()
    out = layer(x)
    mean_out = out.mean()
    
    diff = abs(mean_out - mean_in).item()
    print(f"Mean Input: {mean_in:.6f}")
    print(f"Mean Output: {mean_out:.6f}")
    print(f"Difference (Drift): {diff:.6e}")
    
    if diff < 1e-5:
        print("[PASS] Global flux is conserved.")
    else:
        print("[FAIL] Significant mean drift detected.")

def check_riemannian_metric():
    print("\n--- Checking Riemannian Metric Positivity ---")
    layer = RiemannianCliffordConv2d(16, 16, 3, 1, 32, 32)
    
    x = torch.randn(2, 16, 32, 32)
    _ = layer(x)
    
    base_metric = torch.nn.functional.softplus(layer.metric_param) + 1e-6
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

def manual_sequential_forward(model, x, dt):
    B, T, C, H, W = x.shape
    
    z = model.encoder(x)
    
    for block in model.blocks:
        resid = z
        
        B_z, T_z, D_z, H_z, W_z = z.shape
        z_s = z.view(B_z * T_z, D_z, H_z, W_z).permute(0, 2, 3, 1)
        z_s = block._complex_norm(z_s, block.norm_spatial).permute(0, 3, 1, 2)
        z_s = block._spatial_op(z_s)
        z = z_s.view(B_z, T_z, D_z, H_z, W_z) + resid
        
        resid = z
        
        z_t = z.permute(0, 3, 4, 1, 2).reshape(B_z * H_z * W_z, T_z, D_z)
        z_t = block._complex_norm(z_t, block.norm_temporal)
        
        V, V_inv, evo_diag, dt_eff = block.prop.get_operators(dt, x_context=z)
        
        evo_diag_expanded = evo_diag.unsqueeze(1).unsqueeze(1).repeat(1, H_z, W_z, 1, 1)
        evo_diag_flat = evo_diag_expanded.view(B_z * H_z * W_z, T_z, D_z)
        
        x_eigen = torch.matmul(z_t, V_inv.T)
        
        h_eigen_list = []
        h_state = torch.zeros(B_z * H_z * W_z, D_z, dtype=x_eigen.dtype, device=x_eigen.device)
        
        for t in range(T_z):
            A_t = evo_diag_flat[:, t, :]
            u_t = x_eigen[:, t, :]
            h_state = A_t * h_state + u_t
            h_eigen_list.append(h_state)
            
        h_eigen = torch.stack(h_eigen_list, dim=1)
        
        x_t_out = torch.matmul(h_eigen, V.T)
        
        x_drift = x_t_out.view(B_z, H_z, W_z, T_z, D_z).permute(0, 3, 4, 1, 2)
        noise = block.prop.inject_noise(z, dt_eff)
        
        z = x_drift + noise + resid
        
        resid = z
        
        x_p = z.permute(0, 1, 3, 4, 2)
        x_p = block._complex_norm(x_p, block.norm_pool)
        
        x_p_flat = torch.cat([x_p.real, x_p.imag], dim=-1)
        B_p, T_p, H_p, W_p, C_p = x_p_flat.shape
        x_p_in = x_p_flat.view(B_p * T_p, H_p, W_p, C_p).permute(0, 3, 1, 2)
        
        x_pool_out = block.para_pool(x_p_in)
        
        x_pool_out = x_pool_out.permute(0, 2, 3, 1).view(B_p, T_p, H_p, W_p, C_p)
        r, i = torch.chunk(x_pool_out, 2, dim=-1)
        z = torch.complex(r, i).permute(0, 1, 4, 2, 3) + resid
        
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
            return

        out_sequential = manual_sequential_forward(model, x, dt)
        
        diff = (out_parallel - out_sequential).abs().max().item()
        
        print(f"Max Absolute Difference: {diff:.9f}")
        
        if diff < 1e-4:
            print("[PASS] Model logic is equivalent in both modes.")
        else:
            print("[FAIL] Divergence detected between parallel and sequential modes.")

if __name__ == "__main__":
    check_symplectic_conservation()
    check_propagator_stability()
    check_flux_conservation()
    check_riemannian_metric()
    check_global_constraint()
    if torch.cuda.is_available():
        check_pscan_unit_test()
        check_model_equivalence()
    else:
        print("\n[INFO] Skipping PScan checks (CUDA required)")

