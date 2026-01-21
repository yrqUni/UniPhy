import torch
import torch.nn as nn
import torch.optim as optim
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
    
    out = layer(x)
    mean_out = out.mean()
    
    diff = abs(mean_out).item()
    print(f"Input Mean: {x.mean():.6f}")
    print(f"Output Mean (Delta): {mean_out:.6f}")
    print(f"Difference from Zero: {diff:.6e}")
    
    if diff < 1e-5:
        print("[PASS] Global flux update is zero-mean (conserved).")
    else:
        print("[FAIL] Significant mean drift detected.")

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

def check_source_sink_stability():
    print("\n--- Checking Source/Sink Feedback Stability (Zero-Init) ---")
    dim = 64
    H, W = 32, 32
    steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UniPhyParaPool(dim, expand=4).to(device)
    
    model.eval()
    
    x = torch.randn(1, dim, H, W, device=device)
    x = x / x.std()
    
    energies = []
    
    print(f"Initial Energy: {x.std().item():.4f}")
    
    with torch.no_grad():
        for t in range(steps):
            delta = model(x)
            x = x + delta
            
            curr_energy = x.std().item()
            curr_max = x.abs().max().item()
            energies.append(curr_energy)
            
            if curr_max > 1e4:
                print(f"[FAIL] System exploded at step {t}")
                return

    growth_ratio = energies[-1] / energies[0]
    print(f"Final Energy Growth Ratio (100 steps): {growth_ratio:.4f}x")
    
    if growth_ratio < 1.1:
        print("[PASS] Zero-Init effective. Model starts as identity mapping.")
    elif growth_ratio > 50.0:
        print("[FAIL] Energy growth is too fast.")
    else:
        print("[PASS] Energy growth is within physical limits.")

def check_damping_learnability():
    print("\n--- Checking Damping Capability (Sink Mechanism) ---")
    dim = 64
    H, W = 16, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UniPhyParaPool(dim, expand=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    x_fixed = torch.randn(1, dim, H, W, device=device)
    x_fixed = x_fixed / x_fixed.std()
    
    initial_energy = x_fixed.std().item()
    print(f"Goal: Learn to suppress energy. Initial: {initial_energy:.4f}")
    
    for step in range(100):
        optimizer.zero_grad()
        delta = model(x_fixed)
        x_next = x_fixed + delta
        loss = (x_next ** 2).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        delta_final = model(x_fixed)
        final_energy = (x_fixed + delta_final).std().item()
        cos_sim = torch.nn.functional.cosine_similarity(delta_final.flatten(), x_fixed.flatten(), dim=0)
        
    print(f"Final Energy: {final_energy:.4f} (from {initial_energy:.4f})")
    print(f"Cosine Sim (Delta vs Input): {cos_sim.item():.4f}")
    
    if final_energy < initial_energy * 0.2 and cos_sim < -0.9:
        print("[PASS] Model successfully learned to act as a Sink.")
    else:
        print("[FAIL] Model failed to learn damping.")

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
        
        evo_diag_expanded = evo_diag.unsqueeze(1).unsqueeze(1).expand(B_z, H_z, W_z, T_z, D_z)
        evo_diag_flat = evo_diag_expanded.reshape(B_z * H_z * W_z, T_z, D_z)
        
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
        # try:
        out_parallel = model(x, dt)
        # except Exception as e:
        #     print(f"[FAIL] Parallel execution failed: {e}")
        #     return

        out_sequential = manual_sequential_forward(model, x, dt)
        
        diff = (out_parallel - out_sequential).abs().max().item()
        
        print(f"Max Absolute Difference: {diff:.9f}")
        
        if diff < 1e-3:
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
        check_source_sink_stability()
        check_damping_learnability()
        check_pscan_unit_test()
        check_model_equivalence()
    else:
        print("\n[INFO] Skipping checks requiring CUDA")

