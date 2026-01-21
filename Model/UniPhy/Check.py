import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import torch.fft

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import MetriplecticPropagator, RiemannianCliffordConv2d, SpectralStep, GatedChannelMixer
from UniPhyIO import GlobalConservationConstraint
from PScan import PScanTriton
from ModelUniPhy import UniPhyModel, UniPhyBlock

def check_propagator_stability():
    print("\n--- Checking Propagator Stability & SSM Logic ---")
    dim = 16
    H, W = 32, 32
    prop = MetriplecticPropagator(dim, H, W, stochastic=False)
    
    dt = torch.tensor(1.0)
    x_dummy = torch.randn(1, 10, dim, H, W, dtype=torch.cfloat)
    
    _, B_valve, A_op, dt_eff = prop.get_operators(dt, x_dummy)
    
    if B_valve is not None:
        print(f"[PASS] B_valve generated with shape: {B_valve.shape}")
        if B_valve.max() > 2.1:
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
    model = UniPhyModel(in_channels=C, dim=16, out_channels=C, img_height=H, img_width=W, depth=1).to(device)
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
    x_raw = x
    
    z = model.embedding(x)
    
    for block in model.blocks:
        residual = z
        
        x_spatial = block.spatial_mixer(z)
        z = z + x_spatial
        
        x_spectral = block.spectral_mixer(z)
        z = z + x_spectral
        
        if z.is_complex():
            u_in = torch.cat([z.real, z.imag], dim=0)
        else:
            u_in = z
            
        z_fft = torch.fft.rfft2(u_in, norm='ortho')
        B_stacked, T, C_latent, H_latent, W_f = z_fft.shape
        
        _, B_op, A_op, dt_eff = block.propagator.get_operators(dt, z)
        
        if z.is_complex() and A_op.shape[0] == z.shape[0]:
            A_op = torch.cat([A_op, A_op], dim=0)
            
        if z.is_complex() and B_op.ndim == 3 and B_op.shape[0] == z.shape[0]:
             B_op = torch.cat([B_op, B_op], dim=0)
             
        if B_op.ndim == 3:
             u_fft = z_fft * B_op.unsqueeze(-1).unsqueeze(-1)
        else:
             u_fft = z_fft
             
        u_flat = u_fft.permute(0, 3, 4, 1, 2).reshape(B_stacked * H_latent * W_f, T, C_latent)
        A_flat = A_op.permute(0, 3, 4, 1, 2).reshape(B_stacked * H_latent * W_f, T, C_latent)
        
        h_state = torch.zeros(B_stacked * H_latent * W_f, C_latent, dtype=u_flat.dtype, device=u_flat.device)
        h_list = []
        
        for t in range(T):
            A_val = A_flat[:, t, :]
            u_val = u_flat[:, t, :]
            h_state = A_val * h_state + u_val
            h_list.append(h_state)
            
        h_stack = torch.stack(h_list, dim=1)
        
        h_freq = h_stack.view(B_stacked, H_latent, W_f, T, C_latent).permute(0, 3, 4, 1, 2)
        
        x_temporal = torch.fft.irfft2(h_freq, s=(block.H, block.W), norm='ortho')
        
        if z.is_complex():
            x_r, x_i = torch.chunk(x_temporal, 2, dim=0)
            x_temporal = torch.complex(x_r, x_i)
        
        noise = block.propagator.inject_noise(z, dt_eff)
        z = z + x_temporal + noise
        
        x_perm = z.permute(0, 1, 3, 4, 2)
        x_cat = torch.cat([x_perm.real, x_perm.imag], dim=-1)
        
        dt_scalar = dt_eff.mean() if isinstance(dt_eff, torch.Tensor) else dt
        x_ph = block.ph_layer(x_cat, dt=dt_scalar)
        
        x_ph = x_ph.permute(0, 1, 4, 2, 3)
        x_r, x_i = torch.chunk(x_ph, 2, dim=2)
        z_out = torch.complex(x_r, x_i)
        
        z = z_out
        
    out = model.decoder(z, x_raw)
    return out

def check_model_equivalence():
    print("\n--- Checking Parallel vs Sequential Equivalence (Whole Model) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("[WARNING] Triton PScan requires CUDA. Check might fail or skip PScan.")
    
    B, T, C, H, W = 1, 8, 2, 32, 32
    model = UniPhyModel(
        in_channels=C, 
        dim=16, 
        out_channels=C, 
        img_height=H, 
        img_width=W,
        depth=2, 
        expand=2
    ).to(device)
    
    model.eval()
    
    for block in model.blocks:
        block.propagator.stochastic = False
    
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

