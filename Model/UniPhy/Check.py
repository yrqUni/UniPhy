import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import TemporalPropagator, ComplexSVDTransform
from UniPhyFFN import UniPhyFeedForwardNetwork
from ModelUniPhy import UniPhyModel

def check_basis_invertibility():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = ComplexSVDTransform(dim).to(device)
    x = torch.randn(16, dim, device=device, dtype=torch.cdouble)
    x_enc = basis.encode(x)
    x_dec = basis.decode(x_enc)
    err = (x - x_dec).abs().max().item()
    if err < 1e-12: pass
    else: print(f"Basis Inversion Error: {err:.2e}")

def check_history_dependency():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim, noise_scale=0.0).to(device)
    prop.eval()
    
    T = 5
    x_seq_1 = torch.randn(1, T, dim, 4, 4, device=device, dtype=torch.cdouble)
    x_seq_2 = x_seq_1.clone()
    x_seq_2[:, 0] += 10.0 
    
    mean_1 = x_seq_1.mean(dim=(-2, -1))
    mean_2 = x_seq_2.mean(dim=(-2, -1))
    
    def manual_scan_project(mean_seq):
        A, X = prop.flux_tracker.get_operators(mean_seq)
        B, D, T_ = A.shape
        h = torch.zeros(B, D, device=device, dtype=torch.cdouble)
        states = []
        for t in range(T_):
            h = h * A[:, :, t] + X[:, :, t]
            states.append(h)
        flux_states = torch.stack(states, dim=2)
        return prop.flux_tracker.project(flux_states)

    src_1 = manual_scan_project(mean_1)
    src_2 = manual_scan_project(mean_2)
    
    diff_at_last_step = (src_1[:, -1] - src_2[:, -1]).abs().mean().item()
    if diff_at_last_step > 1e-5: pass 
    else: print(f"History Dependency Error: Past change did not affect future source. Diff: {diff_at_last_step:.2e}")

def check_eigenvalue_stability():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim).to(device)
    lambda_val = prop._get_effective_lambda()
    max_real = lambda_val.real.max().item()
    if max_real <= 1e-6: pass
    else: print(f"Eigenvalue Stability Error: Max Real Part {max_real:.2e} > 0")

def check_full_model_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Model Consistency on {device}...")
    
    B, T, C, H, W = 1, 5, 2, 32, 32
    dt_ref = 6.0
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    
    for block in model.blocks:
        block.prop.noise_scale = 0.0
    
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        
        z = model.encoder(x)
        B_z, T_z, D_z, H_z, W_z = z.shape
        
        for block in model.blocks:
            h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
            flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
            z_steps = []
            
            for t in range(T):
                x_step = z[:, t]
                dt_step = dt[:, t]
                
                z_next, h_next, flux_next = block.forward_step(x_step, h_state, dt_step, flux_state)
                
                z_steps.append(z_next)
                h_state = h_next
                flux_state = flux_next
            
            z = torch.stack(z_steps, dim=1)
            
        out_serial = model.decoder(z)
        
        diff = (out_parallel - out_serial).abs().max().item()
        
    if diff < 1e-7:
        print(f"Consistency Check Passed. Diff: {diff:.2e}")
    else:
        print(f"Consistency Check FAILED. Diff: {diff:.2e}")
        print(f"   Parallel Mean: {out_parallel.mean():.6f}")
        print(f"   Serial   Mean: {out_serial.mean():.6f}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("Running Checks...")
    check_basis_invertibility()
    check_history_dependency()
    check_eigenvalue_stability()
    if torch.cuda.is_available():
        check_full_model_consistency()
    print("Checks Completed.")
    