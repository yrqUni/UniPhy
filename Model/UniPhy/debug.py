import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyBlock

def check(name, p, s, threshold=1e-6):
    p = p.detach().cpu()
    s = s.detach().cpu()
    diff = (p - s).abs().max().item()
    status = "PASS" if diff < threshold else "FAIL"
    print(f"[{status}] {name} | Max Diff: {diff:.2e}")
    if status == "FAIL":
        # Print stats to help debug
        print(f"    Parallel Mean: {p.mean():.4f}, Std: {p.std():.4f}")
        print(f"    Serial   Mean: {s.mean():.4f}, Std: {s.std():.4f}")

def debug_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Debug on {device} (float64)...\n")
    
    B, T, D, H, W = 2, 5, 4, 4, 4
    
    # Initialize block and FORCE noise to 0.0 to ensure deterministic comparison
    block = UniPhyBlock(D, 4, 4, H, W, dt_ref=1.0).to(device)
    block.prop.noise_scale = 0.0 # Crucial fix
    block.eval()
    
    # Inputs
    x_in = torch.randn(B, T, D, H, W, device=device) + 1j * torch.randn(B, T, D, H, W, device=device)
    dt_in = torch.ones(B, T, device=device)
    
    # -------------------------------------------------------------------------
    # 1. Check Forcing Term Consistency
    #    (Verifies: GlobalFluxTracker, Source Expansion, Basis Encoding)
    # -------------------------------------------------------------------------
    print("--- Checking Component: Forcing Term ---")
    
    # -- Parallel Path --
    x_t_par = x_in.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
    x_t_par = block._complex_norm(x_t_par, block.norm_temporal)
    x_eigen = block.prop.basis.encode(x_t_par)
    
    x_eigen_input = x_eigen.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)
    x_mean_seq = x_eigen_input.mean(dim=(-2, -1))
    
    flux_A, flux_X = block.prop.flux_tracker.get_operators(x_mean_seq)
    
    # Simulate PScan for Flux
    h = torch.zeros(B, D, device=device, dtype=torch.complex128)
    h_list = []
    for t in range(T):
        h = flux_A[:, :, t] * h + flux_X[:, :, t]
        h_list.append(h)
    flux_states = torch.stack(h_list, dim=2)
    source_seq = block.prop.flux_tracker.project(flux_states)
    
    source_expanded = source_seq.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, D)
    source_flat = source_expanded.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
    
    forcing_par = x_eigen + source_flat
    
    # -- Serial Path --
    forcing_ser_list = []
    h_flux = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    for t in range(T):
        xt = x_t_par[:, t].unsqueeze(1) # (N, 1, D)
        
        # Logic from forward_step: Encode -> Mean
        x_encoded_local = block.prop.basis.encode(xt)
        x_global_mean = x_encoded_local.view(B, H*W, D).mean(dim=1) # (B, D)
        
        # Flux Step
        h_flux, source = block.prop.flux_tracker.forward_step(h_flux, x_global_mean)
        
        # Expand Source (mimics UniPhyOps.forward_step)
        source_exp = source.view(B, 1, 1, D).expand(B, H*W, 1, D).reshape(B*H*W, 1, D)
        
        f_t = x_encoded_local + source_exp
        forcing_ser_list.append(f_t)
        
    forcing_ser = torch.stack(forcing_ser_list, dim=1).squeeze(2)
    
    check("Forcing Term", forcing_par, forcing_ser)
    
    # -------------------------------------------------------------------------
    # 2. Check Drift Consistency
    #    (Verifies: TemporalPropagator State Update, op_decay alignment)
    # -------------------------------------------------------------------------
    print("\n--- Checking Component: Drift Calculation ---")
    
    # Expand dt for parallel
    dt_flat = dt_in.reshape(B, 1, 1, T).expand(B, H, W, T).reshape(B*H*W, T)
    op_decay, op_forcing = block.prop.get_transition_operators(dt_flat)
    
    u_t = forcing_par * op_forcing
    
    # -- Parallel Path (Manual Scan) --
    h_main = torch.zeros(B*H*W, D, device=device, dtype=torch.complex128)
    h_main_list = []
    for t in range(T):
        h_main = h_main * op_decay[:, t] + u_t[:, t]
        h_main_list.append(h_main)
    h_eigen = torch.stack(h_main_list, dim=1)
    
    drift_par = block.prop.basis.decode(h_eigen).real
    
    # -- Serial Path --
    drift_ser_list = []
    h_state = torch.zeros(B*H*W, D, device=device, dtype=torch.complex128)
    
    for t in range(T):
        f_t = forcing_ser[:, t] # (N, D)
        d_t = op_decay[:, t]    # (N, D)
        o_t = op_forcing[:, t]  # (N, D)
        
        # Recurrence: h = h*d + forcing*forcing_op
        h_state = h_state * d_t + f_t * o_t
        drift_ser_list.append(block.prop.basis.decode(h_state).real)
        
    drift_ser = torch.stack(drift_ser_list, dim=1)
    
    check("Drift Calculation", drift_par, drift_ser)

if __name__ == "__main__":
    debug_main()