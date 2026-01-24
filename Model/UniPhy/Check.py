import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import GlobalFluxTracker
# Try to import PScan to check if it exists/works
try:
    from PScan import PScanTriton
    pscan_available = True
except ImportError:
    pscan_available = False
    print("Warning: PScanTriton not found, mocking it for logic check.")

def debug_flux():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Flux Tracker Isolation Debug on {device}...\n")
    
    B, T, D = 1, 5, 16
    
    # Init Tracker
    tracker = GlobalFluxTracker(D).to(device)
    tracker.eval()
    
    # Random Input (Mean Sequence)
    x_mean = torch.randn(B, T, D, device=device, dtype=torch.float64) + \
             1j * torch.randn(B, T, D, device=device, dtype=torch.float64)
    
    # =========================================================
    # 1. Parallel Path (PScan)
    # =========================================================
    print(">>> 1. Parallel Path (PScan)")
    A_par, X_par = tracker.get_operators(x_mean)
    # A_par: (B, D, T), X_par: (B, D, T)
    
    # Use actual PScan if available, else manual simulation of PScan logic
    if pscan_available:
        pscan = PScanTriton()
        flux_states_par = pscan(A_par, X_par) # (B, D, T)
    else:
        # Manual PScan Simulation (Scan along last dim T)
        # h_t = A_t * h_{t-1} + X_t
        h = torch.zeros(B, D, device=device, dtype=torch.complex128)
        outs = []
        for t in range(T):
            A_t = A_par[:, :, t]
            X_t = X_par[:, :, t]
            h = h * A_t + X_t
            outs.append(h)
        flux_states_par = torch.stack(outs, dim=2)
        
    print(f"    Flux States Par Shape: {flux_states_par.shape}")
    print(f"    Flux States Par Mean:  {flux_states_par.mean().item():.6f}")

    # =========================================================
    # 2. Serial Path (Step-by-Step)
    # =========================================================
    print("\n>>> 2. Serial Path (Loop)")
    
    flux_state = torch.zeros(B, D, device=device, dtype=torch.complex128)
    flux_states_ser_list = []
    
    for t in range(T):
        x_m = x_mean[:, t] # (B, D)
        
        # Step
        # forward_step(flux_state, x_mean) -> returns (new_state, source)
        flux_next, _ = tracker.forward_step(flux_state, x_m)
        
        flux_states_ser_list.append(flux_next)
        flux_state = flux_next
        
    flux_states_ser = torch.stack(flux_states_ser_list, dim=2) # (B, D, T)
    print(f"    Flux States Ser Mean:  {flux_states_ser.mean().item():.6f}")

    # =========================================================
    # 3. Compare Internal Operators (A, X) vs (Decay, Input)
    # =========================================================
    print("\n>>> 3. Component Check")
    
    # Check A vs Decay
    # A_par is (B, D, T). Serial uses tracker._get_decay() (D)
    decay_ref = tracker._get_decay()
    A_slice = A_par[0, :, 0] # Should match decay
    diff_A = (A_slice - decay_ref).abs().max().item()
    print(f"    Operator A vs Decay Diff: {diff_A:.2e}")
    
    # Check X vs Input Mix
    # X_par is (B, D, T). Serial uses input_mix(x_m)
    x_m_0 = x_mean[:, 0]
    x_cat = torch.cat([x_m_0.real, x_m_0.imag], dim=-1)
    x_in = tracker.input_mix(x_cat)
    x_re, x_im = torch.chunk(x_in, 2, dim=-1)
    x_c = torch.complex(x_re, x_im) # (B, D)
    
    X_slice = X_par[:, :, 0] # (B, D)
    diff_X = (X_slice - x_c).abs().max().item()
    print(f"    Operator X vs Input Diff: {diff_X:.2e}")

    # =========================================================
    # 4. Final Result
    # =========================================================
    print("\n>>> 4. Final Result")
    diff_final = (flux_states_par - flux_states_ser).abs().max().item()
    print(f"    Max Diff Flux States: {diff_final:.2e}")
    
    if diff_final > 1e-6:
        print("    [FAIL] Mismatch detected.")
        print(f"    Par[0,0,:]: {flux_states_par[0,0,:].detach().cpu().numpy()}")
        print(f"    Ser[0,0,:]: {flux_states_ser[0,0,:].detach().cpu().numpy()}")
    else:
        print("    [PASS] Flux Tracker Logic is Consistent.")

if __name__ == "__main__":
    debug_flux()
    