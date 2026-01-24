import torch
import sys
import os

# Adjust path to find PScan
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from PScan import PScanTriton
except ImportError:
    print("Error: Could not import PScanTriton. Check path.")
    sys.exit(1)

def verify_pscan_logic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Verifying PScan Math Logic on {device}...\n")
    
    B, D, T = 1, 1, 5
    
    # Simple Inputs
    # A = 0.5 everywhere
    # X = 1.0 everywhere
    # h0 = X0 = 1.0
    # h1 = 0.5 * 1.0 + 1.0 = 1.5
    # h2 = 0.5 * 1.5 + 1.0 = 1.75
    # ...
    
    A = torch.ones(B, D, T, device=device, dtype=torch.float64) * 0.5
    X = torch.ones(B, D, T, device=device, dtype=torch.float64) * 1.0
    
    # 1. Run PScan
    pscan = PScanTriton()
    # PScan input usually expects (B, D, T)
    # But let's check if it needs Complex or Float. Usually works for both if implemented right.
    # Let's try Float first.
    try:
        out_par = pscan(A, X)
    except Exception as e:
        print(f"PScan failed with Float64: {e}")
        print("Switching to ComplexDouble...")
        A = A.to(torch.complex128)
        X = X.to(torch.complex128)
        out_par = pscan(A, X)

    # 2. Run Serial Reference (Standard Linear Recurrence)
    out_ser = torch.zeros(B, D, T, device=device, dtype=A.dtype)
    h = torch.zeros(B, D, device=device, dtype=A.dtype)
    
    vals = []
    for t in range(T):
        h = h * A[:, :, t] + X[:, :, t]
        vals.append(h)
    out_ser = torch.stack(vals, dim=2)
    
    print("Time | Parallel | Serial | Diff")
    for t in range(T):
        p_val = out_par[0,0,t].item()
        s_val = out_ser[0,0,t].item()
        diff = abs(p_val - s_val)
        print(f" t={t} | {p_val:.4f} | {s_val:.4f} | {diff:.2e}")
        
    total_diff = (out_par - out_ser).abs().max().item()
    if total_diff < 1e-6:
        print("\n[SUCCESS] PScan logic matches h_t = A * h_{t-1} + X")
    else:
        print("\n[FAIL] PScan logic does NOT match standard recurrence!")
        
        # Test Alternative Logic: h_t = A * h_{t-1} + A * X
        print("Testing Alternative Logic: h_t = A * h_{t-1} + A * X")
        h = torch.zeros(B, D, device=device, dtype=A.dtype)
        vals_alt = []
        for t in range(T):
            h = h * A[:, :, t] + A[:, :, t] * X[:, :, t]
            vals_alt.append(h)
        out_alt = torch.stack(vals_alt, dim=2)
        
        diff_alt = (out_par - out_alt).abs().max().item()
        if diff_alt < 1e-6:
             print("[MATCH] PScan actually implements: h_t = A * (h_{t-1} + X) or similar variant.")
        else:
             print("[UNKNOWN] PScan logic is unknown.")

if __name__ == "__main__":
    verify_pscan_logic()
    