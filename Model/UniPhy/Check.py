import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import AnalyticSpectralPropagator
from ModelUniPhy import UniPhyModel

def check_plu_invertibility():
    print("--- Checking PLU Basis Invertibility ---")
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = AnalyticSpectralPropagator(dim).to(device)
    
    x = torch.randn(16, dim, device=device)
    x_rec = prop.basis.decode(prop.basis.encode(x)).real
    
    err = (x - x_rec).abs().max().item()
    print(f"Reconstruction Error: {err:.2e}")
    
    if err < 1e-12:
        print("[PASS] PLU Basis is strictly invertible.")
    else:
        print("[FAIL] PLU Basis inversion failed.")

def check_semigroup_property():
    print("\n--- Checking Semigroup Property (Free Evolution) ---")
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = AnalyticSpectralPropagator(dim).to(device)
    
    h0 = torch.randn(1, dim, device=device, dtype=torch.cfloat)
    x0 = torch.zeros(1, dim, device=device, dtype=torch.cfloat)
    T_total = 10.0
    
    h_jump = prop(h0, x0, dt=torch.tensor(T_total, device=device))
    
    steps = 100
    dt_small = torch.tensor(T_total / steps, device=device)
    h_step = h0
    for _ in range(steps):
        h_step = prop(h_step, x0, dt=dt_small)
    
    diff = (h_jump - h_step).abs().max().item()
    print(f"Jump vs 100 Steps Error: {diff:.2e}")
    
    if diff < 1e-10:
        print("[PASS] Semigroup property holds (Exact Analytic Integration).")
    else:
        print("[FAIL] Physics integration mismatch.")

def check_forced_response_integral():
    print("\n--- Checking Forced Response Integral ---")
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = AnalyticSpectralPropagator(dim).to(device)
    
    h0 = torch.zeros(1, dim, device=device, dtype=torch.cfloat)
    x_const = torch.randn(1, dim, device=device, dtype=torch.cfloat)
    T_total = 5.0
    
    h_jump = prop(h0, x_const, dt=torch.tensor(T_total, device=device))
    
    steps = 50
    dt_small = torch.tensor(T_total / steps, device=device)
    h_step = h0
    for _ in range(steps):
        h_step = prop(h_step, x_const, dt=dt_small)
    
    diff = (h_jump - h_step).abs().max().item()
    print(f"Integral Error: {diff:.2e}")
    
    if diff < 1e-10:
        print("[PASS] Forced response integral is accurate.")
    else:
        print("[FAIL] Integral calculation failed.")

def check_adaptive_step_invariance():
    print("\n--- Checking Adaptive Step Invariance ---")
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = AnalyticSpectralPropagator(dim).to(device)
    
    h0 = torch.randn(1, dim, device=device, dtype=torch.cfloat)
    x0 = torch.zeros(1, dim, device=device, dtype=torch.cfloat)
    
    T1 = 4.0
    T2 = 6.0
    T_total = T1 + T2
    
    h_uniform = h0
    steps = 100
    dt_uni = torch.tensor(T_total / steps, device=device)
    for _ in range(steps):
        h_uniform = prop(h_uniform, x0, dt=dt_uni)
        
    h_adaptive = h0
    h_adaptive = prop(h_adaptive, x0, dt=torch.tensor(T1, device=device))
    
    steps_remainder = 60
    dt_rem = torch.tensor(T2 / steps_remainder, device=device)
    for _ in range(steps_remainder):
        h_adaptive = prop(h_adaptive, x0, dt=dt_rem)
        
    diff = (h_uniform - h_adaptive).abs().max().item()
    print(f"Uniform vs Adaptive Sequence Error: {diff:.2e}")
    
    if diff < 1e-10:
        print("[PASS] System is invariant to discretization strategy.")
    else:
        print("[FAIL] Adaptive stepping inconsistency.")

def check_variable_dt_broadcasting():
    print("\n--- Checking Variable dt Broadcasting ---")
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = AnalyticSpectralPropagator(dim).to(device)
    
    h = torch.randn(2, dim, device=device, dtype=torch.cfloat)
    x = torch.zeros(2, dim, device=device, dtype=torch.cfloat)
    
    dt_batch = torch.tensor([1.5, 3.5], device=device)
    
    out_batch = prop(h, x, dt=dt_batch)
    
    out_0 = prop(h[0:1], x[0:1], dt=torch.tensor(1.5, device=device))
    out_1 = prop(h[1:2], x[1:2], dt=torch.tensor(3.5, device=device))
    
    err0 = (out_batch[0] - out_0[0]).abs().max().item()
    err1 = (out_batch[1] - out_1[0]).abs().max().item()
    
    print(f"Batch Consistency Errors: {err0:.2e}, {err1:.2e}")
    
    if err0 < 1e-12 and err1 < 1e-12:
        print("[PASS] Variable dt broadcasting is correct.")
    else:
        print("[FAIL] Broadcasting logic error.")

def check_full_model_adaptive_forward():
    print("\n--- Checking Full Model Adaptive Forward ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, T, C, H, W = 1, 10, 2, 16, 16
    model = UniPhyModel(in_channels=C, out_channels=C, img_height=H, img_width=W, depth=2).to(device).double()
    model.eval()
    
    x = torch.randn(B, T, C, H, W, device=device)
    
    dt_adaptive = torch.rand(B, T, device=device) * 2.0 + 0.1
    
    try:
        with torch.no_grad():
            out = model(x, dt_adaptive)
        print(f"Output Shape: {out.shape}")
        if out.shape == x.shape:
            print("[PASS] Full model accepts adaptive dt map.")
        else:
            print("[FAIL] Output shape mismatch.")
    except Exception as e:
        print(f"[FAIL] Forward execution error: {e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("Running Checks in Float64 Precision...\n")
    
    check_plu_invertibility()
    check_semigroup_property()
    check_forced_response_integral()
    check_adaptive_step_invariance()
    check_variable_dt_broadcasting()
    check_full_model_adaptive_forward()

