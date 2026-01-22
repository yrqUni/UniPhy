import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import AnalyticSpectralPropagator, RiemannianCliffordConv2d
from UniPhyParaPool import UniPhyParaPool, FluxConservingSwiGLU, SymplecticExchange
from UniPhyIO import GlobalConservationConstraint
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

def check_pscan_adaptive_equivalence():
    print("\n--- Checking PScan Adaptive dt Equivalence ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("[SKIP] PScan requires CUDA.")
        return

    B, T, C, H, W = 1, 10, 2, 16, 16
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=1, img_height=H, img_width=W).to(device).double()
    model.eval()
    
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt_adaptive = torch.rand(B, T, device=device, dtype=torch.float64) + 0.1
    
    with torch.no_grad():
        out_parallel = model(x, dt_adaptive)
        
    z = model.encoder(x)
    block = model.blocks[0]
    
    z_seq_list = []
    B_z, T_z, D_z, H_z, W_z = z.shape
    h_curr = torch.zeros(B_z * H_z * W_z, D_z, device=device, dtype=torch.complex128)
    
    for t in range(T_z):
        x_step = z[:, t] 
        dt_step = dt_adaptive[:, t] 
        z_next, h_next = block.forward_step(x_step, h_curr, dt_step)
        z_seq_list.append(z_next)
        h_curr = h_next
        
    z_seq = torch.stack(z_seq_list, dim=1)
    out_seq = model.decoder(z_seq, x)
    
    diff = (out_parallel - out_seq).abs().max().item()
    print(f"PScan vs Sequential Error (Adaptive dt): {diff:.2e}")
    
    if diff < 1e-10:
        print("[PASS] PScan correctly handles adaptive time steps.")
    else:
        print("[FAIL] PScan logic divergence.")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("Running Checks in Float64 Precision...\n")
    
    check_plu_invertibility()
    check_semigroup_property()
    check_forced_response_integral()
    check_adaptive_step_invariance()
    check_variable_dt_broadcasting()
    if torch.cuda.is_available():
        check_pscan_adaptive_equivalence()

