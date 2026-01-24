import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import GlobalFluxTracker, TemporalPropagator, ComplexSVDTransform
from ModelUniPhy import UniPhyModel

def report_diff(name, tensor_a, tensor_b, threshold=1e-6):
    a = tensor_a.detach().cpu()
    b = tensor_b.detach().cpu()
    diff = (a - b).abs().max().item()
    status = "[PASS]" if diff < threshold else "[FAIL]"
    print(f"{status} {name} Max Diff: {diff:.2e}")
    return diff

def test_global_flux_tracker(device):
    print("\n=== Test 1: GlobalFluxTracker Consistency ===")
    B, T, D = 4, 16, 32
    tracker = GlobalFluxTracker(D).to(device)
    tracker.eval()
    
    x_re = torch.randn(B, T, D, device=device, dtype=torch.float64)
    x_im = torch.randn(B, T, D, device=device, dtype=torch.float64)
    x_input = torch.complex(x_re, x_im)
    
    flux_A, flux_X = tracker.get_operators(x_input)
    
    h = torch.zeros(B, D, device=device, dtype=torch.complex128)
    h_list = []
    for t in range(T):
        h = flux_A[:, :, t] * h + flux_X[:, :, t]
        h_list.append(h)
    flux_states = torch.stack(h_list, dim=2)
    source_seq_parallel = tracker.project(flux_states)
    
    source_seq_serial_list = []
    flux_state = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    for t in range(T):
        flux_state, source = tracker.forward_step(flux_state, x_input[:, t])
        source_seq_serial_list.append(source)
        
    source_seq_serial = torch.stack(source_seq_serial_list, dim=1)
    
    report_diff("GlobalFluxTracker Output", source_seq_parallel, source_seq_serial)

def test_temporal_propagator(device):
    print("\n=== Test 2: TemporalPropagator Consistency ===")
    B, T, D = 4, 16, 32
    prop = TemporalPropagator(D, noise_scale=0.0).to(device)
    prop.eval()
    
    h_init = torch.zeros(B, D, device=device, dtype=torch.complex128)
    x_input = torch.randn(B, T, D, device=device, dtype=torch.complex128)
    dt = torch.ones(B, T, device=device, dtype=torch.float64)
    
    op_decay, op_forcing = prop.get_transition_operators(dt)
    x_eigen = prop.basis.encode(x_input)
    
    flux_A, flux_X = prop.flux_tracker.get_operators(x_eigen)
    h = torch.zeros(B, D, device=device, dtype=torch.complex128)
    h_list = []
    for t in range(T):
        h = flux_A[:, :, t] * h + flux_X[:, :, t]
        h_list.append(h)
    flux_states = torch.stack(h_list, dim=2)
    source_seq = prop.flux_tracker.project(flux_states)
    
    forcing_term = x_eigen + source_seq
    u_t = forcing_term * op_forcing
    
    A_main = op_decay.permute(0, 2, 1).contiguous()
    X_main = u_t.permute(0, 2, 1).contiguous()
    
    h_main = h_init
    h_main_list = []
    for t in range(T):
        h_main = A_main[:, :, t] * h_main + X_main[:, :, t]
        h_main_list.append(h_main)
    h_eigen = torch.stack(h_main_list, dim=2).permute(0, 2, 1)
    
    out_parallel = prop.basis.decode(h_eigen)
    
    out_serial_list = []
    h_curr = h_init
    flux_curr = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    for t in range(T):
        x_t = x_input[:, t].unsqueeze(1)
        dt_t = dt[:, t]
        
        x_t_encoded = prop.basis.encode(x_t).squeeze(1)
        
        h_next, flux_next = prop.forward_step(h_curr, x_t, x_t_encoded, dt_t, flux_curr)
        
        out_serial_list.append(h_next)
        h_curr = h_next
        flux_curr = flux_next
        
    out_serial = torch.stack(out_serial_list, dim=1).squeeze(2)
    
    report_diff("TemporalPropagator Logic", out_parallel, out_serial)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Debug on {device}...")
    
    test_global_flux_tracker(device)
    test_temporal_propagator(device)
    