import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import GlobalFluxTracker, TemporalPropagator
from ModelUniPhy import UniPhyBlock

def check(name, parallel_out, serial_out, threshold=1e-6):
    p = parallel_out.detach().cpu()
    s = serial_out.detach().cpu()
    diff = (p - s).abs().max().item()
    status = "PASS" if diff < threshold else "FAIL"
    print(f"[{status}] {name} | Max Diff: {diff:.2e}")
    return diff

def debug_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Debug on {device} (float64)...\n")
    
    B, T, D, H, W = 2, 8, 16, 4, 4
    
    print("--- 1. Testing GlobalFluxTracker ---")
    tracker = GlobalFluxTracker(D).to(device)
    tracker.eval()
    
    x_mean = torch.randn(B, T, D, device=device) + 1j * torch.randn(B, T, D, device=device)
    
    A, X = tracker.get_operators(x_mean)
    h_par = torch.zeros(B, D, device=device, dtype=torch.complex128)
    h_states = []
    for t in range(T):
        h_par = A[:, :, t] * h_par + X[:, :, t]
        h_states.append(h_par)
    h_stack = torch.stack(h_states, dim=2)
    src_par = tracker.project(h_stack)
    
    src_ser_list = []
    flux_state = torch.zeros(B, D, device=device, dtype=torch.complex128)
    for t in range(T):
        flux_state, src = tracker.forward_step(flux_state, x_mean[:, t])
        src_ser_list.append(src)
    src_ser = torch.stack(src_ser_list, dim=1)
    
    check("GlobalFluxTracker", src_par, src_ser)

    print("\n--- 2. Testing TemporalPropagator ---")
    prop = TemporalPropagator(D, noise_scale=0.0).to(device)
    prop.eval()
    
    x_raw = torch.randn(B * H * W, T, D, device=device) + 1j * torch.randn(B * H * W, T, D, device=device)
    dt = torch.ones(B * H * W, T, device=device)
    
    op_decay, op_forcing = prop.get_transition_operators(dt)
    x_eigen = prop.basis.encode(x_raw)
    
    x_eigen_reshaped = x_eigen.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)
    x_mean_fake = x_eigen_reshaped.mean(dim=(-2, -1))
    
    fA, fX = prop.flux_tracker.get_operators(x_mean_fake)
    fh = torch.zeros(B, D, device=device, dtype=torch.complex128)
    fh_list = []
    for t in range(T):
        fh = fA[:, :, t] * fh + fX[:, :, t]
        fh_list.append(fh)
    f_states = torch.stack(fh_list, dim=2)
    src_seq = prop.flux_tracker.project(f_states)
    
    src_expanded = src_seq.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, D)
    src_flat = src_expanded.permute(0, 2, 3, 1, 4).reshape(B*H*W, T, D)
    
    forcing = x_eigen + src_flat
    u_t = forcing * op_forcing
    
    h_main = torch.zeros(B*H*W, 1, D, device=device, dtype=torch.complex128) # Latent state is (N, 1, D)
    h_main_list = []
    for t in range(T):
        decay_t = op_decay[:, t, :].unsqueeze(1)
        u_curr = u_t[:, t, :].unsqueeze(1)
        h_main = h_main * decay_t + u_curr
        h_main_list.append(h_main) # (N, 1, D)
    h_final = torch.stack(h_main_list, dim=1).squeeze(2) # (N, T, D)
    out_par = prop.basis.decode(h_final)
    
    out_ser_list = []
    # h_curr must match forward_step input expectation (Latent)
    h_curr = torch.zeros(B*H*W, 1, D, device=device, dtype=torch.complex128)
    flux_curr = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    for t in range(T):
        xt = x_raw[:, t].unsqueeze(1)
        dtt = dt[:, t].unsqueeze(1)
        x_mean_t = x_mean_fake[:, t]
        
        # Propagator forward step now returns LATENT h_next
        h_next_latent, flux_next = prop.forward_step(h_curr, xt, x_mean_t, dtt, flux_curr)
        
        # Decode manually for check
        out_next = prop.basis.decode(h_next_latent)
        out_ser_list.append(out_next)
        
        h_curr = h_next_latent
        flux_curr = flux_next
        
    out_ser = torch.stack(out_ser_list, dim=1).squeeze(2)
    
    check("TemporalPropagator", out_par, out_ser)

    print("\n--- 3. Testing UniPhyBlock Integration ---")
    block = UniPhyBlock(D, 4, 4, H, W, dt_ref=1.0, noise_scale=0.0).to(device)
    block.eval()
    
    x_block = torch.randn(B, T, D, H, W, device=device) + 1j * torch.randn(B, T, D, H, W, device=device)
    dt_block = torch.ones(B, T, device=device)
    
    y_par = block(x_block, dt_block)
    
    y_ser_list = []
    # Initialize h_state in Latent Space: (N, 1, D)
    h_state = torch.zeros(B*H*W, 1, D, device=device, dtype=torch.complex128)
    f_state = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    z = x_block 
    for t in range(T):
        xt = z[:, t]
        dtt = dt_block[:, t]
        
        y_next, h_next, f_next = block.forward_step(xt, h_state, dtt, f_state)
        y_ser_list.append(y_next)
        h_state = h_next
        f_state = f_next
        
    y_ser = torch.stack(y_ser_list, dim=1)
    
    check("UniPhyBlock", y_par, y_ser)

if __name__ == "__main__":
    debug_main()
    