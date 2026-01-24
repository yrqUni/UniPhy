import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyBlock

def report(name, p, s):
    # Move to CPU for comparison
    diff = (p.detach().cpu() - s.detach().cpu()).abs().max().item()
    status = "PASS" if diff < 1e-6 else "FAIL"
    print(f"[{status}] {name} Diff: {diff:.2e}")

def debug_block_components():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Debugging UniPhyBlock Components on {device}...\n")
    
    B, T, D, H, W = 2, 5, 4, 4, 4 
    # Ensure noise is 0 to rule out stochastic mismatch
    block = UniPhyBlock(D, 4, 4, H, W, dt_ref=1.0, noise_scale=0.0).to(device)
    block.eval()
    
    # Complex Inputs
    x_in = torch.randn(B, T, D, H, W, device=device) + 1j * torch.randn(B, T, D, H, W, device=device)
    
    # -------------------------------------------------------------------------
    # 1. Spatial Part Consistency
    # -------------------------------------------------------------------------
    # Parallel Spatial Path
    x = x_in
    resid = x
    x_s_par = x.reshape(B * T, D, H, W).permute(0, 2, 3, 1) # (BT, H, W, D)
    x_s_par = block._complex_norm(x_s_par, block.norm_spatial).permute(0, 3, 1, 2)
    x_s_par = block._spatial_op(x_s_par)
    x_spatial_par = x_s_par.reshape(B, T, D, H, W) + resid
    
    # Serial Spatial Path
    x_spatial_ser_list = []
    for t in range(T):
        xt = x_in[:, t] # (B, D, H, W)
        resid_t = xt
        x_s_ser = xt.permute(0, 2, 3, 1) # (B, H, W, D)
        x_s_ser = block._complex_norm(x_s_ser, block.norm_spatial).permute(0, 3, 1, 2)
        x_s_ser = block._spatial_op(x_s_ser)
        x_t_out = x_s_ser + resid_t
        x_spatial_ser_list.append(x_t_out)
    x_spatial_ser = torch.stack(x_spatial_ser_list, dim=1)
    
    report("Spatial Ops", x_spatial_par, x_spatial_ser)
    
    # -------------------------------------------------------------------------
    # 2. Temporal Input Prep Consistency
    # -------------------------------------------------------------------------
    # Use consistent input from previous stage
    x_temporal_in = x_spatial_par
    
    # Parallel Prep
    resid_par = x_temporal_in
    x_t_par = x_temporal_in.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
    x_t_par = block._complex_norm(x_t_par, block.norm_temporal) # (N, T, D)
    
    # Serial Prep
    x_t_ser_list = []
    for t in range(T):
        xt = x_temporal_in[:, t] # (B, D, H, W)
        # Mimic forward_step logic
        xt_reshaped = xt.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
        xt_norm = block._complex_norm(xt_reshaped, block.norm_temporal)
        x_t_ser_list.append(xt_norm)
    x_t_ser = torch.stack(x_t_ser_list, dim=1).squeeze(2) # (N, T, D)
    
    report("Temporal Input Prep", x_t_par, x_t_ser)
    
    # -------------------------------------------------------------------------
    # 3. Propagator Integration (Using Prepped Inputs)
    # -------------------------------------------------------------------------
    # Use consistent prepped input
    x_prop_in = x_t_par
    dt_prop_in = torch.ones(B*H*W, T, device=device)
    
    # Parallel Prop Logic (Manually expanded from ModelUniPhy.forward)
    op_decay, op_forcing = block.prop.get_transition_operators(dt_prop_in)
    x_eigen = block.prop.basis.encode(x_prop_in)
    
    x_eigen_input = x_eigen.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2) 
    x_mean_seq = x_eigen_input.mean(dim=(-2, -1))
    
    flux_A, flux_X = block.prop.flux_tracker.get_operators(x_mean_seq)
    flux_states = block.pscan(flux_A, flux_X)
    source_seq = block.prop.flux_tracker.project(flux_states)
    
    source_expanded = source_seq.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, D)
    source_flat = source_expanded.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
    
    forcing = x_eigen + source_flat
    u_t = forcing * op_forcing
    
    A = op_decay.permute(0, 2, 1).contiguous()
    X = u_t.permute(0, 2, 1).contiguous()
    h_eigen = block.pscan(A, X).permute(0, 2, 1)
    
    x_drift_par = block.prop.basis.decode(h_eigen).real
    x_out_par = x_drift_par.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2) + resid_par
    
    # Serial Prop Logic
    x_out_ser_list = []
    h_state = torch.zeros(B*H*W, D, device=device, dtype=torch.complex128)
    f_state = torch.zeros(B, D, device=device, dtype=torch.complex128)
    
    # Need to unsqueeze dt to match serial step logic (N, 1)
    dt_serial_in = dt_prop_in.unsqueeze(2) 
    
    for t in range(T):
        # xt_step from serial prep list is (N, 1, D)
        xt_step = x_t_ser_list[t] 
        resid_t = resid_par[:, t] # (B, D, H, W)
        
        # Calculate mean from Encoded xt_step
        x_encoded_local = block.prop.basis.encode(xt_step)
        x_global_mean_encoded = x_encoded_local.view(B, H * W, D).mean(dim=1)
        
        dt_step = dt_serial_in[:, t]
        
        h_next, f_next = block.prop.forward_step(h_state, xt_step, x_global_mean_encoded, dt_step, f_state)
        
        x_drift_t = h_next.real.reshape(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
        x_out_t = x_drift_t + resid_t
        x_out_ser_list.append(x_out_t)
        
        h_state = h_next
        f_state = f_next
        
    x_out_ser = torch.stack(x_out_ser_list, dim=1)
    
    report("Propagator+Residual Integration", x_out_par, x_out_ser)

if __name__ == "__main__":
    debug_block_components()
    