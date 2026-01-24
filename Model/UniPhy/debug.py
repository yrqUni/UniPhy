import torch
import torch.nn as nn
import os
import sys

# 确保路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyBlock

def check_step(step_name, par_tensor, ser_tensor, threshold=1e-6):
    # Move to CPU
    p = par_tensor.detach().cpu()
    s = ser_tensor.detach().cpu()
    
    # Check shape first
    if p.shape != s.shape:
        print(f"[FAIL] {step_name} | Shape Mismatch! Par: {p.shape}, Ser: {s.shape}")
        return False
        
    diff = (p - s).abs().max().item()
    status = "PASS" if diff < threshold else "FAIL"
    print(f"[{status}] {step_name:<25} | Max Diff: {diff:.2e}")
    
    if status == "FAIL":
        print(f"    -> Stop Debugging here. Fix this step first.")
        return False
    return True

def debug_trace():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Strict Trace Debug on {device} (float64)...\n")
    
    # Config
    B, T, D, H, W = 2, 5, 4, 4, 4
    
    # Init Block with NO NOISE
    block = UniPhyBlock(D, 4, 4, H, W, dt_ref=1.0, noise_scale=0.0).to(device)
    block.eval()
    
    # Inputs
    x_in = torch.randn(B, T, D, H, W, device=device) + 1j * torch.randn(B, T, D, H, W, device=device)
    dt_in = torch.ones(B, T, device=device)
    
    print("=== Start Step-by-Step Trace ===")

    # -----------------------------------------------------------
    # Step 1: Spatial Operation (Norm + Conv + Activation)
    # -----------------------------------------------------------
    # [Parallel]
    x_s_par = x_in.reshape(B * T, D, H, W).permute(0, 2, 3, 1)
    x_s_par = block._complex_norm(x_s_par, block.norm_spatial).permute(0, 3, 1, 2)
    x_s_par = block._spatial_op(x_s_par) # (BT, D, H, W)
    x_s_par = x_s_par.reshape(B, T, D, H, W)
    
    # [Serial]
    x_s_ser_list = []
    for t in range(T):
        xt = x_in[:, t] # (B, D, H, W)
        xs = xt.permute(0, 2, 3, 1)
        xs = block._complex_norm(xs, block.norm_spatial).permute(0, 3, 1, 2)
        xs = block._spatial_op(xs)
        x_s_ser_list.append(xs)
    x_s_ser = torch.stack(x_s_ser_list, dim=1)
    
    if not check_step("1. Spatial Op", x_s_par, x_s_ser): return

    # -----------------------------------------------------------
    # Step 2: Update Residual (The Spatial Residual)
    # -----------------------------------------------------------
    # [Parallel]
    x_after_spatial_par = x_s_par + x_in # Residual 1
    # CRITICAL: In Parallel forward, 'resid' is updated to this value
    resid_par = x_after_spatial_par 
    
    # [Serial]
    x_after_spatial_ser_list = []
    for t in range(T):
        x_after = x_s_ser[:, t] + x_in[:, t]
        x_after_spatial_ser_list.append(x_after)
    x_after_spatial_ser = torch.stack(x_after_spatial_ser_list, dim=1)
    
    if not check_step("2. Spatial Residual", x_after_spatial_par, x_after_spatial_ser): return

    # -----------------------------------------------------------
    # Step 3: Temporal Prep (Permute + Norm)
    # -----------------------------------------------------------
    # [Parallel] Use result from Step 2
    x_t_in_par = x_after_spatial_par
    x_t_par = x_t_in_par.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
    x_t_par = block._complex_norm(x_t_par, block.norm_temporal)
    
    # [Serial]
    x_t_ser_list = []
    for t in range(T):
        xt = x_after_spatial_ser[:, t]
        xt_r = xt.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
        xt_n = block._complex_norm(xt_r, block.norm_temporal)
        x_t_ser_list.append(xt_n)
    x_t_ser = torch.stack(x_t_ser_list, dim=1).squeeze(2)
    
    if not check_step("3. Temporal Norm", x_t_par, x_t_ser): return

    # -----------------------------------------------------------
    # Step 4: Encoding & Global Mean & Flux
    # -----------------------------------------------------------
    # [Parallel]
    x_enc_par = block.prop.basis.encode(x_t_par)
    x_enc_reshaped = x_enc_par.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)
    x_mean_par = x_enc_reshaped.mean(dim=(-2, -1))
    
    # [Serial] Simulating getting mean per step
    x_enc_ser_list = []
    x_mean_ser_list = []
    for t in range(T):
        xt_n = x_t_ser_list[t] # (N, 1, D)
        x_enc = block.prop.basis.encode(xt_n) # (N, 1, D)
        x_mean = x_enc.view(B, H*W, D).mean(dim=1) # (B, D)
        x_enc_ser_list.append(x_enc)
        x_mean_ser_list.append(x_mean)
    x_enc_ser = torch.stack(x_enc_ser_list, dim=1).squeeze(2)
    x_mean_ser = torch.stack(x_mean_ser_list, dim=1)
    
    if not check_step("4a. Encoding", x_enc_par, x_enc_ser): return
    if not check_step("4b. Global Mean", x_mean_par, x_mean_ser): return

    # -----------------------------------------------------------
    # Step 5: Forcing Term (Eigen + Flux Source)
    # -----------------------------------------------------------
    # [Parallel]
    flux_A, flux_X = block.prop.flux_tracker.get_operators(x_mean_par)
    # Manual Flux Scan
    h_f = torch.zeros(B, D, device=device, dtype=torch.complex128)
    h_f_list = []
    for t in range(T):
        h_f = flux_A[:, :, t] * h_f + flux_X[:, :, t]
        h_f_list.append(h_f)
    flux_states = torch.stack(h_f_list, dim=2)
    source_par = block.prop.flux_tracker.project(flux_states)
    
    source_exp_par = source_par.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, D)
    source_flat_par = source_exp_par.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
    forcing_par = x_enc_par + source_flat_par
    
    # [Serial]
    forcing_ser_list = []
    h_f_ser = torch.zeros(B, D, device=device, dtype=torch.complex128)
    for t in range(T):
        x_m_t = x_mean_ser[:, t]
        h_f_ser, src_t = block.prop.flux_tracker.forward_step(h_f_ser, x_m_t)
        # Expand Source
        src_exp_t = src_t.view(B, 1, 1, D).expand(B, H*W, 1, D).reshape(B*H*W, 1, D)
        forcing_t = x_enc_ser_list[t] + src_exp_t
        forcing_ser_list.append(forcing_t)
    forcing_ser = torch.stack(forcing_ser_list, dim=1).squeeze(2)
    
    if not check_step("5. Forcing Term", forcing_par, forcing_ser): return
    
    # -----------------------------------------------------------
    # Step 6: Recurrence (PScan vs Step) - IN LATENT SPACE
    # -----------------------------------------------------------
    dt_flat = dt_in.reshape(B, 1, 1, T).expand(B, H, W, T).reshape(B*H*W, T)
    op_decay, op_forcing = block.prop.get_transition_operators(dt_flat)
    
    # [Parallel] Manual Scan
    u_t = forcing_par * op_forcing
    h_latent_par = torch.zeros(B*H*W, D, device=device, dtype=torch.complex128)
    h_latent_list = []
    for t in range(T):
        h_latent_par = h_latent_par * op_decay[:, t] + u_t[:, t]
        h_latent_list.append(h_latent_par)
    h_latent_stack_par = torch.stack(h_latent_list, dim=1) # (N, T, D)
    
    # [Serial] Using prop.forward_step
    h_latent_ser_list = []
    h_curr = torch.zeros(B*H*W, 1, D, device=device, dtype=torch.complex128) # Keep as (N, 1, D)
    flux_dummy = torch.zeros(B, D, device=device, dtype=torch.complex128) # Don't care, checked above
    
    for t in range(T):
        # We need to feed forward_step exactly what it expects.
        # But wait, prop.forward_step does Encoding internally!
        # This creates a mismatch in how we testing.
        # Let's call the internal recurrence logic manually to be 100% fair.
        
        # Manually: h_next = h_curr * decay + forcing * op_forcing
        d_t = op_decay[:, t].unsqueeze(1) # (N, 1, D)
        o_t = op_forcing[:, t].unsqueeze(1)
        f_t = forcing_ser_list[t] # (N, 1, D)
        
        h_next = h_curr * d_t + f_t * o_t
        h_latent_ser_list.append(h_next)
        h_curr = h_next
        
    h_latent_stack_ser = torch.stack(h_latent_ser_list, dim=1).squeeze(2)
    
    if not check_step("6. Latent State Recurrence", h_latent_stack_par, h_latent_stack_ser): return

    # -----------------------------------------------------------
    # Step 7: Final Decode & Output (Drift + Residual)
    # -----------------------------------------------------------
    # [Parallel]
    # Decode
    x_drift_par = block.prop.basis.decode(h_latent_stack_par).real
    x_drift_par = x_drift_par.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2) # (B, T, D, H, W)
    # Add Residual (This resid comes from Step 2)
    x_out_par = x_drift_par + resid_par 
    
    # [Serial]
    x_out_ser_list = []
    for t in range(T):
        h_t = h_latent_ser_list[t] # (N, 1, D)
        # Decode
        drift_t = block.prop.basis.decode(h_t).real # (N, 1, D)
        drift_t = drift_t.reshape(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1) # (B, D, H, W)
        
        # Add Residual
        # CRITICAL: In Serial Loop, we must add the residual corresponding to time t
        # The 'resid' here MUST be the output of spatial part (Step 2)
        resid_t = x_after_spatial_ser[:, t] 
        
        x_out_t = drift_t + resid_t
        x_out_ser_list.append(x_out_t)
        
    x_out_ser = torch.stack(x_out_ser_list, dim=1)
    
    check_step("7. Final Output (Drift+Resid)", x_out_par, x_out_ser)

if __name__ == "__main__":
    debug_trace()
    