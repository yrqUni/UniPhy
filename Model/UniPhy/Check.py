import torch
import torch.nn as nn
import sys
import os
import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyModel

# 全局变量用于存储串行运行的中间结果，以便后续对比
serial_intermediates = {"h_next": [], "drift": [], "forcing": []}

def forward_step_hook(self, x_step, h_prev_latent, dt_step, flux_prev):
    B, D, H, W = x_step.shape
    resid = x_step
    
    # 1. Spatial
    x_s = x_step.permute(0, 2, 3, 1)
    x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
    x_s = self._spatial_op(x_s)
    x = x_s + resid
    resid = x # Resid Update
    
    # 2. Temporal
    x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
    x_t = self._complex_norm(x_t, self.norm_temporal)
    
    dt_t = torch.as_tensor(dt_step, device=x.device, dtype=x.real.dtype)
    if dt_t.numel() == B:
        dt_expanded = dt_t.reshape(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
    else:
        dt_expanded = dt_t.reshape(-1, 1)
    
    # 3. Propagator Logic
    x_encoded_local = self.prop.basis.encode(x_t)
    x_global_mean_encoded = x_encoded_local.view(B, H * W, D).mean(dim=1)
    
    # Inside Propagator Forward Step (Manually expanded to capture forcing)
    # forcing = x_tilde + source_expanded
    # We reconstruct forcing here for check
    x_tilde = x_encoded_local # (N, 1, D)
    _, source = self.prop.flux_tracker.forward_step(flux_prev, x_global_mean_encoded)
    source_expanded = source.view(B, 1, 1, D).expand(B, H*W, 1, D).reshape(B*H*W, 1, D)
    forcing = x_tilde + source_expanded
    serial_intermediates["forcing"].append(forcing.detach())

    # Actual Step
    h_next_latent, flux_next = self.prop.forward_step(h_prev_latent, x_t, x_global_mean_encoded, dt_expanded, flux_prev)
    serial_intermediates["h_next"].append(h_next_latent.detach()) # (N, 1, D)
    
    # Decode
    x_drift = self.prop.basis.decode(h_next_latent).real.reshape(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
    serial_intermediates["drift"].append(x_drift.detach())
    
    x = x_drift + resid
    x = x + self.ffn(x)
    return x, h_next_latent, flux_next

def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Step-by-Step Comparison on {device}...\n")
    
    B, T, C, H, W = 1, 5, 2, 32, 32
    dt_ref = 6.0
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    for b in model.blocks: b.prop.noise_scale = 0.0
    
    # Hook Block 0
    model.blocks[0].forward_step = types.MethodType(forward_step_hook, model.blocks[0])
    
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    # --- 1. Parallel Run ---
    # We need to extract intermediates from parallel run.
    # We will do this by temporarily modifying the forward method or just calculating manually.
    # Calculating manually is safer to avoid breaking autograd logic in model.
    
    print(">>> 1. Executing Parallel Forward (Reference)...")
    with torch.no_grad():
        z = model.encoder(x)
        block = model.blocks[0]
        
        # --- Manual Parallel Trace of Block 0 ---
        resid = z
        # Spatial
        x_s = z.reshape(B*T, 16, H, W).permute(0, 2, 3, 1)
        x_s = block._complex_norm(x_s, block.norm_spatial).permute(0, 3, 1, 2)
        x_s = block._spatial_op(x_s)
        x = x_s.reshape(B, T, 16, H, W) + resid
        resid = x # Update Resid
        
        # Temporal Input
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, 16)
        x_t = block._complex_norm(x_t, block.norm_temporal)
        
        # Propagator Prep
        dt_exp = dt.reshape(B, 1, 1, T).expand(B, H, W, T).reshape(B*H*W, T)
        op_decay, op_forcing = block.prop.get_transition_operators(dt_exp)
        
        x_eigen = block.prop.basis.encode(x_t)
        x_mean_seq = x_eigen.reshape(B, H, W, T, 16).permute(0, 3, 4, 1, 2).mean(dim=(-2, -1))
        
        # Flux
        fA, fX = block.prop.flux_tracker.get_operators(x_mean_seq)
        f_states = block.pscan(fA, fX)
        src_seq = block.prop.flux_tracker.project(f_states)
        src_flat = src_seq.unsqueeze(2).unsqueeze(2).expand(B, T, H, W, 16).permute(0, 2, 3, 1, 4).reshape(B*H*W, T, 16)
        
        # Forcing
        forcing_par = x_eigen + src_flat # (N, T, D)
        
        # Recurrence (PScan)
        u_t = forcing_par * op_forcing
        A = op_decay.permute(0, 2, 1).contiguous()
        X = u_t.permute(0, 2, 1).contiguous()
        h_eigen_par = block.pscan(A, X).permute(0, 2, 1) # (N, T, D)
        
        # Drift
        drift_par = block.prop.basis.decode(h_eigen_par).real # (N, T, D)
        
    # --- 2. Serial Run ---
    print(">>> 2. Executing Serial Forward (Test)...")
    with torch.no_grad():
        # Reset Serial Intermediates
        serial_intermediates["h_next"] = []
        serial_intermediates["drift"] = []
        serial_intermediates["forcing"] = []
        
        # We simulate the loop over T for Block 0 only
        B_z, T_z, D_z, H_z, W_z = z.shape
        h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
        flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
        
        for t in range(T):
            x_step = z[:, t]
            dt_step = dt[:, t]
            # This triggers the hook
            block.forward_step(x_step, h_state, dt_step, flux_state)
            
            # Update h_state for next step using the captured result
            h_state = serial_intermediates["h_next"][-1]
            # flux_state is updated inside forward_step but returned. 
            # We need to capture it properly. 
            # Actually, the hook returns flux_next, let's grab it from the return value if we run it properly.
            # But the hook modifies the return. Let's fix the loop execution.
            
    # Re-run Serial properly to handle state passing
    serial_intermediates["h_next"] = []
    serial_intermediates["drift"] = [] 
    serial_intermediates["forcing"] = []
    
    h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
    
    for t in range(T):
        x_step = z[:, t]
        dt_step = dt[:, t]
        _, h_next, flux_next = block.forward_step(x_step, h_state, dt_step, flux_state)
        h_state = h_next
        flux_state = flux_next

    # --- 3. Comparison ---
    print("\n>>> 3. Comparison Results (Diff per Step)")
    
    # Compare Forcing
    print(f"--- Forcing Term (Inputs to Recurrence) ---")
    forcing_ser_stack = torch.stack(serial_intermediates["forcing"], dim=1).squeeze(2)
    diff_f = (forcing_par - forcing_ser_stack).abs().max().item()
    print(f"Max Diff Forcing: {diff_f:.2e}")
    if diff_f > 1e-6:
        for t in range(T):
            d = (forcing_par[:, t] - forcing_ser_stack[:, t]).abs().max().item()
            print(f"  T={t} Diff: {d:.2e}")

    # Compare H_Next (Output of Recurrence)
    print(f"\n--- Latent State H (Output of Recurrence) ---")
    h_ser_stack = torch.stack(serial_intermediates["h_next"], dim=1).squeeze(2)
    diff_h = (h_eigen_par - h_ser_stack).abs().max().item()
    print(f"Max Diff H State: {diff_h:.2e}")
    if diff_h > 1e-6:
        for t in range(T):
            d = (h_eigen_par[:, t] - h_ser_stack[:, t]).abs().max().item()
            print(f"  T={t} Diff: {d:.2e}")

    # Compare Drift
    print(f"\n--- Drift (Decoded Output) ---")
    # drift_par is (N, T, D). serial drift is list of (B, D, H, W) -> need reshape
    drift_ser_stack = torch.stack(serial_intermediates["drift"], dim=1) # (B, T, D, H, W)
    drift_ser_flat = drift_ser_stack.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, 16)
    
    diff_d = (drift_par - drift_ser_flat).abs().max().item()
    print(f"Max Diff Drift:   {diff_d:.2e}")

if __name__ == "__main__":
    run_comparison()