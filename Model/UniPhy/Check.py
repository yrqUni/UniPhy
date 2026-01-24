import torch
import torch.nn as nn
import sys
import os
import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyModel

# 全局变量用于存储串行运行的中间结果
serial_intermediates = {"h_next": [], "drift": [], "forcing": []}

def forward_step_hook(self, x_step, h_prev_latent, dt_step, flux_prev):
    B, D, H, W = x_step.shape
    resid = x_step
    
    # 1. Spatial
    x_s = x_step.permute(0, 2, 3, 1)
    x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
    x_s = self._spatial_op(x_s)
    x = x_s + resid
    resid = x 
    
    # 2. Temporal
    x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
    x_t = self._complex_norm(x_t, self.norm_temporal)
    
    dt_t = torch.as_tensor(dt_step, device=x.device, dtype=x.real.dtype)
    if dt_t.numel() == B:
        dt_expanded = dt_t.reshape(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
    else:
        dt_expanded = dt_t.reshape(-1, 1)
    
    # 3. Propagator Logic Capture
    x_encoded_local = self.prop.basis.encode(x_t)
    x_global_mean_encoded = x_encoded_local.view(B, H * W, D).mean(dim=1)
    
    # Reconstruct forcing for check
    x_tilde = x_encoded_local 
    _, source = self.prop.flux_tracker.forward_step(flux_prev, x_global_mean_encoded)
    source_expanded = source.view(B, 1, 1, D).expand(B, H*W, 1, D).reshape(B*H*W, 1, D)
    forcing = x_tilde + source_expanded
    serial_intermediates["forcing"].append(forcing.detach())

    # Actual Step
    h_next_latent, flux_next = self.prop.forward_step(h_prev_latent, x_t, x_global_mean_encoded, dt_expanded, flux_prev)
    serial_intermediates["h_next"].append(h_next_latent.detach()) 
    
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
    
    B, T, C, H_in, W_in = 1, 5, 2, 32, 32
    dt_ref = 6.0
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H_in, img_width=W_in, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    for b in model.blocks: b.prop.noise_scale = 0.0
    
    # Hook Block 0
    model.blocks[0].forward_step = types.MethodType(forward_step_hook, model.blocks[0])
    
    x = torch.randn(B, T, C, H_in, W_in, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    print(">>> 1. Executing Parallel Forward (Reference)...")
    with torch.no_grad():
        z = model.encoder(x)
        
        # 获取正确的 Latent 尺寸
        B_z, T_z, D_z, H_z, W_z = z.shape
        print(f"    Encoder Output Shape: {z.shape}")
        
        block = model.blocks[0]
        
        # --- Manual Parallel Trace using H_z, W_z ---
        resid = z
        # Spatial
        x_s = z.reshape(B_z * T_z, D_z, H_z, W_z).permute(0, 2, 3, 1)
        x_s = block._complex_norm(x_s, block.norm_spatial).permute(0, 3, 1, 2)
        x_s = block._spatial_op(x_s)
        x = x_s.reshape(B_z, T_z, D_z, H_z, W_z) + resid
        resid = x # Update Resid
        
        # Temporal Input
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B_z * H_z * W_z, T_z, D_z)
        x_t = block._complex_norm(x_t, block.norm_temporal)
        
        # Propagator Prep
        # Expand dt to match latent spatial size (H_z * W_z)
        dt_exp = dt.reshape(B, 1, 1, T).expand(B, H_z, W_z, T).reshape(B*H_z*W_z, T)
        op_decay, op_forcing = block.prop.get_transition_operators(dt_exp)
        
        x_eigen = block.prop.basis.encode(x_t)
        x_eigen_input = x_eigen.reshape(B_z, H_z, W_z, T_z, D_z).permute(0, 3, 4, 1, 2)
        x_mean_seq = x_eigen_input.mean(dim=(-2, -1))
        
        # Flux
        fA, fX = block.prop.flux_tracker.get_operators(x_mean_seq)
        f_states = block.pscan(fA, fX)
        src_seq = block.prop.flux_tracker.project(f_states)
        src_flat = src_seq.unsqueeze(2).unsqueeze(2).expand(B_z, T_z, H_z, W_z, D_z).permute(0, 2, 3, 1, 4).reshape(B_z*H_z*W_z, T_z, D_z)
        
        # Forcing
        forcing_par = x_eigen + src_flat 
        
        # Recurrence (PScan)
        u_t = forcing_par * op_forcing
        A = op_decay.permute(0, 2, 1).contiguous()
        X = u_t.permute(0, 2, 1).contiguous()
        h_eigen_par = block.pscan(A, X).permute(0, 2, 1) 
        
        # Drift
        drift_par = block.prop.basis.decode(h_eigen_par).real
        
    print(">>> 2. Executing Serial Forward (Test)...")
    with torch.no_grad():
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

    print("\n>>> 3. Comparison Results")
    
    # Forcing
    print(f"--- Forcing Term ---")
    forcing_ser_stack = torch.stack(serial_intermediates["forcing"], dim=1).squeeze(2)
    diff_f = (forcing_par - forcing_ser_stack).abs().max().item()
    print(f"Max Diff Forcing: {diff_f:.2e}")

    # H_Next
    print(f"\n--- Latent State H ---")
    h_ser_stack = torch.stack(serial_intermediates["h_next"], dim=1).squeeze(2)
    diff_h = (h_eigen_par - h_ser_stack).abs().max().item()
    print(f"Max Diff H State: {diff_h:.2e}")

    # Drift
    print(f"\n--- Drift ---")
    drift_ser_stack = torch.stack(serial_intermediates["drift"], dim=1) 
    drift_ser_flat = drift_ser_stack.permute(0, 3, 4, 1, 2).reshape(B_z*H_z*W_z, T, D_z)
    
    diff_d = (drift_par - drift_ser_flat).abs().max().item()
    print(f"Max Diff Drift:   {diff_d:.2e}")

if __name__ == "__main__":
    run_comparison()
    