import torch
import torch.nn as nn
import sys
import os
import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModelUniPhy import UniPhyModel

# --- Monkey Patching to Inspect Internals ---
def forward_step_debug(self, x_step, h_prev_latent, dt_step, flux_prev):
    # Serial execution probe
    B, D, H, W = x_step.shape
    resid = x_step
    
    print(f"    [S] Input Mean: {x_step.mean().item():.6f}")
    
    # 1. Spatial
    x_s = x_step.permute(0, 2, 3, 1)
    x_s = self._complex_norm(x_s, self.norm_spatial).permute(0, 3, 1, 2)
    x_s = self._spatial_op(x_s)
    x = x_s + resid
    
    resid = x # Update resid!
    print(f"    [S] After Spatial Mean: {x.mean().item():.6f}")
    
    # 2. Temporal Input
    x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)
    x_t = self._complex_norm(x_t, self.norm_temporal)
    
    print(f"    [S] Temporal Input Norm Mean: {x_t.mean().item():.6f}")
    
    dt_t = torch.as_tensor(dt_step, device=x.device, dtype=x.real.dtype)
    if dt_t.numel() == B:
        dt_expanded = dt_t.reshape(B, 1, 1, 1).expand(B, H, W, 1).reshape(-1, 1)
    else:
        dt_expanded = dt_t.reshape(-1, 1)
    
    x_encoded_local = self.prop.basis.encode(x_t)
    x_global_mean_encoded = x_encoded_local.view(B, H * W, D).mean(dim=1)
    
    print(f"    [S] Global Mean: {x_global_mean_encoded.mean().item():.6f}")
    
    # Propagate
    h_next_latent, flux_next = self.prop.forward_step(h_prev_latent, x_t, x_global_mean_encoded, dt_expanded, flux_prev)
    
    print(f"    [S] H_Next Mean: {h_next_latent.mean().item():.6f}")
    
    x_drift = self.prop.basis.decode(h_next_latent).real.reshape(B, H, W, 1, D).permute(0, 3, 4, 1, 2).squeeze(1)
    
    print(f"    [S] Drift Mean: {x_drift.mean().item():.6f}")
    
    x = x_drift + resid
    
    print(f"    [S] Pre-FFN Mean: {x.mean().item():.6f}")
    
    x = x + self.ffn(x)
    return x, h_next_latent, flux_next

def run_trace():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Deep Trace on {device}...\n")
    
    B, T, C, H_in, W_in = 1, 5, 2, 32, 32
    dt_ref = 6.0
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H_in, img_width=W_in, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    for b in model.blocks: b.prop.noise_scale = 0.0
    
    # Patch the FIRST block
    model.blocks[0].forward_step = types.MethodType(forward_step_debug, model.blocks[0])
    
    x = torch.randn(B, T, C, H_in, W_in, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    # --- Parallel Run ---
    print(">>> Running Parallel Forward...")
    out_par = model(x, dt)
    
    # --- Manual Re-run of Block 0 Parallel Logic for Comparison ---
    print("\n>>> Re-running Block 0 Parallel Components for Step 0...")
    z = model.encoder(x)
    
    # Correctly get feature map dimensions
    B_z, T_z, D_z, H_z, W_z = z.shape
    print(f"    Encoder Output Shape: {z.shape}")
    
    block = model.blocks[0]
    
    # 1. Spatial Par
    x0 = z[:, 0] # Step 0 input (B, D, H, W)
    print(f"    [P] Input Mean: {x0.mean().item():.6f}")
    
    resid = z
    # Reshape using CORRECT dimensions (H_z, W_z)
    x_s = z.reshape(B_z * T_z, D_z, H_z, W_z).permute(0, 2, 3, 1)
    x_s = block._complex_norm(x_s, block.norm_spatial).permute(0, 3, 1, 2)
    x_s = block._spatial_op(x_s)
    x_out_s = x_s.reshape(B_z, T_z, D_z, H_z, W_z) + resid
    
    resid = x_out_s # Update Parallel Resid!
    print(f"    [P] After Spatial Mean (t=0): {x_out_s[:,0].mean().item():.6f}")
    
    # 2. Temporal Input
    x_t = x_out_s.permute(0, 3, 4, 1, 2).reshape(B_z * H_z * W_z, T_z, D_z)
    x_t = block._complex_norm(x_t, block.norm_temporal)
    print(f"    [P] Temporal Input Norm Mean (t=0): {x_t[:,0].mean().item():.6f}")
    
    # 3. Global Mean
    x_eigen = block.prop.basis.encode(x_t)
    x_eigen_input = x_eigen.reshape(B_z, H_z, W_z, T_z, D_z).permute(0, 3, 4, 1, 2)
    x_mean_seq = x_eigen_input.mean(dim=(-2, -1))
    print(f"    [P] Global Mean (t=0): {x_mean_seq[:,0].mean().item():.6f}")
    
    # --- Serial Run ---
    print("\n>>> Running Serial Forward (Watch Logs)...")
    
    h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
    
    # Only run step 0 for Block 0
    t = 0
    x_step = z[:, t]
    dt_step = dt[:, t]
    
    # This calls our patched method
    z_next, _, _ = block.forward_step(x_step, h_state, dt_step, flux_state)

if __name__ == "__main__":
    run_trace()
    