import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import TemporalPropagator, ComplexSVDTransform
from UniPhyFFN import UniPhyFeedForwardNetwork
from ModelUniPhy import UniPhyModel

def compare_tensors(name, p, s, tolerance=1e-6):
    p = p.detach().cpu()
    s = s.detach().cpu()
    diff = (p - s).abs().max().item()
    if diff > tolerance:
        print(f"  [FAIL] {name} | Diff: {diff:.2e}")
        print(f"         Parallel Mean: {p.mean():.6f}, Std: {p.std():.6f}")
        print(f"         Serial   Mean: {s.mean():.6f}, Std: {s.std():.6f}")
        return False
    else:
        print(f"  [PASS] {name} | Diff: {diff:.2e}")
        return True

def run_trace():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    print(f"Running Layer-wise Trace on {device}...\n")
    
    # Config
    B, T, C, H, W = 1, 5, 2, 32, 32
    dt_ref = 6.0
    
    # 1. Initialize Model & Force Zero Noise
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    for b in model.blocks: b.prop.noise_scale = 0.0
    
    # Inputs
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    # =========================================================================
    # Phase 1: Encoder Check
    # =========================================================================
    print(">>> Phase 1: Encoder Check")
    z = model.encoder(x) # (B, T, D, H, W)
    print(f"  Encoder Output Shape: {z.shape}")
    
    # Current input for the blocks
    current_input_par = z.clone()
    current_input_ser = z.clone() # Should be identical initially
    
    B_z, T_z, D_z, H_z, W_z = z.shape
    
    # =========================================================================
    # Phase 2: Block-by-Block Check
    # =========================================================================
    for i, block in enumerate(model.blocks):
        print(f"\n>>> Phase 2.{i}: Checking Block {i}")
        
        # --- A. Run Parallel Block ---
        out_par = block(current_input_par, dt)
        
        # Capture Parallel Internal States (stored in self.last_...)
        h_final_par = block.last_h_state # (N, 1, D)
        
        # --- B. Run Serial Block (Step-by-Step Trace) ---
        h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
        flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
        
        z_steps = []
        
        # Trace individual steps inside the block loop
        step_failed = False
        for t in range(T):
            x_step_in = current_input_ser[:, t] # Input for this step
            dt_step_in = dt[:, t]
            
            # Run Serial Step
            z_next, h_next, flux_next = block.forward_step(x_step_in, h_state, dt_step_in, flux_state)
            
            # --- MICRO CHECK: Compare Output at t=0 ---
            if t == 0:
                # To compare, we need to hack into the parallel output at t=0
                z_par_t0 = out_par[:, 0]
                print(f"  ... Checking Step t=0 consistency ...")
                if not compare_tensors(f"Block {i} Step 0 Output", z_par_t0, z_next):
                    step_failed = True
                    # If step 0 fails, we need to dig deeper into why forward_step differs
                    print("  !!! Critical Failure at First Step. Checking Internals...")
                    
                    # Debug: Re-run components manually for t=0
                    # 1. Spatial
                    x_s_par = x_step_in.permute(0, 2, 3, 1)
                    x_s_par = block._complex_norm(x_s_par, block.norm_spatial).permute(0, 3, 1, 2)
                    x_s_par = block._spatial_op(x_s_par)
                    res_spatial = x_s_par + x_step_in
                    # 2. Temporal Input
                    xt_in = res_spatial.permute(0, 2, 3, 1).reshape(-1, 1, block.dim)
                    xt_in = block._complex_norm(xt_in, block.norm_temporal)
                    # 3. Propagator
                    enc = block.prop.basis.encode(xt_in) # (N, 1, D)
                    mean = enc.view(B, -1, block.dim).mean(dim=1)
                    op_d, op_f = block.prop.get_transition_operators(dt_step_in.reshape(-1,1).expand(-1, H*W).reshape(-1,1))
                    
                    # Manually compute H0
                    # h0 = 0 * d + (enc + src) * f
                    # Just print means to see what matches
                    print(f"      [Debug] Spatial Res Mean: {res_spatial.mean().item():.6f}")
                    print(f"      [Debug] Encoded Mean:     {enc.mean().item():.6f}")
                    print(f"      [Debug] Op Forcing Mean:  {op_f.mean().item():.6f}")
                    break

            z_steps.append(z_next)
            h_state = h_next
            flux_state = flux_next
        
        if step_failed:
            print("  !!! Stopping Trace due to error.")
            return

        out_ser = torch.stack(z_steps, dim=1)
        
        # --- C. Compare Block Output ---
        if not compare_tensors(f"Block {i} Full Output", out_par, out_ser):
            print("  !!! Mismatch in Block Output. Stopping.")
            return
            
        # --- D. Compare Hidden State (Final) ---
        # Note: h_state is (N, 1, D) latent
        if not compare_tensors(f"Block {i} Final Hidden State", h_final_par, h_state):
            print("  !!! Mismatch in Hidden State.")
            
        # Update inputs for next block
        current_input_par = out_par
        current_input_ser = out_ser

    # =========================================================================
    # Phase 3: Decoder Check
    # =========================================================================
    print("\n>>> Phase 3: Decoder Check")
    final_par = model.decoder(current_input_par)
    final_ser = model.decoder(current_input_ser)
    compare_tensors("Decoder Output", final_par, final_ser)

    print("\n>>> Trace Completed.")

if __name__ == "__main__":
    run_trace()
    