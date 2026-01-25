import torch
import torch.nn as nn
import sys
import os

# 确保可以导入当前目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ModelUniPhy import UniPhyBlock
from PScan import pscan

# 用于存储中间结果的容器
DEBUG_LOGS = {"parallel": {}, "serial": {}}

class InstrumentedUniPhyBlock(UniPhyBlock):
    """
    继承 UniPhyBlock 并重写 forward 和 forward_step，
    用于捕获中间变量。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log(self, mode, step, name, tensor):
        if tensor is None: return
        # 如果是并行模式，直接存储
        if mode == "parallel":
            DEBUG_LOGS[mode][name] = tensor.detach().cpu()
        # 如果是串行模式，收集每一步的结果
        elif mode == "serial":
            if name not in DEBUG_LOGS[mode]:
                DEBUG_LOGS[mode][name] = []
            DEBUG_LOGS[mode][name].append(tensor.detach().cpu())

    def forward(self, x, dt):
        # --- Parallel Forward (Copy from ModelUniPhy.py with hooks) ---
        mode = "parallel"
        B, T, D, H, W = x.shape
        
        # 1. Input
        self._log(mode, None, "0_input", x)

        x_flat = x.reshape(B * T, D, H, W)
        x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        
        # 2. Spatial Cliff Output
        self._log(mode, None, "1_spatial_cliff", x_spatial.reshape(B, T, -1))
        
        x_spatial = x_spatial.reshape(B, T, D * 2, H, W)
        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=2)
        x = x + torch.complex(x_s_re, x_s_im)
        x_perm = x.permute(0, 1, 3, 4, 2)
        x_eigen = self.prop.basis.encode(x_perm)
        
        # 3. Basis Encode
        self._log(mode, None, "2_basis_encode", x_eigen)

        x_mean = x_eigen.mean(dim=(2, 3))
        A_flux, X_flux = self.prop.flux_tracker.get_operators(x_mean)
        flux_seq = self._run_pscan(
            A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)
        ).squeeze(-1)
        
        # 4. Flux Output
        self._log(mode, None, "3_flux_seq", flux_seq)

        source_seq = self.prop.flux_tracker.project(flux_seq)
        source_expanded = (
            source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        )
        gate_seq = torch.sigmoid(
            self.prop.flux_tracker.gate_net(
                torch.cat([flux_seq.real, flux_seq.imag], dim=-1)
            )
        )
        gate_expanded = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
        forcing = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)
        
        # 5. Forcing Term
        self._log(mode, None, "4_forcing", forcing)

        if dt.ndim == 1:
            dt_expanded = dt.unsqueeze(0).expand(B, T)
        else:
            dt_expanded = dt
        op_decay, op_forcing = self.prop.get_transition_operators(dt_expanded)
        op_decay = op_decay.view(B, T, 1, 1, D)
        op_forcing = op_forcing.view(B, T, 1, 1, D)
        u_t = forcing * op_forcing
        
        # --- CRITICAL SECTION: Reshape for PScan ---
        # Log the shape and content before PScan
        A_time_raw = op_decay.expand(B, T, H, W, D)
        self._log(mode, None, "5_A_time_raw", A_time_raw)

        # Correct logic from fixed ModelUniPhy.py
        A_time = A_time_raw.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        X_time = u_t.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
        
        Y_time = self._run_pscan(A_time, X_time)
        
        # 6. Time PScan Output
        # Reshape back for logging: (B, H, W, T, D) -> (B, T, D, H, W)
        self._log(mode, None, "6_pscan_out", Y_time.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)) 

        u_t = Y_time.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)
        
        noise = self.prop.generate_stochastic_term(
            u_t.shape, dt_expanded, u_t.dtype, h_state=u_t
        )
        u_t = u_t + noise
        
        x_out = self.prop.basis.decode(u_t)
        x_out = x_out.permute(0, 1, 4, 2, 3)
        
        # 7. Decode Output
        self._log(mode, None, "7_decode_out", x_out)

        x_out_flat = x_out.reshape(B * T, D, H, W)
        x_out_real = torch.cat([x_out_flat.real, x_out_flat.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)
        delta = self.ffn(x_out_complex)
        
        # 8. FFN Output
        self._log(mode, None, "8_ffn_out", delta.reshape(B, T, D, H, W))
        
        x_out_complex = x_out_complex + delta
        x_out_complex = x_out_complex.reshape(B, T, D, H, W)
        return x + x_out_complex

    def forward_step(self, x_curr, h_prev, dt, flux_prev):
        # --- Serial Step Forward ---
        mode = "serial"
        step = 0 # Placeholder, we use append list
        
        # 1. Input (Partial)
        self._log(mode, step, "0_input", x_curr)
        
        B, D, H, W = x_curr.shape
        x_real = torch.cat([x_curr.real, x_curr.imag], dim=1)
        x_norm = self.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial = self.spatial_cliff(x_norm)
        
        # 2. Spatial Cliff Output
        self._log(mode, step, "1_spatial_cliff", x_spatial)

        x_s_re, x_s_im = torch.chunk(x_spatial, 2, dim=1)
        x_curr = x_curr + torch.complex(x_s_re, x_s_im)
        x_perm = x_curr.permute(0, 2, 3, 1)
        x_eigen = self.prop.basis.encode(x_perm)
        
        # 3. Basis Encode
        self._log(mode, step, "2_basis_encode", x_eigen)

        x_mean = x_eigen.mean(dim=(1, 2))
        
        # Prop Step
        h_tilde_next, flux_next = self.prop.forward_step(
            h_prev, x_eigen.reshape(B * H * W, D), x_mean, dt, flux_prev
        )
        
        # 4. Flux Output
        self._log(mode, step, "3_flux_seq", flux_next)
        
        # 6. PScan Equivalent (Hidden State)
        # h_tilde_next is (B*H*W, 1, D)
        # Reshape to (B, H, W, D) for logging
        h_log = h_tilde_next.reshape(B, H, W, D).permute(0, 3, 1, 2) # -> B, D, H, W
        self._log(mode, step, "6_pscan_out", h_log) 

        h_out = h_tilde_next.reshape(B, H, W, 1, D)
        x_out = self.prop.basis.decode(h_out.squeeze(3))
        x_out = x_out.permute(0, 3, 1, 2)
        
        # 7. Decode Output
        self._log(mode, step, "7_decode_out", x_out)

        x_out_real = torch.cat([x_out.real, x_out.imag], dim=1)
        x_out_norm = self.norm_temporal(x_out_real.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        x_out_re, x_out_im = torch.chunk(x_out_norm, 2, dim=1)
        x_out_complex = torch.complex(x_out_re, x_out_im)
        delta = self.ffn(x_out_complex)
        
        # 8. FFN Output
        self._log(mode, step, "8_ffn_out", delta)

        x_out_complex = x_out_complex + delta
        z_next = x_curr + x_out_complex
        return z_next, h_tilde_next, flux_next


def run_debug():
    print("="*60)
    print("UniPhy Block Consistency Debugger")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Config
    B, T, D, H, W = 1, 5, 4, 16, 16 # Small size for debug
    
    # Init Block
    block = InstrumentedUniPhyBlock(
        dim=D, expand=2, num_experts=2, 
        img_height=H, img_width=W, 
        sde_mode="det" # Deterministic mode is crucial for consistency check
    ).to(device).double() # Use Double precision to rule out fp error
    
    block.eval()
    
    # Inputs
    x = torch.randn(B, T, D, H, W, device=device, dtype=torch.cdouble)
    dt = torch.ones(T, device=device, dtype=torch.float64)
    
    # --- Run Parallel ---
    print("Running Parallel Forward...")
    out_parallel = block(x, dt)
    
    # --- Run Serial ---
    print("Running Serial Forward...")
    
    # FIX: Initialize h_state with shape (B*H*W, 1, D) to match forward_step expectation
    h_state = torch.zeros(B * H * W, 1, D, device=device, dtype=torch.cdouble)
    flux_state = torch.zeros(B, D, device=device, dtype=torch.cdouble)
    out_serial_list = []
    
    for t in range(T):
        x_t = x[:, t]
        dt_t = dt[t:t+1]
        z_next, h_state, flux_state = block.forward_step(x_t, h_state, dt_t, flux_state)
        out_serial_list.append(z_next)
        
    out_serial = torch.stack(out_serial_list, dim=1)
    
    # --- Compare ---
    print("\nComparing Intermediate Tensors (Max Absolute Difference):")
    print("-" * 70)
    print(f"{'Layer Name':<20} | {'Shape (Parallel)':<20} | {'Max Diff':<15} | {'Status'}")
    print("-" * 70)
    
    keys = sorted(DEBUG_LOGS["parallel"].keys())
    
    # Skip raw params that don't have serial equivalents
    skip_keys = ["5_A_time_raw"] 
    
    first_fail = True
    
    for k in keys:
        if k in skip_keys: continue
        
        val_p = DEBUG_LOGS["parallel"][k] # (B, T, ...)
        val_s_list = DEBUG_LOGS["serial"].get(k)
        
        if not val_s_list:
            print(f"{k:<20} | {'N/A':<20} | {'N/A':<15} | Serial Missing")
            continue
            
        # Stack serial list to match parallel: (B, T, ...)
        try:
            val_s = torch.stack(val_s_list, dim=1)
        except:
             val_s = torch.stack(val_s_list, dim=1)

        # Align shapes if necessary
        if val_p.shape != val_s.shape:
             # Try to reshape if element count matches
             if val_p.numel() == val_s.numel():
                 val_s = val_s.reshape(val_p.shape)
             else:
                 print(f"{k:<20} | {str(tuple(val_p.shape)):<20} | {'Shape Mismatch':<15} | Serial: {tuple(val_s.shape)}")
                 continue
             
        diff = (val_p - val_s).abs().max().item()
        status = "OK" if diff < 1e-5 else "FAIL"
        
        print(f"{k:<20} | {str(tuple(val_p.shape)):<20} | {diff:.2e}        | {status}")

        if status == "FAIL" and first_fail:
            print("\n>>> FIRST FAILURE DETECTED AT: ", k)
            first_fail = False

    print("-" * 70)
    final_diff = (out_parallel - out_serial).abs().max().item()
    print(f"\nFinal Output Difference: {final_diff:.2e}")

if __name__ == "__main__":
    run_debug()
    