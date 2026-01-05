import os
import sys
import gc
import torch
import torch.nn.functional as F
from types import SimpleNamespace, MethodType
from collections import defaultdict
import numpy as np

# -----------------------------------------------------------------------------
# Setup Paths
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "UniPhy")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import ModelUniPhy
from ModelUniPhy import UniPhy, _match_dt_seq

# -----------------------------------------------------------------------------
# Global Debug Storage
# -----------------------------------------------------------------------------
# Structure: DATA_LOG[mode][layer_name][variable_name] = Tensor or List of Tensors
DATA_LOG = {
    "p": defaultdict(lambda: defaultdict(list)),
    "i": defaultdict(lambda: defaultdict(list))
}
CURRENT_MODE = "p" # 'p' or 'i'

# -----------------------------------------------------------------------------
# Patched Forward Function for PhysicalRecurrentLayer
# -----------------------------------------------------------------------------
def debug_lru_forward(self, x, last_hidden_in=None, listT=None, static_feats=None):
    """
    A modified version of PhysicalRecurrentLayer.forward that logs intermediate states.
    """
    layer_name = getattr(self, 'debug_name', 'Unknown_LRU')
    
    B, C, L, H, W = x.shape
    
    # --- 1. Init State Logic ---
    if last_hidden_in is None:
        if self.init_state is not None and static_feats is not None:
            h0 = self.init_state(static_feats)
            curr_h = h0
        else:
            curr_h = torch.zeros(B, C, H, W, self.rank, device=x.device, dtype=x.dtype)
    else:
        curr_h = last_hidden_in
    
    # Log Initial Hidden State (h0 or passed h)
    # In I-mode this logs h_{t-1}, in P-mode it logs h_0
    DATA_LOG[CURRENT_MODE][layer_name]['h_init'].append(curr_h.detach().cpu())

    if listT is None:
        dt_seq = torch.ones(B, L, device=x.device, dtype=x.dtype)
    else:
        dt_seq = _match_dt_seq(listT.to(x.device, x.dtype), L)
    
    x_perm = x.permute(0, 2, 1, 3, 4)
    
    # --- 2. Hamiltonian (Flows) ---
    flows_all = self.hamiltonian(x_perm)
    DATA_LOG[CURRENT_MODE][layer_name]['flows'].append(flows_all.detach().cpu())
    
    # --- 3. Koopman Params ---
    koopman_params_all = self.koopman.compute_params(x_perm)
    # params is a tuple (nu, theta, sigma), we log them separately
    DATA_LOG[CURRENT_MODE][layer_name]['params_nu'].append(koopman_params_all[0].detach().cpu())
    
    dt_ref = torch.tensor(self.dt_ref, device=x.device, dtype=x.dtype).clamp_min(1e-6)
    h_stack = torch.empty((B, C, L, H, W, self.rank), device=x.device, dtype=x.dtype)
    
    # --- Loop ---
    for t in range(L):
        x_t = x[:, :, t : t + 1]
        dt_t = dt_seq[:, t : t + 1]
        flow_t = flows_all[:, t]
        
        nu_t = koopman_params_all[0][:, t]
        theta_t = koopman_params_all[1][:, t]
        sigma_t = koopman_params_all[2][:, t]
        params_t = (nu_t, theta_t, sigma_t)
        
        # --- 4. Lie Transport ---
        h_trans = self.lie_transport(curr_h, flow_t, dt_t)
        DATA_LOG[CURRENT_MODE][layer_name]['h_trans'].append(h_trans.detach().cpu())
        
        # --- 5. Koopman Step ---
        h_next = self.koopman.forward_step(h_trans, dt_t, params_t)
        DATA_LOG[CURRENT_MODE][layer_name]['h_next'].append(h_next.detach().cpu())
        
        x_inject = x_t.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, self.rank)
        dt_scaled = (dt_t / dt_ref).clamp_min(0.0)
        if self.inj_k > 0:
            g = 1.0 - torch.exp(-dt_scaled * torch.tensor(self.inj_k, device=x.device, dtype=x.dtype))
        else:
            g = dt_scaled
        curr_h = h_next + g.view(B, 1, 1, 1, 1) * x_inject
        curr_h = self.rec_norm(curr_h)
        h_stack[:, :, t] = curr_h
        
    out = self.proj_out(h_stack).squeeze(-1)
    out = self.post_ifft_proj(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    out = self.norm(out)
    out_gated = self.gate(out)
    x_out = x + out_gated
    
    return x_out, curr_h

# -----------------------------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------------------------
def analyze_differences():
    print(f"\n{'='*30} DEEP DEBUG REPORT {'='*30}")
    print(f"{'Variable':<30} | {'Max Diff':<12} | {'Status':<6}")
    print("-" * 60)

    # We only care about DownBlock_1_LRU based on previous run, but let's check all tracked
    for layer_name in sorted(DATA_LOG['p'].keys()):
        print(f"--- Layer: {layer_name} ---")
        
        vars_to_check = ['h_init', 'flows', 'params_nu', 'h_trans', 'h_next']
        
        for var in vars_to_check:
            # Prepare P data
            # P logs are lists containing 1 tensor (since forward called once with L=4)
            # EXCEPTION: Inside the loop, h_trans/h_next are appended L times.
            # BUT: debug_lru_forward logs 'flows' once [B, L, ...].
            # h_trans is logged L times inside forward.
            
            p_data = DATA_LOG['p'][layer_name][var]
            i_data = DATA_LOG['i'][layer_name][var]
            
            if not p_data or not i_data:
                print(f"{var:<30} | {'MISSING':<12} | SKIP")
                continue

            # Assemble Tensors
            try:
                # Determine how to stack I-mode data
                # I-mode forward called L times.
                # 'flows': Each I-step produces [B, 1, ...]. We stack dim 1.
                # 'h_trans': Each I-step produces [B, ..., H, W, R].
                
                # Case 1: Variables computed ONCE per forward (flows, params)
                if var in ['flows', 'params_nu']:
                    # P-mode: list len 1, tensor shape [B, L, ...]
                    tensor_p = p_data[0] 
                    # I-mode: list len L, each tensor [B, 1, ...]
                    tensor_i = torch.cat(i_data, dim=1) 
                
                # Case 2: Variables computed INSIDE loop (h_trans, h_next)
                elif var in ['h_trans', 'h_next']:
                    # P-mode: list len L (appended inside loop)
                    # Each element is [B, C, H, W, R] (time is collapsed in variable)
                    tensor_p = torch.stack(p_data, dim=1) # [B, L, ...]
                    
                    # I-mode: list len L (appended inside loop, forward called L times)
                    # Each element is [B, C, H, W, R]
                    tensor_i = torch.stack(i_data, dim=1)
                
                # Case 3: h_init
                elif var == 'h_init':
                    # h_init is special.
                    # P-mode: Computed once at start. List len 1. [B, C, H, W, R]
                    # I-mode: Computed at start of EACH forward.
                    #   Step 0: h_init is from static_feats (should match P-mode h_init)
                    #   Step 1: h_init is last_hidden_in (should match P-mode h_stack[t-1])
                    
                    # Let's just compare the FIRST step h_init to see if initialization matches.
                    tensor_p = p_data[0]
                    tensor_i = i_data[0]
                    var_label = "h_init (t=0)"
                    
                    diff = (tensor_p - tensor_i).abs().max().item()
                    status = "FAIL" if diff > 1e-4 else "OK"
                    print(f"{var_label:<30} | {diff:.2e}    | {status}")
                    continue

                else:
                    continue

                # Check shapes
                if tensor_p.shape != tensor_i.shape:
                    print(f"{var:<30} | SHAPE MIS | P{tuple(tensor_p.shape)} vs I{tuple(tensor_i.shape)}")
                    continue
                
                diff = (tensor_p - tensor_i).abs().max().item()
                status = "FAIL" if diff > 1e-4 else "OK"
                print(f"{var:<30} | {diff:.2e}    | {status}")

            except Exception as e:
                print(f"{var:<30} | ERROR       | {e}")


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------
def get_args():
    return SimpleNamespace(
        input_ch=2, out_ch=2, input_size=(32, 32), emb_ch=16,
        static_ch=4, hidden_factor=(2, 2), ConvType="conv", Arch="unet",
        dist_mode="gaussian", diff_mode="sobel", convlru_num_blocks=2,
        down_mode="avg", ffn_ratio=2.0, lru_rank=8,
        koopman_use_noise=False, koopman_noise_scale=1.0,
        learnable_init_state=True, dt_ref=1.0, inj_k=2.0, max_velocity=5.0,
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Deep Debug on {device} with static_ch=4...")
    
    args = get_args()
    model = UniPhy(args).to(device).eval()
    
    # --- Apply Monkey Patches ---
    # We locate all PhysicalRecurrentLayers and patch their forward method
    # and give them a debug_name
    
    count = 0
    # Walk through the model to find LRU layers
    if hasattr(model.uniphy_model, 'down_blocks'):
        for i, blk in enumerate(model.uniphy_model.down_blocks):
            blk.lru_layer.debug_name = f"DownBlock_{i}_LRU"
            blk.lru_layer.forward = MethodType(debug_lru_forward, blk.lru_layer)
            count += 1
        for i, blk in enumerate(model.uniphy_model.up_blocks):
            blk.lru_layer.debug_name = f"UpBlock_{i}_LRU"
            blk.lru_layer.forward = MethodType(debug_lru_forward, blk.lru_layer)
            count += 1
    else:
        for i, blk in enumerate(model.uniphy_model.uniphy_blocks):
            blk.lru_layer.debug_name = f"Block_{i}_LRU"
            blk.lru_layer.forward = MethodType(debug_lru_forward, blk.lru_layer)
            count += 1
            
    print(f"Patched {count} LRU layers.")

    # --- Setup Data ---
    B, L, H, W = 1, 4, args.input_size[0], args.input_size[1]
    C = args.input_ch
    torch.manual_seed(42)
    x_init = torch.randn(B, 1, C, H, W, device=device)
    static_feats = torch.randn(B, args.static_ch, H, W, device=device)
    stats = model.revin.stats(x_init)

    # --- Run Iterative Mode (I) ---
    global CURRENT_MODE
    CURRENT_MODE = "i"
    print("Executing Iterative Mode...")
    
    listT_i = torch.ones(B, 1, device=device)
    listT_future = torch.ones(B, L - 1, device=device)
    
    with torch.no_grad():
        out_i = model(
            x_init, mode="i", out_gen_num=L,
            listT=listT_i, listT_future=listT_future,
            static_feats=static_feats, revin_stats=stats
        )
        
        # Construct input for P mode
        preds = out_i[..., :args.out_ch, :, :]
        # Simple concat logic for checking
        p_list = [x_init]
        for t in range(L-1):
            pred_t = preds[:, t:t+1]
            if pred_t.shape[2] != C:
                 # zero pad if needed
                 diff = C - pred_t.shape[2]
                 pred_t = torch.cat([pred_t, torch.zeros(B, 1, diff, H, W, device=device)], dim=2)
            p_list.append(pred_t)
        x_p = torch.cat(p_list, dim=1)

    # --- Run Parallel Mode (P) ---
    CURRENT_MODE = "p"
    print("Executing Parallel Mode...")
    
    listT_p = torch.ones(B, L, device=device)
    with torch.no_grad():
        model(x_p, mode="p", listT=listT_p, static_feats=static_feats, revin_stats=stats)
        
    # --- Analyze ---
    analyze_differences()

if __name__ == "__main__":
    main()

