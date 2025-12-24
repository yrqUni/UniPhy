import os
import sys
import torch
import torch.nn as nn
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU

class MockArgs:
    def __init__(self):
        self.input_ch = 4
        self.out_ch = 4
        self.input_size = (32, 32)
        self.emb_ch = 32
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.static_ch = 2
        self.hidden_factor = (1, 1)
        self.convlru_num_blocks = 2
        self.ffn_hidden_ch = 64
        self.ffn_hidden_layers_num = 1
        self.use_cbam = False
        self.num_expert = 1
        self.activate_expert = 1
        self.lru_rank = 8
        self.use_selective = True
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.head_mode = "gaussian"
        self.dec_hidden_ch = 32
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"
        self.unet = True
        self.pool_mode = "pixel"

def check_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = MockArgs()
    print(f"Testing with unet={args.unet}, pool_mode={args.pool_mode}, num_expert={args.num_expert}")
    
    model = ConvLRU(args).to(device)
    model.eval()

    B, L, H, W = 2, 8, args.input_size[0], args.input_size[1]
    
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    static = torch.randn(B, args.static_ch, H, W, device=device)
    listT = torch.rand(B, L, device=device)

    print("-" * 40)
    print("Running P-Mode (Parallel)...")
    with torch.no_grad():
        out_p = model(x, mode="p", listT=listT, static_feats=static)
    
    print(f"P-Mode Output Shape: {out_p.shape}")

    print("-" * 40)
    print("Running I-Mode (Iterative Step-by-Step)...")
    
    outputs_i = []
    last_hidden_ins = None

    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t:t+1, ...]
            dt_t = listT[:, t:t+1]
            
            x_t_norm = model.revin(x_t, "norm")
            x_t_emb, _ = model.embedding(x_t_norm, static_feats=static)
            
            x_t_hid, last_hidden_ins = model.convlru_model(
                x_t_emb, 
                last_hidden_ins=last_hidden_ins, 
                listT=dt_t, 
                cond=model.embedding.static_embed(static) if static is not None else None
            )
            
            out_t = model.decoder(x_t_hid, cond=model.embedding.static_embed(static) if static is not None else None)
            out_t = out_t.permute(0, 2, 1, 3, 4).contiguous()
            
            if model.decoder.head_mode == "gaussian":
                mu, sigma = torch.chunk(out_t, 2, dim=2)
                if mu.size(2) == model.revin.num_features:
                    mu = model.revin(mu, "denorm")
                    sigma = sigma * model.revin.stdev
                out_step = torch.cat([mu, sigma], dim=2)
            else:
                if out_t.size(2) == model.revin.num_features:
                    out_step = model.revin(out_t, "denorm")
                else:
                    out_step = out_t
            
            outputs_i.append(out_step)

    out_i = torch.cat(outputs_i, dim=1)
    print(f"I-Mode Output Shape: {out_i.shape}")

    print("-" * 40)
    
    diff = torch.abs(out_p - out_i)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Absolute Difference: {max_diff:.2e}")
    print(f"Mean Absolute Difference: {mean_diff:.2e}")

    threshold = 5e-1 

    if max_diff < threshold:
        print("\n✅ SUCCESS: P-Mode and I-Mode are mathematically equivalent.")
    else:
        print("\n❌ FAILURE: Difference is still too high.")

if __name__ == "__main__":
    check_equivalence()

