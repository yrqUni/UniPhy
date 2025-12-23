import os
import sys
import torch
import torch.nn as nn

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
        self.emb_ch = 16
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.static_ch = 2
        self.hidden_factor = (1, 1)
        self.convlru_num_blocks = 1
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.use_cbam = False
        self.num_expert = 4
        self.activate_expert = 2
        self.lru_rank = 8
        self.use_selective = True
        self.bidirectional = True
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.head_mode = "gaussian"
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"
        self.use_checkpointing = False

def get_moe_aux_loss(model):
    total_aux_loss = 0.0
    count = 0
    for module in model.modules():
        if hasattr(module, "aux_loss") and isinstance(module.aux_loss, torch.Tensor):
            total_aux_loss += module.aux_loss
            count += 1
    return total_aux_loss, count

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    model = ConvLRU(args).to(device)

    B, L, H, W = 2, 8, 32, 32
    
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    static = torch.randn(B, args.static_ch, H, W, device=device)
    listT = torch.ones(B, L, device=device)

    print(f"Testing on {device}...")

    out_p = model(x, mode="p", listT=listT, static_feats=static)
    print(f"Forward Pass (Mode P) Output: {out_p.shape}")

    moe_loss, moe_layers = get_moe_aux_loss(model)
    print(f"MoE Auxiliary Loss: {moe_loss}")
    print(f"MoE Layers Found: {moe_layers}")

    if moe_layers > 0:
        print("MoE Loss Check Passed.")
    else:
        print("Warning: No MoE layers found (Check args.ffn_hidden_layers_num).")

    loss = out_p.sum() + moe_loss
    loss.backward()
    print("Backward Pass (Mode P + MoE Loss) Successful.")

    out_gen_num = 5
    listT_future = torch.ones(B, out_gen_num - 1, device=device)

    model.eval()
    with torch.no_grad():
        out_i = model(
            x, 
            mode="i", 
            out_gen_num=out_gen_num, 
            listT=listT, 
            listT_future=listT_future, 
            static_feats=static
        )
    
    print(f"Inference (Mode I) Output: {out_i.shape}")

    expected_ch = args.out_ch * 2 if args.head_mode == "gaussian" else args.out_ch
    expected_shape = (B, out_gen_num, expected_ch, H, W)
    
    if out_i.shape == expected_shape:
        print("Mode I Shape Check Passed.")
    else:
        print(f"Mode I Shape Mismatch. Expected {expected_shape}, got {out_i.shape}")

if __name__ == "__main__":
    run_test()

