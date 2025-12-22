import os
import sys
import torch

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
        self.ffn_hidden_layers_num = 0
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

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    model = ConvLRU(args).to(device)

    B, L, H, W = 2, 8, 32, 32
    x = torch.randn(B, L, 4, H, W, device=device)
    static = torch.randn(B, args.static_ch, H, W, device=device)
    listT = torch.ones(B, L, device=device)

    print(f"Testing on {device}...")

    out = model(x, mode="p", listT=listT, static_feats=static)
    print(f"Forward Pass Output: {out.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward Pass Successful.")

if __name__ == "__main__":
    run_test()

