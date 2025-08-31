import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))
from ModelConvLRU import ConvLRU
from pscan import pscan_check

def set_seed(s=0):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ArgsBase:
    def __init__(self):
        self.input_size = (144, 144)
        self.input_ch = 8
        self.out_ch = 8
        self.emb_ch = 16
        self.convlru_num_blocks = 2
        self.hidden_factor = (2, 2)
        self.use_gate = True
        self.use_cbam = False
        self.emb_hidden_ch = 16
        self.emb_hidden_layers_num = 1
        self.emb_strategy = 'pxus'
        self.ffn_hidden_ch = 16
        self.ffn_hidden_layers_num = 1
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.dec_strategy = 'pxsf'
        self.gen_factor = 2
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Identity'
        self.use_freq_prior = False
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.use_sh_prior = False
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lru_rank = 32

def build_args(name):
    a = ArgsBase()
    if name == "square_pxus_pxsf_no_priors":
        a.input_size = (144, 144)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = False
        a.use_sh_prior = False
    elif name == "rect_pxus_pxsf_no_priors":
        a.input_size = (144, 288)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = False
        a.use_sh_prior = False
    elif name == "rect_pxus_deconv_no_priors":
        a.input_size = (144, 288)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "deconv"
        a.use_freq_prior = False
        a.use_sh_prior = False
    elif name == "square_pxus_pxsf_freq":
        a.input_size = (144, 144)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = True
        a.freq_rank = 8
        a.freq_gain_init = 0.1
        a.use_sh_prior = False
    elif name == "rect_pxus_pxsf_sh":
        a.input_size = (144, 288)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = False
        a.use_sh_prior = True
        a.sh_Lmax = 6
        a.sh_rank = 8
        a.sh_gain_init = 0.05
    elif name == "rect_pxus_pxsf_both":
        a.input_size = (144, 288)
        a.emb_strategy = "pxus"
        a.hidden_factor = (2, 2)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = True
        a.freq_rank = 8
        a.freq_gain_init = 0.05
        a.use_sh_prior = True
        a.sh_Lmax = 6
        a.sh_rank = 8
        a.sh_gain_init = 0.05
    else:
        pass
    return a

def run_once(name, args, B=1, L=8, out_frames_num=8, lr=1e-3):
    device = pick_device()
    set_seed(123)
    print("[case]", name)
    print("[dev]", device.type)
    model = ConvLRU(args).to(device).train()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[params]", total, trainable)
    H, W = args.input_size
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    y = torch.randn(B, L, args.out_ch, H, W, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    opt.zero_grad()
    out = model(x, mode="p")
    assert out.shape == y.shape
    loss_p = nn.MSELoss()(out, y)
    loss_p.backward()
    opt.step()
    print("[p]loss", float(loss_p.detach().cpu()), "shape", tuple(out.shape))
    del out, loss_p
    torch.cuda.empty_cache() if device.type == "cuda" else None
    opt.zero_grad()
    assert out_frames_num % args.gen_factor == 0
    out_gen_num = out_frames_num // args.gen_factor
    y = torch.randn(B, out_frames_num, args.out_ch, H, W, device=device)
    out_i = model(x, mode="i", out_gen_num=out_gen_num, gen_factor=args.gen_factor)
    assert out_i.shape == y.shape
    loss_i = nn.MSELoss()(out_i, y)
    loss_i.backward()
    opt.step()
    print("[i]loss", float(loss_i.detach().cpu()), "shape", tuple(out_i.shape))
    t1 = time.time()
    print("[time]" + f"{t1 - t0:.3f}s")
    del model, x, y, out_i, loss_i, opt
    torch.cuda.empty_cache() if device.type == "cuda" else None

def main():
    print("[pscan]", pscan_check())
    names = [
        "square_pxus_pxsf_no_priors",
        "rect_pxus_pxsf_no_priors",
        "rect_pxus_deconv_no_priors",
        "square_pxus_pxsf_freq",
        "rect_pxus_pxsf_sh",
        "rect_pxus_pxsf_both",
    ]
    for nm in names:
        a = build_args(nm)
        run_once(nm, a, B=1, L=8, out_frames_num=8, lr=1e-3)

if __name__ == "__main__":
    main()
