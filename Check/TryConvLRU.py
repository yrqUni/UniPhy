import os
import sys
import argparse
import math
import torch
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.freq_amp_mode = "linear"
        self.use_sh_prior = False
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lru_rank = 32
        self.dynamic_lambda = True
        self.lambda_mix = 1.0
        self.dyn_r_min = 0.80
        self.dyn_r_max = 0.99

def build_args(name, dynamic_lambda=True, lambda_mix=1.0, dyn_r_min=0.80, dyn_r_max=0.99):
    a = ArgsBase()
    a.dynamic_lambda = bool(dynamic_lambda)
    a.lambda_mix = float(lambda_mix)
    a.dyn_r_min = float(dyn_r_min)
    a.dyn_r_max = float(dyn_r_max)
    if name == "square_no_priors":
        a.input_size = (144, 144)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = False
        a.use_sh_prior = False
    elif name == "rect_no_priors":
        a.input_size = (144, 288)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = False
        a.use_sh_prior = False
    elif name == "square_freq_linear":
        a.input_size = (144, 144)
        a.dec_strategy = "pxsf"
        a.use_freq_prior = True
        a.freq_amp_mode = "linear"
        a.freq_rank = 8
        a.freq_gain_init = 0.05
        a.use_sh_prior = False
    return a

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def forward_full_p(model, x):
    return model(x, mode="p")

@torch.no_grad()
def forward_streaming_p_equiv(model, x, chunk):
    B, L, C, H, W = x.shape
    em = model.embedding
    dm = model.decoder
    m = model.convlru_model
    outs = []
    last_hidden = None
    pos = 0
    while pos < L:
        n = min(chunk, L - pos)
        xe = em(x[:, pos:pos+n])
        xe, last_hidden = m(xe, last_hidden_ins=last_hidden)
        yo = dm(xe)
        outs.append(yo)
        pos += n
    return torch.cat(outs, dim=1)

@torch.no_grad()
def forward_i_single_block(model, x):
    B, L, C, H, W = x.shape
    return model(x, mode="i", out_gen_num=1, gen_factor=L)

def max_err(a, b):
    return float((a - b).abs().max().cpu())

def mae(a, b):
    return float((a - b).abs().mean().cpu())

def effective_tol(y_ref: torch.Tensor, L: int, base_atol: float, base_rtol: float, k_abs: float = 64.0, k_rel: float = 64.0):
    eps = np.finfo(np.float32).eps
    mag = float(y_ref.abs().max().cpu()) + 1.0
    growth = max(1.0, math.log2(L + 1))
    atol_eff = max(base_atol, k_abs * eps * mag * growth)
    rtol_eff = max(base_rtol, k_rel * eps * growth)
    return atol_eff, rtol_eff

def run_case(name, args, B=1, L=12, chunks=(3,4,6), atol=3e-6, rtol=1e-5, seeds=(123,456), device=None):
    if device is None:
        device = pick_device()
    print("[case]", name)
    print("[dev]", device.type)
    model = ConvLRU(args).to(device).eval()
    total, trainable = count_params(model)
    print("[params]", total, trainable)
    H, W = args.input_size
    dtype_real = torch.float32
    for sd in seeds:
        set_seed(sd)
        x = torch.randn(B, L, args.input_ch, H, W, device=device, dtype=dtype_real)
        y_full = forward_full_p(model, x)
        y_i1 = forward_i_single_block(model, x)
        e2, m2 = max_err(y_full, y_i1), mae(y_full, y_i1)
        ok_i = torch.allclose(y_full, y_i1, rtol=rtol, atol=atol)
        print(f"[p vs i(1-block)] seed={sd} max_err {e2:.3e} mae {m2:.3e} | tol(rtol={rtol:.2e}, atol={atol:.2e}) {'OK' if ok_i else 'FAIL'}")
        assert ok_i, "p vs i(1-block) mismatch"
        for ck in chunks:
            y_stream = forward_streaming_p_equiv(model, x, chunk=ck)
            e1, m1 = max_err(y_full, y_stream), mae(y_full, y_stream)
            atol_eff, rtol_eff = effective_tol(y_full, L, atol, rtol)
            ok = torch.allclose(y_full, y_stream, rtol=rtol_eff, atol=atol_eff)
            print(f"[p vs p-stream]   seed={sd} chunk={ck} max_err {e1:.3e} mae {m1:.3e} | tol(rtol={rtol_eff:.2e}, atol={atol_eff:.2e}) {'OK' if ok else 'FAIL'}")
            assert ok, f"p vs p-stream mismatch (chunk={ck}, seed={sd}, max_err={e1:.3e}, tol_used rtol={rtol_eff:.2e}, atol={atol_eff:.2e})"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

def main():
    print("[pscan]", pscan_check())
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=3e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--chunks", type=int, nargs="+", default=[3,4,6])
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--dyn_r_min", type=float, default=0.80)
    parser.add_argument("--dyn_r_max", type=float, default=0.99)
    args_cli = parser.parse_args()
    names = ["square_no_priors", "rect_no_priors", "square_freq_linear"]
    for nm in names:
        a = build_args(
            nm,
            dynamic_lambda=args_cli.dynamic,
            lambda_mix=args_cli.lambda_mix,
            dyn_r_min=args_cli.dyn_r_min,
            dyn_r_max=args_cli.dyn_r_max
        )
        run_case(
            nm, a, B=args_cli.B, L=args_cli.L,
            chunks=tuple(args_cli.chunks),
            atol=args_cli.atol, rtol=args_cli.rtol,
            seeds=(123,456),
            device=pick_device()
        )

if __name__ == "__main__":
    main()
