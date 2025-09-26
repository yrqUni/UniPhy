import os
import sys
import math
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))
from ModelConvLRU import ConvLRU
from pscan import pscan_check

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
        self.input_size = (64, 64)
        self.input_ch = 4
        self.out_ch = 4
        self.emb_ch = 8
        self.convlru_num_blocks = 2
        self.hidden_factor = (2, 2)
        self.use_gate = True
        self.use_cbam = False
        self.emb_hidden_ch = 8
        self.emb_hidden_layers_num = 1
        self.emb_strategy = 'pxus'
        self.ffn_hidden_ch = 8
        self.ffn_hidden_layers_num = 1
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.dec_strategy = 'pxsf'
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Identity'
        self.use_freq_prior = False
        self.freq_rank = 4
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0
        self.lru_rank = 8
        self.lambda_type = "exogenous"
        self.lambda_mlp_hidden = 8

def make_args(name):
    a = ArgsBase()
    if name == "sq_pxsf_exo_noP":
        a.input_size = (64, 64)
        a.lambda_type = "exogenous"
        a.use_freq_prior = False
        a.use_sh_prior = False
        a.dec_strategy = "pxsf"
    elif name == "sq_pxsf_sta_noP":
        a.input_size = (64, 64)
        a.lambda_type = "static"
        a.use_freq_prior = False
        a.use_sh_prior = False
        a.dec_strategy = "pxsf"
    elif name == "sq_pxsf_exo_freq_lin":
        a.input_size = (64, 64)
        a.lambda_type = "exogenous"
        a.use_freq_prior = True
        a.freq_mode = "linear"
        a.freq_rank = 4
        a.freq_gain_init = 0.05
        a.use_sh_prior = False
        a.dec_strategy = "pxsf"
    elif name == "sq_pxsf_exo_freq_exp":
        a.input_size = (64, 64)
        a.lambda_type = "exogenous"
        a.use_freq_prior = True
        a.freq_mode = "exp"
        a.freq_rank = 4
        a.freq_gain_init = 0.03
        a.use_sh_prior = False
        a.dec_strategy = "pxsf"
    elif name == "sq_deconv_sta_noP":
        a.input_size = (64, 64)
        a.lambda_type = "static"
        a.use_freq_prior = False
        a.use_sh_prior = False
        a.dec_strategy = "deconv"
    elif name == "rect_deconv_sta_noP":
        a.input_size = (48, 64)
        a.lambda_type = "static"
        a.use_freq_prior = False
        a.use_sh_prior = False
        a.dec_strategy = "deconv"
    elif name == "rect_pxsf_exo_noP":
        a.input_size = (48, 64)
        a.lambda_type = "exogenous"
        a.use_freq_prior = False
        a.use_sh_prior = False
        a.dec_strategy = "pxsf"
    elif name == "sq_pxsf_exo_sh":
        a.input_size = (64, 64)
        a.lambda_type = "exogenous"
        a.use_freq_prior = False
        a.use_sh_prior = True
        a.sh_Lmax = 4
        a.sh_rank = 4
        a.sh_gain_init = 0.02
        a.dec_strategy = "pxsf"
    return a

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def forward_full_p(model, x, listT=None):
    return model(x, mode="p", listT=listT)

def forward_streaming_p_equiv(model, x, chunk_sizes, listT=None):
    B, L, C, H, W = x.shape
    em = model.embedding
    dm = model.decoder
    m = model.convlru_model
    outs = []
    last_hidden = None
    pos = 0
    for n in chunk_sizes:
        if pos >= L:
            break
        n = min(n, L - pos)
        xe = em(x[:, pos:pos+n])
        listT_slice = None
        if listT is not None:
            listT_slice = listT[:, pos:pos+n]
        xe, last_hidden = m(xe, last_hidden_ins=last_hidden, listT=listT_slice)
        yo = dm(xe)
        outs.append(yo)
        pos += n
    if pos < L:
        xe = em(x[:, pos:])
        listT_slice = listT[:, pos:] if listT is not None else None
        xe, last_hidden = m(xe, last_hidden_ins=last_hidden, listT=listT_slice)
        yo = dm(xe)
        outs.append(yo)
    return torch.cat(outs, dim=1)

def max_err(a, b):
    return float((a - b).abs().max().cpu())

def mae(a, b):
    return float((a - b).abs().mean().cpu())

def effective_tol(y_ref: torch.Tensor, L: int, base_atol: float = 3e-6, base_rtol: float = 1e-5, k_abs: float = 64.0, k_rel: float = 64.0):
    eps = np.finfo(np.float32).eps
    mag = float(y_ref.abs().max().cpu()) + 1.0
    growth = max(1.0, math.log2(L + 1))
    atol_eff = max(base_atol, k_abs * eps * mag * growth)
    rtol_eff = max(base_rtol, k_rel * eps * growth)
    return atol_eff, rtol_eff

def gen_chunk_patterns(L):
    pats = [
        [1]*L,
        [2]*(L//2) + ([L%2] if L%2 else []),
        [3]*(L//3) + ([L%3] if L%3 else []),
        [4]*(L//4) + ([L%4] if L%4 else []),
        [L],
        [2,3,1,4,2, L]
    ]
    return pats

def make_listT_cases(B, L, device, dtype):
    ones = torch.ones(B, L, device=device, dtype=dtype)
    rnd = torch.rand(B, L, device=device, dtype=dtype) * 1.1 + 0.1
    inc = torch.linspace(0.2, 1.6, steps=L, device=device, dtype=dtype).unsqueeze(0).repeat(B,1)
    burst = torch.ones(B, L, device=device, dtype=dtype) * 0.5
    if L >= 4:
        burst[:, L//2:L//2+2] = 1.8
    return [("ones", ones), ("rand", rnd), ("inc", inc), ("burst", burst)]

def list_unused_parameters(model, x, listT):
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    y = model(x, mode="p", listT=listT)
    loss = y.pow(2).mean()
    loss.backward()
    unused = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append(n)
    return unused

def run_equivalence_and_unused(name, args, device, B=1, L=10):
    print(f"[case] {name}")
    model = ConvLRU(args).to(device)
    total, trainable = count_params(model)
    print(f"[params] total={total} trainable={trainable}")
    H, W = args.input_size
    dtype_real = torch.float32
    set_seed(42)
    x = torch.randn(B, L, args.input_ch, H, W, device=device, dtype=dtype_real)
    with torch.no_grad():
        for lt_name, listT in make_listT_cases(B, L, device, dtype_real):
            y_full = forward_full_p(model.eval(), x, listT=listT)
            for pat in gen_chunk_patterns(L):
                y_stream = forward_streaming_p_equiv(model, x, pat, listT=listT)
                e1, m1 = max_err(y_full, y_stream), mae(y_full, y_stream)
                atol_eff, rtol_eff = effective_tol(y_full, L)
                ok = torch.allclose(y_full, y_stream, rtol=rtol_eff, atol=atol_eff)
                print(f"[p~stream] listT={lt_name:<6} pat_len={len(pat):>2} max_err={e1:.3e} mae={m1:.3e} tol(r={rtol_eff:.2e},a={atol_eff:.2e}) {'OK' if ok else 'FAIL'}")
    for lt_name, listT in make_listT_cases(B, L, device, dtype_real):
        unused = list_unused_parameters(model, x, listT)
        print(f"[unused] listT={lt_name:<6} {'none' if len(unused)==0 else ','.join(unused[:5])}")
    print(f"[i-mode] {name}")
    with torch.no_grad():
        for K in [1, 2, 4]:
            listT_hist = torch.rand(B, L, device=device, dtype=dtype_real) * 1.1 + 0.1
            listT_fut = torch.rand(B, K, device=device, dtype=dtype_real) * 1.1 + 0.1
            y_i = model.eval()(x, mode="i", out_gen_num=K, listT=listT_hist, listT_future=listT_fut)
            assert y_i.shape == (B, K, args.out_ch, H, W)
            assert torch.isfinite(y_i).all()
            print(f"[i] K={K} ok")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

def main():
    print("[pscan]", pscan_check())
    device = pick_device()
    print("[device]", device.type)
    cfgs = [
        {"name":"sq_pxsf_exo_noP"},
        {"name":"sq_pxsf_sta_noP"},
        {"name":"sq_pxsf_exo_freq_lin"},
        {"name":"sq_pxsf_exo_freq_exp"},
        {"name":"sq_deconv_sta_noP"},
        {"name":"rect_deconv_sta_noP"},
        {"name":"rect_pxsf_exo_noP"},
        {"name":"sq_pxsf_exo_sh"},
    ]
    for cfg in cfgs:
        args = make_args(cfg["name"])
        run_equivalence_and_unused(cfg["name"], args, device=device, B=1, L=10)

if __name__ == "__main__":
    main()
