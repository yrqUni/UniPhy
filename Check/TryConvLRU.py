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
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Identity'
        self.use_freq_prior = False
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = False
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lru_rank = 32
        self.lambda_type = "exogenous"
        self.lambda_mlp_hidden = 16

def make_cfgs():
    cfgs = []
    a = ArgsBase()
    a.input_size = (144, 144)
    a.dec_strategy = "pxsf"
    a.lambda_type = "exogenous"
    a.use_freq_prior = False
    a.use_sh_prior = False
    cfgs.append(("sq_pxsf_exo_noP", a))
    a = ArgsBase()
    a.input_size = (144, 144)
    a.dec_strategy = "pxsf"
    a.lambda_type = "static"
    a.use_freq_prior = False
    a.use_sh_prior = False
    cfgs.append(("sq_pxsf_sta_noP", a))
    a = ArgsBase()
    a.input_size = (144, 144)
    a.dec_strategy = "pxsf"
    a.lambda_type = "exogenous"
    a.use_freq_prior = True
    a.freq_mode = "linear"
    a.freq_gain_init = 0.05
    a.use_sh_prior = False
    cfgs.append(("sq_pxsf_exo_freq_lin", a))
    a = ArgsBase()
    a.input_size = (144, 144)
    a.dec_strategy = "pxsf"
    a.lambda_type = "exogenous"
    a.use_freq_prior = True
    a.freq_mode = "exp"
    a.freq_gain_init = 0.02
    a.use_sh_prior = True
    cfgs.append(("sq_pxsf_exo_freq_exp_sh", a))
    a = ArgsBase()
    a.input_size = (120, 200)
    a.dec_strategy = "pxsf"
    a.lambda_type = "exogenous"
    a.use_freq_prior = False
    a.use_sh_prior = False
    cfgs.append(("rect_pxsf_exo_noP", a))
    a = ArgsBase()
    a.input_size = (120, 200)
    a.dec_strategy = "pxsf"
    a.lambda_type = "static"
    a.use_freq_prior = True
    a.freq_mode = "linear"
    a.freq_gain_init = 0.03
    a.use_sh_prior = True
    cfgs.append(("rect_pxsf_sta_freq_sh", a))
    a = ArgsBase()
    a.input_size = (96, 96)
    a.dec_strategy = "deconv"
    a.dec_hidden_layers_num = 0
    a.lambda_type = "static"
    a.use_freq_prior = False
    a.use_sh_prior = False
    cfgs.append(("sq_deconv_sta_noP", a))
    a = ArgsBase()
    a.input_size = (96, 144)
    a.dec_strategy = "deconv"
    a.dec_hidden_layers_num = 0
    a.lambda_type = "exogenous"
    a.use_freq_prior = True
    a.freq_mode = "linear"
    a.freq_gain_init = 0.05
    a.use_sh_prior = True
    a.use_gate = False
    cfgs.append(("rect_deconv_exo_freq_sh_nogate", a))
    return cfgs

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def forward_full_p(model, x, listT=None):
    return model(x, mode="p", listT=listT)

@torch.no_grad()
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
        listT_slice = listT[:, pos:pos+n] if listT is not None else None
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

def list_unused_parameters(model, x, listT, mode="p"):
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if mode == "p":
        y = model(x, mode="p", listT=listT)
        loss = y.sum()
    else:
        B, L, C, H, W = x.shape
        K = 3
        listT_future = torch.rand(B, K, device=x.device, dtype=x.dtype) * 1.2 + 0.1
        y = model(x, mode="i", out_gen_num=K, listT=listT, listT_future=listT_future)
        loss = y.sum()
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
    set_seed(2024)
    x = torch.randn(B, L, args.input_ch, H, W, device=device, dtype=dtype_real)
    lt_cases = make_listT_cases(B, L, device, dtype_real)
    for lt_name, listT in lt_cases:
        y_full = forward_full_p(model.eval(), x, listT=listT)
        for pat in gen_chunk_patterns(L):
            y_stream = forward_streaming_p_equiv(model.eval(), x, pat, listT=listT)
            e1, m1 = max_err(y_full, y_stream), mae(y_full, y_stream)
            atol_eff, rtol_eff = effective_tol(y_full, L)
            ok = torch.allclose(y_full, y_stream, rtol=rtol_eff, atol=atol_eff)
            print(f"[p~stream] listT={lt_name:<6} pat_len={len(pat):>2} max_err={e1:.3e} mae={m1:.3e} tol(r={rtol_eff:.2e},a={atol_eff:.2e}) {'OK' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError("p vs streaming mismatch")
        unused_p = list_unused_parameters(model, x, listT, mode="p")
        print(f"[unused-p] listT={lt_name:<6} {'none' if len(unused_p)==0 else ', '.join(unused_p[:8]) + (' ...' if len(unused_p)>8 else '')}")
    half = L // 2
    listT_half = lt_cases[0][1][:, :half]
    unused_i = list_unused_parameters(model, x[:, :half], listT=listT_half, mode="i")
    print(f"[unused-i] {'none' if len(unused_i)==0 else ', '.join(unused_i[:12]) + (' ...' if len(unused_i)>12 else '')}")

def main():
    print("[pscan]", pscan_check())
    device = pick_device()
    print("[device]", device.type)
    cfgs = make_cfgs()
    for name, args in cfgs:
        run_equivalence_and_unused(name, args, device=device, B=1, L=10)

if __name__ == "__main__":
    main()
