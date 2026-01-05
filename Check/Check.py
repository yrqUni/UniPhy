import itertools
import os
import sys
import gc
from types import SimpleNamespace
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU

def get_base_args() -> Any:
    return SimpleNamespace(
        input_ch=2,
        out_ch=2,
        input_size=(32, 32),
        emb_ch=16,
        static_ch=0,
        hidden_factor=(2, 2),
        ConvType="conv",
        Arch="unet",
        dist_mode="gaussian",
        diff_mode="sobel",
        convlru_num_blocks=2,
        down_mode="avg",
        ffn_ratio=2.0,
        lru_rank=8,
        koopman_use_noise=False,
        koopman_noise_scale=1.0,
        learnable_init_state=True,
        dt_ref=1.0,
        inj_k=2.0,
    )

def run_single_check(args: Any, device: torch.device) -> None:
    model = ConvLRU(args).to(device).eval()
    
    B, L, C, H, W = 1, 4, args.input_ch, args.input_size[0], args.input_size[1]
    x = torch.randn(B, L, C, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    
    static_feats = None
    if args.static_ch > 0:
        static_feats = torch.randn(B, args.static_ch, H, W, device=device)
    
    with torch.no_grad():
        out_p, _ = model(x, mode="p", listT=listT, static_feats=static_feats)
        
        expected_ch = args.out_ch * 2
        if args.dist_mode == "mdn":
            expected_ch = args.out_ch * 3 * 3
        
        if out_p.shape != (B, L, expected_ch, H, W):
            raise RuntimeError(f"P-mode shape mismatch: expected {(B, L, expected_ch, H, W)}, got {out_p.shape}")

        x_init = x[:, :1]
        out_gen_num = 3
        listT_future = torch.ones(B, out_gen_num - 1, device=device)
        
        out_i = model(
            x_init, 
            mode="i", 
            out_gen_num=out_gen_num, 
            listT=listT[:, :1], 
            listT_future=listT_future,
            static_feats=static_feats
        )
        
        if out_i.shape != (B, out_gen_num, expected_ch, H, W):
            raise RuntimeError(f"I-mode shape mismatch: expected {(B, out_gen_num, expected_ch, H, W)}, got {out_i.shape}")

    del model, x, listT, static_feats, out_p, out_i
    gc.collect()
    torch.cuda.empty_cache()

def main():
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running checks on {device}")

    param_grid = {
        "dist_mode": ["gaussian", "laplace", "mdn"],
        "diff_mode": ["sobel", "learnable", "diffconv"],
        "Arch": ["unet", "bifpn"],
        "down_mode": ["avg", "shuffle"],
        "static_ch": [0, 4]
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    total = len(combinations)
    print(f"Total configurations to test: {total}")

    for idx, combo in enumerate(combinations):
        args = get_base_args()
        config_str_parts = []
        
        for k, v in zip(keys, combo):
            setattr(args, k, v)
            config_str_parts.append(f"{k}={v}")
        
        config_str = ", ".join(config_str_parts)
        print(f"[{idx+1}/{total}] Testing: {config_str} ...", end="", flush=True)
        
        try:
            run_single_check(args, device)
            print(" PASS")
        except Exception as e:
            print(f" FAIL")
            print(f"Error details: {e}")
            sys.exit(1)

    print("\nAll hyperparameter combinations check finished.")

if __name__ == "__main__":
    main()

