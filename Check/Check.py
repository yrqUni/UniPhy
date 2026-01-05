import itertools
import os
import sys
import gc
from types import SimpleNamespace
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "UniPhy")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import ModelUniPhy
from ModelUniPhy import UniPhy, RevINStats

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
        max_velocity=5.0,
    )

def check_p_i_equivalence(model: torch.nn.Module, device: torch.device, args: Any) -> float:
    B, L, H, W = 1, 4, args.input_size[0], args.input_size[1]
    C = args.input_ch

    x_init = torch.randn(B, 1, C, H, W, device=device)

    static_feats = None
    if args.static_ch > 0:
        static_feats = torch.randn(B, args.static_ch, H, W, device=device)

    stats = model.revin.stats(x_init)

    listT_i = torch.ones(B, 1, device=device)
    listT_future = torch.ones(B, L - 1, device=device)

    with torch.no_grad():
        out_i = model(
            x_init,
            mode="i",
            out_gen_num=L,
            listT=listT_i,
            listT_future=listT_future,
            static_feats=static_feats,
            revin_stats=stats
        )

        preds_i_mean = out_i[..., :args.out_ch, :, :]

        p_input_list = [x_init]
        for t in range(L - 1):
            pred_t = preds_i_mean[:, t:t+1]
            if pred_t.shape[2] != C:
                if pred_t.shape[2] > C:
                    pred_t = pred_t[:, :, :C]
                else:
                    diff = C - pred_t.shape[2]
                    zeros = torch.zeros(B, 1, diff, H, W, device=device)
                    pred_t = torch.cat([pred_t, zeros], dim=2)
            p_input_list.append(pred_t)

        x_p = torch.cat(p_input_list, dim=1)

        listT_p = torch.ones(B, L, device=device)

        out_p, _ = model(
            x_p,
            mode="p",
            listT=listT_p,
            static_feats=static_feats,
            revin_stats=stats
        )

        diff = (out_i - out_p).abs().max().item()
        return diff

def run_single_check(args: Any, device: torch.device, force_no_triton: bool = False) -> None:
    original_triton_flag = getattr(ModelUniPhy, "HAS_TRITON", False)

    if force_no_triton:
        ModelUniPhy.HAS_TRITON = False

    try:
        model = UniPhy(args).to(device).eval()
        diff = check_p_i_equivalence(model, device, args)
        if diff > 1e-4:
            print(f"\n[WARN] P/I diff={diff:.2e} static_ch={args.static_ch}")
    finally:
        ModelUniPhy.HAS_TRITON = original_triton_flag

    gc.collect()
    torch.cuda.empty_cache()

def main():
    torch.backends.cudnn.benchmark = False
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running checks on {device_cuda}")

    param_grid = {
        "dist_mode": ["gaussian", "laplace"],
        "diff_mode": ["sobel", "learnable"],
        "Arch": ["unet", "bifpn"],
        "down_mode": ["avg", "shuffle"],
        "static_ch": [0, 4]
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    total = len(combinations)
    print(f"Total configurations: {total}")

    for idx, combo in enumerate(combinations):
        args = get_base_args()
        config_str_parts = []
        for k, v in zip(keys, combo):
            setattr(args, k, v)
            config_str_parts.append(f"{k}={v}")
        config_str = ", ".join(config_str_parts)

        print(f"[{idx+1}/{total}] {config_str} | Triton: ON ...", end="", flush=True)
        run_single_check(args, device_cuda, force_no_triton=False)
        print(" DONE", end="", flush=True)

        print(" | Triton: OFF ...", end="", flush=True)
        run_single_check(args, device_cuda, force_no_triton=True)
        print(" DONE")

    print("\nRunning CPU fallback check on base config...")
    if torch.cuda.is_available():
        args = get_base_args()
        run_single_check(args, torch.device("cpu"), force_no_triton=False)
        print("CPU Check: DONE")

    print("\nAll checks finished.")

if __name__ == "__main__":
    main()

