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
from ModelUniPhy import UniPhy


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


def check_once(args: Any, device: torch.device, tol: float = 1e-4) -> None:
    B, L = 1, 4
    H, W = args.input_size
    C = args.input_ch

    model = UniPhy(args).to(device).eval()

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
            revin_stats=stats,
        )

        preds_i = out_i[..., :args.out_ch, :, :]

        p_inputs = [x_init]
        for t in range(L - 1):
            pred_t = preds_i[:, t:t+1]
            if pred_t.shape[2] != C:
                if pred_t.shape[2] > C:
                    pred_t = pred_t[:, :, :C]
                else:
                    pad = torch.zeros(B, 1, C - pred_t.shape[2], H, W, device=device)
                    pred_t = torch.cat([pred_t, pad], dim=2)
            p_inputs.append(pred_t)

        x_p = torch.cat(p_inputs, dim=1)
        listT_p = torch.ones(B, L, device=device)

        out_p, _ = model(
            x_p,
            mode="p",
            listT=listT_p,
            static_feats=static_feats,
            revin_stats=stats,
        )

        diff = (out_i - out_p).abs().max().item()

    cfg = ", ".join(f"{k}={getattr(args, k)}" for k in vars(args))
    if diff <= tol:
        print(f"[PASS] diff={diff:.3e} | {cfg}")
    else:
        print(f"[WARN] diff={diff:.3e} | {cfg}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    grid = {
        "dist_mode": ["gaussian", "laplace"],
        "diff_mode": ["sobel", "learnable"],
        "Arch": ["unet", "bifpn"],
        "down_mode": ["avg", "shuffle"],
        "static_ch": [0, 4],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    print(f"[Total cases] {len(combos)}")

    for idx, combo in enumerate(combos, 1):
        args = get_base_args()
        for k, v in zip(keys, combo):
            setattr(args, k, v)

        print(f"\n[{idx}/{len(combos)}] RUN")
        check_once(args, device)


if __name__ == "__main__":
    main()

