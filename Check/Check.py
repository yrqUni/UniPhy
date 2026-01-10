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

try:
    from ModelUniPhy import UniPhy
except ImportError:
    print(f"Error: Cannot import ModelUniPhy from {MODEL_DIR}")
    sys.exit(1)

def get_base_args() -> Any:
    return SimpleNamespace(
        input_ch=2,
        out_ch=2,
        input_size=(32, 32),
        emb_ch=16,
        hidden_factor=(2, 2),
        ConvType="conv",
        Arch="unet",
        dist_mode="gaussian",
        convlru_num_blocks=2,
        down_mode="avg",
        ffn_ratio=2.0,
        lru_rank=8,
        koopman_use_noise=False,
        koopman_noise_scale=1.0,
        dt_ref=1.0,
        inj_k=2.0,
        dynamics_mode="advection",
        spectral_modes_h=12,
        spectral_modes_w=12,
        interpolation_mode="bilinear",
        pscan_use_decay=True,
        pscan_use_residual=True,
        pscan_chunk_size=32,
    )

def extract_mu(dist_out: torch.Tensor, out_ch: int) -> torch.Tensor:
    return dist_out[:, :, :out_ch]

def check_once(args: Any, device: torch.device) -> None:
    B, L = 1, 4
    H, W = args.input_size
    C = int(args.input_ch)

    model = UniPhy(args).to(device).eval()

    x_init = torch.randn(B, 1, C, H, W, device=device)
    
    stats = model.revin.stats(x_init)
    
    listT_i = torch.ones(B, 1, device=device, dtype=x_init.dtype)
    listT_future = torch.ones(B, L - 1, device=device, dtype=x_init.dtype)

    with torch.no_grad():
        out_i, _ = model(
            x_init,
            mode="i",
            out_gen_num=L,
            listT=listT_i,
            listT_future=listT_future,
            revin_stats=stats,
        )

        mu_i = extract_mu(out_i, int(args.out_ch))

        p_inputs = [x_init]
        for t in range(L - 1):
            pred_t = mu_i[:, t : t + 1]
            if pred_t.shape[2] != C:
                if pred_t.shape[2] > C:
                    pred_t = pred_t[:, :, :C]
                else:
                    pad = torch.zeros(B, 1, C - pred_t.shape[2], H, W, device=device, dtype=pred_t.dtype)
                    pred_t = torch.cat([pred_t, pad], dim=2)
            p_inputs.append(pred_t)

        x_p = torch.cat(p_inputs, dim=1)
        listT_p = torch.ones(B, L, device=device, dtype=x_init.dtype)

        out_p, _ = model(
            x_p,
            mode="p",
            listT=listT_p,
            revin_stats=stats,
        )

        mu_p = extract_mu(out_p, int(args.out_ch))
        
        diff = (mu_i - mu_p).abs().max().item()
        fin_i = torch.isfinite(out_i).all().item()
        fin_p = torch.isfinite(out_p).all().item()

    threshold = 1e-3
    status = "PASS" if (fin_i and fin_p and diff < threshold) else "FAIL"
    
    keys = ["dynamics_mode", "down_mode"]
    cfg_str = " | ".join(f"{k}={getattr(args, k)}" for k in keys)

    print(f"[{status}] Max Diff={diff:.3e} | {cfg_str}")
    
    if status == "FAIL":
        print("    -> Step-wise Max Diffs:")
        for t in range(L):
            step_diff = (mu_i[:, t] - mu_p[:, t]).abs().max().item()
            print(f"       Step {t}: {step_diff:.3e}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def main():
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    grid = {
        "dynamics_mode": ["advection", "spectral"],
        "down_mode": ["avg", "shuffle"],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"[Total cases] {len(combos)}\n")

    for idx, combo in enumerate(combos, 1):
        args = get_base_args()
        for k, v in zip(keys, combo):
            setattr(args, k, v)
        check_once(args, device)

if __name__ == "__main__":
    main()

