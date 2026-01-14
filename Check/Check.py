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

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

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
        conservative_dynamics=False,
        use_pde_refinement=False,
        pde_viscosity=1e-3,
    )

def extract_mu(dist_out: torch.Tensor, out_ch: int) -> torch.Tensor:
    return dist_out[:, :, :out_ch]

def format_cell(text: str, width: int, color: str = "") -> str:
    return f"{color}{text:^{width}}{RESET}"

def check_once(args: Any, device: torch.device, idx: int, total: int) -> float:
    B, L = 1, 4
    H, W = args.input_size
    C = int(args.input_ch)

    try:
        model = UniPhy(args).to(device).eval()
    except Exception as e:
        print(f"{RED}[Init Error]{RESET} {e}")
        return 999.0

    x_init = torch.randn(B, 1, C, H, W, device=device)
    
    listT_i = torch.ones(B, 1, device=device, dtype=x_init.dtype)
    listT_future = torch.ones(B, L - 1, device=device, dtype=x_init.dtype)

    with torch.no_grad():
        out_i_det_cpu, _ = model(
            x_init,
            mode="i",
            out_gen_num=L,
            listT=listT_i,
            listT_future=listT_future,
            sample=False,
        )
        out_i_det = out_i_det_cpu.to(device)
        mu_i = extract_mu(out_i_det, int(args.out_ch))

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

        out_p, _ = model(x_p, mode="p", listT=listT_p)
        mu_p = extract_mu(out_p, int(args.out_ch))
        
        diff_det = (mu_i - mu_p).abs().max().item()
        fin_i = torch.isfinite(out_i_det).all().item()
        fin_p = torch.isfinite(out_p).all().item()

    if not (fin_i and fin_p):
        diff_str = "NaN/Inf"
        diff_color = RED + BOLD
        status_icon = "✗"
    elif diff_det < 1e-6:
        diff_str = f"{diff_det:.1e}"
        diff_color = CYAN
        status_icon = "✓"
    elif diff_det < 1e-4:
        diff_str = f"{diff_det:.1e}"
        diff_color = GREEN
        status_icon = "✓"
    elif diff_det < 1e-2:
        diff_str = f"{diff_det:.1e}"
        diff_color = YELLOW
        status_icon = "~"
    elif diff_det < 1.0:
        diff_str = f"{diff_det:.1e}"
        diff_color = MAGENTA
        status_icon = "!"
    else:
        diff_str = f"{diff_det:.1e}"
        diff_color = RED
        status_icon = "✗"

    cons_str = "Cons" if args.conservative_dynamics else "NonC"
    pde_str = "PDE" if args.use_pde_refinement else "NoPDE"
    arch_str = "UNet" if args.Arch == "unet" else "Flat"
    
    cols = [
        (args.dynamics_mode[:3], 5, CYAN),
        (args.dist_mode[:3], 5, BLUE),
        (arch_str, 6, WHITE),
        (args.down_mode[:4], 5, DIM),
        (args.ConvType[:4], 4, DIM),
        (args.interpolation_mode[:4], 4, DIM),
        (cons_str, 4, DIM),
        (pde_str, 5, DIM),
    ]
    
    cfg_str = f"{DIM}│{RESET}".join([format_cell(c[0], c[1], c[2]) for c in cols])
    
    print(f"{DIM}[{idx:03d}/{total:03d}]{RESET} {diff_color}{status_icon} {diff_str:<8}{RESET} {DIM}│{RESET} {cfg_str}")

    if diff_det >= 1e-2 and (fin_i and fin_p):
        print(f"        {DIM}└─ Step-wise Max Diff:{RESET}")
        for t in range(L):
            step_diff = (mu_i[:, t] - mu_p[:, t]).abs().max().item()
            
            if step_diff < 1e-4: c_s = DIM
            elif step_diff < 1e-2: c_s = YELLOW
            else: c_s = RED
            
            bar_len = int(min(step_diff * 10, 10))
            bar = "█" * bar_len
            
            print(f"           {c_s}t={t}: {step_diff:.1e} {bar}{RESET}")
        print("")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return diff_det

def main():
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{BOLD}UniPhy Consistency Check{RESET}")
    print(f"{DIM}Device: {device}{RESET}\n")

    header_cols = [
        ("Dyn", 5), ("Dist", 5), ("Arch", 6), ("Down", 5), 
        ("Conv", 4), ("Intp", 4), ("Cons", 4), ("PDE", 5)
    ]
    header_str = f"{DIM}│{RESET}".join([f"{h[0]:^{h[1]}}" for h in header_cols])
    print(f"           {BOLD}Diff     {RESET} {DIM}│{RESET} {BOLD}{header_str}{RESET}")
    print(f"{DIM}{'─'*80}{RESET}")

    grid = {
        "dynamics_mode": ["advection", "spectral"],
        "dist_mode": ["gaussian", "mse", "laplace"],
        "Arch": ["unet", "no_unet"],
        "down_mode": ["avg", "shuffle", "conv"],
        "ConvType": ["conv", "dcn"],
        "conservative_dynamics": [False, True],
        "use_pde_refinement": [False, True],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    
    final_combos = []
    for combo in combos:
        t = dict(zip(keys, combo))
        if t["dynamics_mode"] == "advection" and t["conservative_dynamics"]:
            continue
        final_combos.append(combo)
    
    combos = final_combos
    total = len(combos)

    max_diff_global = 0.0

    for idx, combo in enumerate(combos, 1):
        args = get_base_args()
        for k, v in zip(keys, combo):
            setattr(args, k, v)
        
        if args.dynamics_mode == "spectral":
             args.interpolation_mode = "bilinear"
        
        d = check_once(args, device, idx, total)
        if d < 999.0:
            max_diff_global = max(max_diff_global, d)

    print(f"\n{DIM}{'─'*80}{RESET}")
    print(f"{BOLD}Global Max Diff:{RESET} {max_diff_global:.2e}\n")

if __name__ == "__main__":
    main()

