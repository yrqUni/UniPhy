import itertools
import os
import sys
import gc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "UniPhy")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import ModelUniPhy
from ModelUniPhy import UniPhy, RevINStats

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

@dataclass
class CaseResult:
    idx: int
    total: int
    device: str
    triton: str
    dist_mode: str
    diff_mode: str
    Arch: str
    down_mode: str
    static_ch: int
    diff: Optional[float]
    tol: float
    ok: bool
    error: Optional[str]

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
            revin_stats=stats,
        )

        preds_i_mean = out_i[..., :args.out_ch, :, :]

        p_input_list = [x_init]
        for t in range(L - 1):
            pred_t = preds_i_mean[:, t : t + 1]
            if pred_t.shape[2] != C:
                if pred_t.shape[2] > C:
                    pred_t = pred_t[:, :, :C]
                else:
                    diff_ch = C - pred_t.shape[2]
                    zeros = torch.zeros(B, 1, diff_ch, H, W, device=device)
                    pred_t = torch.cat([pred_t, zeros], dim=2)
            p_input_list.append(pred_t)

        x_p = torch.cat(p_input_list, dim=1)
        listT_p = torch.ones(B, L, device=device)

        out_p, _ = model(
            x_p,
            mode="p",
            listT=listT_p,
            static_feats=static_feats,
            revin_stats=stats,
        )

        return (out_i - out_p).abs().max().item()

def run_single_check(args: Any, device: torch.device, force_no_triton: bool) -> float:
    original_triton_flag = getattr(ModelUniPhy, "HAS_TRITON", False)
    if force_no_triton:
        ModelUniPhy.HAS_TRITON = False
    try:
        model = UniPhy(args).to(device).eval()
        diff = check_p_i_equivalence(model, device, args)
        return diff
    finally:
        ModelUniPhy.HAS_TRITON = original_triton_flag
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def main():
    torch.backends.cudnn.benchmark = False
    console = Console()
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tol = 1e-4

    param_grid = {
        "dist_mode": ["gaussian", "laplace"],
        "diff_mode": ["sobel", "learnable"],
        "Arch": ["unet", "bifpn"],
        "down_mode": ["avg", "shuffle"],
        "static_ch": [0, 4],
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    total_cfg = len(combinations)
    total_cases = total_cfg * 2 + (1 if torch.cuda.is_available() else 0)

    console.print(f"[bold]Device:[/bold] {device_cuda}")
    console.print(f"[bold]Total configurations:[/bold] {total_cfg} (each runs Triton ON/OFF)")
    console.print(f"[bold]Total cases:[/bold] {total_cases}")

    results: List[CaseResult] = []
    case_idx = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("Running checks", total=total_cases)

        for cfg_idx, combo in enumerate(combinations, start=1):
            args = get_base_args()
            for k, v in zip(keys, combo):
                setattr(args, k, v)

            for force_no_triton in (False, True):
                case_idx += 1
                triton_str = "OFF" if force_no_triton else "ON"
                desc = f"[{case_idx}/{total_cases}] cfg {cfg_idx}/{total_cfg} Triton={triton_str} static_ch={args.static_ch} Arch={args.Arch} down={args.down_mode} dist={args.dist_mode} diff={args.diff_mode}"
                progress.update(task, description=desc)

                diff = None
                err = None
                try:
                    diff = run_single_check(args, device_cuda, force_no_triton=force_no_triton)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"

                ok = (diff is not None) and (diff <= tol)
                results.append(
                    CaseResult(
                        idx=case_idx,
                        total=total_cases,
                        device=str(device_cuda),
                        triton=triton_str,
                        dist_mode=args.dist_mode,
                        diff_mode=args.diff_mode,
                        Arch=args.Arch,
                        down_mode=args.down_mode,
                        static_ch=int(args.static_ch),
                        diff=diff,
                        tol=tol,
                        ok=ok,
                        error=err,
                    )
                )
                if err is not None:
                    console.print(f"[yellow][WARN][/yellow] cfg={cfg_idx}/{total_cfg} Triton={triton_str} static_ch={args.static_ch} error={err}")
                elif diff is not None and diff > tol:
                    console.print(f"[yellow][WARN][/yellow] cfg={cfg_idx}/{total_cfg} Triton={triton_str} static_ch={args.static_ch} diff={diff:.2e} tol={tol:.1e}")

                progress.advance(task, 1)

        if torch.cuda.is_available():
            case_idx += 1
            args = get_base_args()
            progress.update(task, description=f"[{case_idx}/{total_cases}] CPU fallback base config")
            diff = None
            err = None
            try:
                diff = run_single_check(args, torch.device("cpu"), force_no_triton=False)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            ok = (diff is not None) and (diff <= tol)
            results.append(
                CaseResult(
                    idx=case_idx,
                    total=total_cases,
                    device="cpu",
                    triton="N/A",
                    dist_mode=args.dist_mode,
                    diff_mode=args.diff_mode,
                    Arch=args.Arch,
                    down_mode=args.down_mode,
                    static_ch=int(args.static_ch),
                    diff=diff,
                    tol=tol,
                    ok=ok,
                    error=err,
                )
            )
            if err is not None:
                console.print(f"[yellow][WARN][/yellow] CPU base config error={err}")
            elif diff is not None and diff > tol:
                console.print(f"[yellow][WARN][/yellow] CPU base config diff={diff:.2e} tol={tol:.1e}")
            progress.advance(task, 1)

    total_warn = sum(1 for r in results if (r.error is None and r.diff is not None and r.diff > tol))
    total_err = sum(1 for r in results if r.error is not None)
    total_ok = sum(1 for r in results if r.error is None and r.diff is not None and r.diff <= tol)

    table = Table(title="Check Summary", show_lines=False)
    table.add_column("Cases", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("WARN(diff>tol)", justify="right")
    table.add_column("WARN(error)", justify="right")
    table.add_column("tol", justify="right")
    table.add_row(str(len(results)), str(total_ok), str(total_warn), str(total_err), f"{tol:.1e}")
    console.print()
    console.print(table)

    table2 = Table(title="Worst Cases", show_lines=False)
    table2.add_column("idx", justify="right")
    table2.add_column("device")
    table2.add_column("triton")
    table2.add_column("static_ch", justify="right")
    table2.add_column("Arch")
    table2.add_column("down")
    table2.add_column("dist")
    table2.add_column("diff")
    table2.add_column("status")

    scored = []
    for r in results:
        if r.error is not None:
            scored.append((float("inf"), r))
        elif r.diff is None:
            scored.append((float("inf"), r))
        else:
            scored.append((r.diff, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    topk = min(10, len(scored))
    for _, r in scored[:topk]:
        if r.error is not None:
            d = "ERR"
            status = "WARN"
        elif r.diff is None:
            d = "N/A"
            status = "WARN"
        else:
            d = f"{r.diff:.2e}"
            status = "OK" if r.diff <= tol else "WARN"
        table2.add_row(
            str(r.idx),
            r.device,
            r.triton,
            str(r.static_ch),
            r.Arch,
            r.down_mode,
            r.dist_mode,
            d,
            status,
        )

    console.print()
    console.print(table2)

if __name__ == "__main__":
    main()

