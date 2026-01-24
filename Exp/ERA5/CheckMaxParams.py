import argparse
import csv
import gc
import os
import sys

import numpy as np
import torch

sys.path.append("/nfs/UniPhy/Model/UniPhy")
try:
    from ModelUniPhy import UniPhyModel
except ImportError:
    sys.exit(1)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--H", type=int, default=721)
    parser.add_argument("--W", type=int, default=1440)
    parser.add_argument("--C", type=int, default=30)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--min_search_dim", type=int, default=64)
    parser.add_argument("--max_search_dim", type=int, default=1536)
    parser.add_argument("--dim_step", type=int, default=32)
    parser.add_argument("--search_depths", type=int, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--search_experts", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--save_csv", type=str, default="max_params_result.csv")
    return parser.parse_args()


def format_count(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    else:
        return f"{num}"


def check_config_fit(args, dim, depth, experts, device):
    model = None
    optimizer = None
    scaler = None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        model = UniPhyModel(
            in_channels=args.C,
            out_channels=args.C,
            embed_dim=dim,
            expand=args.expand,
            num_experts=experts,
            depth=depth,
            patch_size=args.patch_size,
            img_height=args.H,
            img_width=args.W,
        ).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.rand(args.bs, args.T, device=device)
        times = []
        for step in range(args.steps):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x, dt)
                loss = out.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            end_evt.record()
            torch.cuda.synchronize()
            if step > 0:
                times.append(start_evt.elapsed_time(end_evt))
        avg_time = np.mean(times) if times else 0
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        del model, optimizer, scaler, x, dt, out, loss
        torch.cuda.empty_cache()
        return True, num_params, avg_time, peak_mem
    except Exception:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scaler is not None:
            del scaler
        torch.cuda.empty_cache()
        return False, 0, 0, 0


def binary_search_max_dim(args, depth, experts, device):
    low = args.min_search_dim
    high = args.max_search_dim
    best_result = None
    low = (low // args.dim_step) * args.dim_step
    high = (high // args.dim_step) * args.dim_step
    print(f"Scanning Exp {experts:<2} | Dep {depth:<2} | Range [{low}, {high}]...", end="\r")
    while low <= high:
        mid = (low + high) // 2
        mid = (mid // args.dim_step) * args.dim_step
        if mid < args.min_search_dim:
            mid = args.min_search_dim
        success, params, t, mem = check_config_fit(args, mid, depth, experts, device)
        if success:
            best_result = (mid, params, t, mem)
            low = mid + args.dim_step
            print(f"Scanning Exp {experts:<2} | Dep {depth:<2} | Dim {mid} \033[92m[OK]\033[0m   ", end="\r")
        else:
            high = mid - args.dim_step
            print(f"Scanning Exp {experts:<2} | Dep {depth:<2} | Dim {mid} \033[91m[OOM]\033[0m  ", end="\r")
    return best_result


def main():
    if int(os.environ.get("RANK", 0)) != 0:
        return
    args = get_args()
    if not torch.cuda.is_available():
        sys.exit(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"\n{'='*100}")
    print(f" UniPhy Max Parameter Search (Binary Search)")
    print(f" Input: {args.bs}x{args.T}x{args.C}x{args.H}x{args.W}")
    print(f" Experts: {args.search_experts} | Depths: {args.search_depths}")
    print(f" AMP: {args.amp} | Steps: {args.steps}")
    print(f"{'='*100}\n")
    results = []
    for experts in sorted(args.search_experts):
        for depth in sorted(args.search_depths):
            res = binary_search_max_dim(args, depth, experts, device)
            if res:
                best_dim, best_params, best_time, best_mem = res
                print(
                    f" Exp {experts:<2} | Dep {depth:<2} | Dim {best_dim:<4} | Params {format_count(best_params):<7} | Time {best_time:>6.1f}ms | Mem {best_mem:>5.2f}GB \033[92m[PASS]\033[0m"
                )
                results.append((best_params, experts, depth, best_dim, best_time, best_mem))
            else:
                print(
                    f" Exp {experts:<2} | Dep {depth:<2} | {'-'*4} | {'-'*14} | {'-'*10} | {'-'*9} \033[91m[FAIL]\033[0m"
                )
    print("\n" + "=" * 110)
    print(
        f"{'Rank':<4} | {'Params':<10} | {'Experts':<7} | {'Depth':<5} | {'Max Dim':<8} | {'Step Time':<12} | {'Peak VRAM':<10}"
    )
    print("-" * 110)
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)
    for i, (params, experts, depth, dim, t, m) in enumerate(sorted_res):
        print(
            f"{i+1:<4} | {format_count(params):<10} | {experts:<7} | {depth:<5} | {dim:<8} | {t:>9.1f} ms | {m:>7.2f} GB"
        )
    print("=" * 110 + "\n")
    if args.save_csv:
        try:
            with open(args.save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Rank", "Params", "Experts", "Depth", "Max_Dim", "Time_ms", "Memory_GB"]
                )
                for i, r in enumerate(sorted_res):
                    writer.writerow(
                        [
                            i + 1,
                            format_count(r[0]),
                            r[1],
                            r[2],
                            r[3],
                            f"{r[4]:.1f}",
                            f"{r[5]:.2f}",
                        ]
                    )
            print(f"Results saved to {args.save_csv}")
        except:
            pass


if __name__ == "__main__":
    main()
    