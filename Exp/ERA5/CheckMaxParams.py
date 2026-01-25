import argparse
import csv
import gc
import os
import sys

import numpy as np
import torch

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel

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
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--min_search_dim", type=int, default=64)
    parser.add_argument("--max_search_dim", type=int, default=1024)
    parser.add_argument("--dim_step", type=int, default=32)
    parser.add_argument("--search_depths", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--search_experts", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=str, default="max_params_results.csv")
    return parser.parse_args()


def check_config_fit(args, embed_dim, depth, experts, device):
    model = None
    optimizer = None

    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        model = UniPhyModel(
            in_channels=args.C,
            out_channels=args.C,
            embed_dim=embed_dim,
            expand=args.expand,
            num_experts=experts,
            depth=depth,
            patch_size=args.patch_size,
            img_height=args.H,
            img_width=args.W,
            dt_ref=6.0,
            sde_mode="det",
            init_noise_scale=0.01,
            max_growth_rate=0.3,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.ones(args.bs, args.T, device=device) * 6.0

        times = []
        for step in range(args.steps):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_evt.record()

            out = model(x, dt)
            if out.is_complex():
                loss = out.abs().mean()
            else:
                loss = out.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            end_evt.record()
            torch.cuda.synchronize()

            if step > 0:
                times.append(start_evt.elapsed_time(end_evt))

        avg_time = np.mean(times) if times else 0
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        del model, optimizer, x, dt, out, loss
        torch.cuda.empty_cache()
        gc.collect()

        return True, num_params, avg_time, peak_mem

    except Exception as e:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0, 0, 0


def binary_search_max_dim(args, depth, experts, device):
    low = args.min_search_dim
    high = args.max_search_dim
    best_result = None

    low = (low // args.dim_step) * args.dim_step
    high = (high // args.dim_step) * args.dim_step

    print(f"Scanning Exp {experts:<2} | Dep {depth:<2} | Range [{low}, {high}]...")

    while low <= high:
        mid = (low + high) // 2
        mid = (mid // args.dim_step) * args.dim_step

        if mid < args.min_search_dim:
            mid = args.min_search_dim

        success, params, t, mem = check_config_fit(args, mid, depth, experts, device)

        if success:
            best_result = {
                "embed_dim": mid,
                "depth": depth,
                "experts": experts,
                "params": params,
                "time_ms": t,
                "mem_gb": mem,
            }
            low = mid + args.dim_step
            print(f"  Dim {mid}: OK | Params {params/1e6:.1f}M | Mem {mem:.2f}GB | Time {t:.1f}ms")
        else:
            high = mid - args.dim_step
            print(f"  Dim {mid}: OOM")

    return best_result


def main():
    args = get_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 70)
    print("UniPhy Max Parameters Search")
    print("=" * 70)
    print(f"Batch Size: {args.bs}")
    print(f"Input Shape: ({args.T}, {args.C}, {args.H}, {args.W})")
    print(f"Patch Size: {args.patch_size}")
    print(f"Search Depths: {args.search_depths}")
    print(f"Search Experts: {args.search_experts}")
    print(f"Dim Range: [{args.min_search_dim}, {args.max_search_dim}]")
    print("=" * 70)

    results = []

    for depth in args.search_depths:
        for experts in args.search_experts:
            result = binary_search_max_dim(args, depth, experts, device)
            if result:
                results.append(result)
                print(f"Best for D={depth}, E={experts}: dim={result['embed_dim']}, "
                      f"params={result['params']/1e6:.1f}M, mem={result['mem_gb']:.2f}GB")
            else:
                print(f"No valid config found for D={depth}, E={experts}")
            print("-" * 70)

    if results:
        results.sort(key=lambda x: x["params"], reverse=True)

        print("\n" + "=" * 70)
        print("Results Summary (sorted by params)")
        print("=" * 70)
        print(f"{'Depth':<8}{'Experts':<10}{'Dim':<8}{'Params':<12}{'Memory':<10}{'Time':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['depth']:<8}{r['experts']:<10}{r['embed_dim']:<8}"
                  f"{r['params']/1e6:.1f}M{'':<6}{r['mem_gb']:.2f}GB{'':<4}{r['time_ms']:.1f}ms")

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {args.output}")

        best = results[0]
        print(f"\nRecommended config:")
        print(f"  embed_dim: {best['embed_dim']}")
        print(f"  depth: {best['depth']}")
        print(f"  num_experts: {best['experts']}")
        print(f"  Total params: {best['params']/1e6:.1f}M")


if __name__ == "__main__":
    main()
    