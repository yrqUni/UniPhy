import argparse
import sys
import os
import torch
import gc
import traceback
import numpy as np

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
    parser.add_argument("--min_search_dim", type=int, default=64)
    parser.add_argument("--max_search_dim", type=int, default=1536)
    parser.add_argument("--dim_step", type=int, default=32)
    parser.add_argument("--search_depths", type=int, nargs='+', default=[4, 6, 8, 12])
    parser.add_argument("--search_experts", type=int, nargs='+', default=[4, 8])
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=16)
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
            img_width=args.W
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.rand(args.bs, args.T, device=device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_evt.record()

        out = model(x, dt)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        end_evt.record()
        torch.cuda.synchronize()

        elapsed_time = start_evt.elapsed_time(end_evt)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        optimizer.zero_grad(set_to_none=True)
        del model, optimizer, x, dt, out, loss
        torch.cuda.empty_cache()
        
        return True, num_params, elapsed_time, peak_mem

    except Exception as e:
        if model is not None: del model
        if optimizer is not None: del optimizer
        torch.cuda.empty_cache()
        return False, 0, 0, 0

def main():
    if int(os.environ.get("RANK", 0)) != 0: return
    args = get_args()
    
    if not torch.cuda.is_available():
        sys.exit(1)
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*100}")
    print(f" UniPhy Max Parameter Search (Aligned with Train)")
    print(f" Input: {args.bs}x{args.T}x{args.C}x{args.H}x{args.W}")
    print(f" Experts: {args.search_experts} | Depths: {args.search_depths}")
    print(f" Expand: {args.expand} | Patch: {args.patch_size}")
    print(f"{'='*100}\n")

    results = []
    
    for experts in sorted(args.search_experts):
        for depth in sorted(args.search_depths):
            best_dim, best_params, best_time, best_mem = 0, 0, 0, 0
            
            for dim in range(args.min_search_dim, args.max_search_dim + 1, args.dim_step):
                success, params, step_time, mem = check_config_fit(args, dim, depth, experts, device)
                
                if success:
                    best_dim, best_params, best_time, best_mem = dim, params, step_time, mem
                    status = f"\033[92m[PASS]\033[0m"
                    print(f" Exp {experts:<2} | Dep {depth:<2} | Dim {dim:<4} | Params {format_count(params):<7} | Time {step_time:>6.1f}ms | Mem {mem:>5.2f}GB {status}")
                else:
                    status = f"\033[91m[OOM ]\033[0m"
                    print(f" Exp {experts:<2} | Dep {depth:<2} | Dim {dim:<4} | {'-'*14} | {'-'*10} | {'-'*9} {status}")
                    break
            
            if best_dim > 0:
                results.append((best_params, experts, depth, best_dim, best_time, best_mem))

    print("\n" + "="*110)
    print(f"{'Rank':<4} | {'Params':<10} | {'Experts':<7} | {'Depth':<5} | {'Max Dim':<8} | {'Step Time':<12} | {'Peak VRAM':<10}")
    print("-" * 110)
    
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)
    for i, (params, experts, depth, dim, t, m) in enumerate(sorted_res):
        print(f"{i+1:<4} | {format_count(params):<10} | {experts:<7} | {depth:<5} | {dim:<8} | {t:>9.1f} ms | {m:>7.2f} GB")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
    