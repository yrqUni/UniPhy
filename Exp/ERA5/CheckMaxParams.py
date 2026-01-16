import argparse
import sys
import os
import torch
import torch.distributed as dist
import gc
import math

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--H", type=int, default=721)
    parser.add_argument("--W", type=int, default=1440)
    parser.add_argument("--C", type=int, default=30)
    parser.add_argument("--T", type=int, default=4)

    parser.add_argument("--min_search_dim", type=int, default=8)
    parser.add_argument("--max_search_dim", type=int, default=2048)
    parser.add_argument("--dim_step", type=int, default=64)
    parser.add_argument("--search_layers", type=int, nargs='+', default=[4, 8, 12, 16, 24])
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=1)
    
    return parser.parse_args()

def format_params(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    else:
        return f"{num}"

def check_config_fit(args, dim, layers, device):
    model = None
    optimizer = None
    try:
        model = UniPhyModel(
            input_shape=(args.H, args.W),
            in_channels=args.C,
            dim=dim,
            patch_size=args.patch_size,
            num_layers=layers,
            para_pool_expansion=args.expansion,
            conserve_energy=True
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.ones(args.bs, args.T, device=device)

        z, _ = model(x, dt)
        loss = z.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        del model, optimizer, x, dt, z, loss
        torch.cuda.empty_cache()
        gc.collect()
        return True, num_params

    except Exception:
        if model is not None: del model
        if optimizer is not None: del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0

def main():
    if int(os.environ.get("RANK", 0)) != 0: return

    args = get_args()
    device = torch.device("cuda:0")

    print(f"--- Linear Search Max Params (BSZ={args.bs}, H={args.H}, W={args.W}, P={args.patch_size}, Exp={args.expansion}) ---")

    results = []
    sorted_layers = sorted(args.search_layers)

    for layers in sorted_layers:
        print(f"\n[Scanning Configuration] Layers={layers}")

        best_dim = 0
        best_params = 0
        
        dims_to_try = range(args.min_search_dim, args.max_search_dim + 1, args.dim_step)
        
        for dim in dims_to_try:
            success, params = check_config_fit(args, dim, layers, device)

            if success:
                best_dim = dim
                best_params = params
                print(f"  > Dim {dim:<5} [OK] {format_params(params)}")
            else:
                print(f"  > Dim {dim:<5} [OOM]")
                break 

        print(f"  -> Found Max Dim: {best_dim}")

        if best_dim > 0:
            results.append((best_params, layers, best_dim))
        else:
            print(f"  [STOP] Min dim failed for Layers={layers}. Stopping.")

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "="*60)
    print(f"FINAL LEADERBOARD (Sorted by Max Params)")
    print(f"{'Rank':<5} | {'Params':<15} | {'Layers':<8} | {'Max Dim':<8}")
    print("-" * 60)

    for i, (params, layers, dim) in enumerate(results):
        print(f"{i+1:<5} | {format_params(params):<15} | {layers:<8} | {dim:<8}")

    if results:
        top_p, top_l, top_d = results[0]
        print("="*60)
        print(f"WINNER CONFIGURATION (BSZ={args.bs}):")
        print(f"  --num_layers {top_l}")
        print(f"  --dim {top_d}")
        print(f"  --patch_size {args.patch_size}")
        print(f"  --para_pool_expansion {args.expansion}")
        print(f"  Total Params: {format_params(top_p)}")
        print("="*60)

if __name__ == "__main__":
    main()

