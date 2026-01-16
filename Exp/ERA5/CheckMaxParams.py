import argparse
import sys
import os
import torch
import torch.distributed as dist
import gc
import math
import traceback

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
    parser.add_argument("--min_search_dim", type=int, default=64)
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
        
        if z.is_complex():
            loss = z.abs().sum()
        else:
            loss = z.sum()
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        del model, optimizer, x, dt, z, loss
        torch.cuda.empty_cache()
        gc.collect()
        return True, num_params

    except Exception as e:
        if "out of memory" not in str(e).lower():
            print(f"\n[ERROR at Dim {dim}, Layers {layers}]:")
            traceback.print_exc()
        
        if model is not None: del model
        if optimizer is not None: del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0

def main():
    if int(os.environ.get("RANK", 0)) != 0: return
    args = get_args()
    device = torch.device("cuda:0")

    print(f"--- Scan Config (BSZ={args.bs}, H={args.H}, W={args.W}, P={args.patch_size}, Exp={args.expansion}) ---")

    results = []
    for layers in sorted(args.search_layers):
        print(f"\n[Scanning] Layers={layers}")
        best_dim = 0
        best_params = 0
        for dim in range(args.min_search_dim, args.max_search_dim + 1, args.dim_step):
            success, params = check_config_fit(args, dim, layers, device)
            if success:
                best_dim = dim
                best_params = params
                print(f"  > Dim {dim:<5} [OK] {format_params(params)}")
            else:
                print(f"  > Dim {dim:<5} [FAIL/OOM]")
                break 

        if best_dim > 0:
            results.append((best_params, layers, best_dim))
        else:
            break

    print("\n" + "="*60)
    print(f"{'Rank':<5} | {'Params':<15} | {'Layers':<8} | {'Max Dim':<8}")
    for i, (params, layers, dim) in enumerate(sorted(results, key=lambda x: x[0], reverse=True)):
        print(f"{i+1:<5} | {format_params(params):<15} | {layers:<8} | {dim:<8}")

if __name__ == "__main__":
    main()

