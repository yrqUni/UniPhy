import argparse
import sys
import os
import torch
import gc
import traceback
import time

try:
    from ModelUniPhy import UniPhyModel
except ImportError:
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
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--min_search_dim", type=int, default=64)
    parser.add_argument("--max_search_dim", type=int, default=2048)
    parser.add_argument("--dim_step", type=int, default=64)
    parser.add_argument("--search_layers", type=int, nargs='+', default=[4, 6, 8, 12, 16, 24])
    parser.add_argument("--patch_size", type=int, default=16)
    return parser.parse_args()

def format_count(num):
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
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.cuda.reset_peak_memory_stats()
        
        model = UniPhyModel(
            in_channels=args.C,
            out_channels=args.C,
            embed_dim=dim,
            depth=layers,
            patch_size=args.patch_size,
            img_height=args.H,
            img_width=args.W,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.rand(args.bs, args.T, device=device)

        torch.cuda.reset_peak_memory_stats()

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
        if "out of memory" not in str(e).lower():
            print(f"\n[ERROR] Configuration Failed: Layers={layers}, Dim={dim}")
            traceback.print_exc()
        
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
    print(f" UniPhy Max Parameter Search")
    print(f" Input: {args.bs}x{args.T}x{args.C}x{args.H}x{args.W} | Patch: {args.patch_size}")
    print(f" Args: {args}") 
    print(f"{'='*100}\n")

    results = []
    
    for layers in sorted(args.search_layers):
        best_dim, best_params, best_time, best_mem = 0, 0, 0, 0
        
        for dim in range(args.min_search_dim, args.max_search_dim + 1, args.dim_step):
            success, params, step_time, mem = check_config_fit(args, dim, layers, device)
            
            if success:
                best_dim, best_params, best_time, best_mem = dim, params, step_time, mem
                status = f"\033[92m[PASS]\033[0m"
                print(f" Depth {layers:<2} | Dim {dim:<4} | Params {format_count(params):<7} | Time {step_time:>6.1f}ms | Mem {mem:>5.2f}GB {status}")
            else:
                status = f"\033[91m[OOM ]\033[0m"
                print(f" Depth {layers:<2} | Dim {dim:<4} | {'-'*14} | {'-'*10} | {'-'*9} {status}")
                break
        
        if best_dim > 0:
            results.append((best_params, layers, best_dim, best_time, best_mem))

    print("\n" + "="*100)
    print(f"{'Rank':<4} | {'Params':<10} | {'Depth':<6} | {'Max Dim':<8} | {'Step Time':<12} | {'Peak VRAM':<10}")
    print("-" * 100)
    
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)
    for i, (params, layers, dim, t, m) in enumerate(sorted_res):
        print(f"{i+1:<4} | {format_count(params):<10} | {layers:<6} | {dim:<8} | {t:>9.1f} ms | {m:>7.2f} GB")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()

