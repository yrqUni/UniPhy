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
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--min_search_dim", type=int, default=64)
    parser.add_argument("--max_search_dim", type=int, default=2048)
    parser.add_argument("--dim_step", type=int, default=64)
    parser.add_argument("--search_layers", type=int, nargs='+', default=[4, 6, 8, 12, 16, 24])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--decoder_type", type=str, default="diffusion", choices=["diffusion", "ensemble"])
    parser.add_argument("--ensemble_size", type=int, default=8)
    return parser.parse_args()

def format_params(num):
    if num >= 1e9: return f"{num / 1e9:.2f}B"
    elif num >= 1e6: return f"{num / 1e6:.2f}M"
    else: return f"{num}"

def check_config_fit(args, dim, layers, device):
    model = None
    optimizer = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        model = UniPhyModel(
            input_shape=(args.H, args.W),
            in_channels=args.C,
            dim=dim,
            patch_size=args.patch_size,
            num_layers=layers,
            para_pool_expansion=args.expansion,
            conserve_energy=True,
            decoder_type=args.decoder_type,
            ensemble_size=args.ensemble_size
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(args.bs, args.T, args.C, args.H, args.W, device=device)
        dt = torch.ones(args.bs, args.T, device=device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_evt.record()

        z, _ = model(x, dt)
        if args.decoder_type == "diffusion":
            loss = z.abs().sum() if z.is_complex() else z.sum()
        else:
            x_ref = x.reshape(-1, args.C, args.H, args.W)
            z_flat = z.reshape(-1, *z.shape[2:])
            pred_ens = model.decoder.generate_ensemble(z_flat, x_ref=x_ref)
            loss = pred_ens.sum()

        loss.backward()
        optimizer.step()

        end_evt.record()
        torch.cuda.synchronize()
        elapsed_time = start_evt.elapsed_time(end_evt)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        optimizer.zero_grad(set_to_none=True)
        del model, optimizer, x, dt, z, loss
        if 'pred_ens' in locals(): del pred_ens
        return True, num_params, elapsed_time, peak_mem

    except Exception as e:
        if "out of memory" not in str(e).lower():
            traceback.print_exc()
        if model is not None: del model
        if optimizer is not None: del optimizer
        return False, 0, 0, 0

def main():
    if int(os.environ.get("RANK", 0)) != 0: return
    args = get_args()
    device = torch.device("cuda:0")
    
    results = []
    for layers in sorted(args.search_layers):
        best_dim, best_params, best_time, best_mem = 0, 0, 0, 0
        for dim in range(args.min_search_dim, args.max_search_dim + 1, args.dim_step):
            success, params, step_time, mem = check_config_fit(args, dim, layers, device)
            if success:
                best_dim, best_params, best_time, best_mem = dim, params, step_time, mem
                print(f"Layers {layers} | Dim {dim:<5} [OK] {format_params(params)} | {step_time:.1f}ms | VRAM: {mem:.2f}GB")
            else:
                print(f"Layers {layers} | Dim {dim:<5} [FAIL/OOM]")
                break
        if best_dim > 0:
            results.append((best_params, layers, best_dim, best_time, best_mem))
        else:
            break

    print("\n" + "="*105)
    print(f"{'Rank':<5} | {'Params':<12} | {'Layers':<8} | {'Max Dim':<8} | {'Step Time':<12} | {'Peak VRAM':<12} | {'Type':<10}")
    print("-" * 105)
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)
    for i, (params, layers, dim, t, m) in enumerate(sorted_res):
        print(f"{i+1:<5} | {format_params(params):<12} | {layers:<8} | {dim:<8} | {t:>9.1f} ms | {m:>9.2f} GB | {args.decoder_type:<10}")

if __name__ == "__main__":
    main()

