import torch
import torch.nn as nn
import time

def benchmark_module(model, x, dt, iterations=100, warmup=10):
    for _ in range(warmup):
        _ = model(x, dt)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        _ = model(x, dt)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations

def run_speed_test():
    from UniPhyOps import UniPhyFluidScan as UniPhyBase
    from UniPhyOpsTriton import UniPhyFluidScan as UniPhyTriton
    
    device = torch.device("cuda")
    configs = [
        (1, 1, 4, 128, 128),
        (1, 16, 4, 128, 128),
        (4, 16, 4, 128, 128),
        (1, 1, 4, 256, 256)
    ]
    
    print(f"{'Config (B,T,C,H,W)':<25} | {'Base (ms)':<10} | {'Triton (ms)':<10} | {'Speedup':<8}")
    print("-" * 65)

    for B, T, C, H, W in configs:
        x = torch.randn(B, T, C, H, W, device=device)
        dt = torch.tensor([0.02] * B, device=device)
        
        model_base = UniPhyBase(C, H, W).to(device)
        model_triton = UniPhyTriton(C, H, W).to(device)
        
        t_base = benchmark_module(model_base, x, dt)
        t_triton = benchmark_module(model_triton, x, dt)
        
        speedup = t_base / t_triton
        config_str = f"({B},{T},{C},{H},{W})"
        print(f"{config_str:<25} | {t_base:10.2f} | {t_triton:10.2f} | {speedup:7.2f}x")

        del x, dt, model_base, model_triton
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_speed_test()

