import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))

import torch
import pandas as pd
import gc

try:
    from pscanTriton import pscan as pscan_triton
    from pscanCUDA import pscan as pscan_cuda
    from pscanTorch import pscan as pscan_naive
except ImportError:
    sys.exit(1)

BATCH_SIZE = 8
CHANNELS = 32
STATE = 8
SEQLENS = [128, 512, 1024, 2048, 4096, 8192]
WARMUP_STEPS = 10
REP_STEPS = 50
CUDA_MAX_L = 1024
NAIVE_MAX_L = 2048

def get_input(L, device="cuda"):
    dtype = torch.complex64
    A = torch.randn(BATCH_SIZE, L, CHANNELS, STATE, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(BATCH_SIZE, L, CHANNELS, STATE, device=device, dtype=dtype, requires_grad=True)
    return A, X

def measure_perf(fn_apply, L, name):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    A, X = get_input(L)
    
    try:
        for _ in range(WARMUP_STEPS):
            y = fn_apply(A, X)
            loss = y.sum().abs()
            loss.backward()
            A.grad = None
            X.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        return {"Error": str(e)}

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(REP_STEPS):
        y = fn_apply(A, X)
    end_event.record()
    torch.cuda.synchronize()
    fwd_time = start_event.elapsed_time(end_event) / REP_STEPS
    
    y = fn_apply(A, X)
    loss = y.sum().abs()
    
    start_event.record()
    for _ in range(REP_STEPS):
        loss.backward(retain_graph=True)
        A.grad = None
        X.grad = None
    end_event.record()
    torch.cuda.synchronize()
    bwd_time = start_event.elapsed_time(end_event) / REP_STEPS
    
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        "Fwd (ms)": fwd_time,
        "Bwd (ms)": bwd_time,
        "Total (ms)": fwd_time + bwd_time,
        "Mem (MB)": max_mem
    }

def check_correctness():
    L = 128
    A, X = get_input(L)
    
    try:
        y_base = pscan_naive(A, X)
        loss_base = y_base.sum().abs()
        loss_base.backward()
        grad_base = A.grad.clone()
        A.grad = None; X.grad = None

        y_cuda = pscan_cuda(A, X)
        loss_cuda = y_cuda.sum().abs()
        loss_cuda.backward()
        grad_cuda = A.grad.clone()
        A.grad = None; X.grad = None
        
        y_triton = pscan_triton(A, X)
        loss_triton = y_triton.sum().abs()
        loss_triton.backward()
        grad_triton = A.grad.clone()
        
        diff_cuda_fwd = (y_base - y_cuda).abs().max().item()
        diff_cuda_bwd = (grad_base - grad_cuda).abs().max().item()
        
        diff_triton_fwd = (y_base - y_triton).abs().max().item()
        diff_triton_bwd = (grad_base - grad_triton).abs().max().item()
        
        print(f"CUDA   Fwd Diff: {diff_cuda_fwd:.2e} | Bwd Diff: {diff_cuda_bwd:.2e}")
        print(f"Triton Fwd Diff: {diff_triton_fwd:.2e} | Bwd Diff: {diff_triton_bwd:.2e}")
        
    except Exception:
        pass

def main():
    if not torch.cuda.is_available():
        return

    check_correctness()
    
    results = []
    
    for L in SEQLENS:
        if L <= NAIVE_MAX_L:
            res = measure_perf(pscan_naive.pscan, L, "Naive")
            if "Error" not in res:
                res["Model"] = "PyTorch"
                res["L"] = L
                results.append(res)
                base_time = res["Total (ms)"]
            else:
                base_time = None
        else:
            base_time = None

        if L <= CUDA_MAX_L:
            res = measure_perf(pscan_cuda.pscan, L, "CUDA")
            if "Error" not in res:
                res["Model"] = "CUDA"
                res["L"] = L
                if base_time: 
                    res["Speedup"] = f"{base_time / res['Total (ms)']:.1f}x"
                results.append(res)

        res = measure_perf(pscan_triton.pscan, L, "Triton")
        if "Error" not in res:
            res["Model"] = "Triton"
            res["L"] = L
            if base_time: 
                res["Speedup"] = f"{base_time / res['Total (ms)']:.1f}x"
            results.append(res)

    if not results:
        return

    df = pd.DataFrame(results)
    df = df.fillna("-")
    cols = ["L", "Model", "Fwd (ms)", "Bwd (ms)", "Total (ms)", "Speedup", "Mem (MB)"]
    df = df[[c for c in cols if c in df.columns]]
    
    try:
        print(df.to_markdown(index=False, floatfmt=".3f"))
    except ImportError:
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
