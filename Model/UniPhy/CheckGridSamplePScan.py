import torch
import time
import sys
import os

try:
    from GridSamplePScan import pscan_flow as pscan_triton
    from GridSamplePScan_Torch import pscan_flow as pscan_torch
except ImportError:
    sys.path.append(os.getcwd())
    from GridSamplePScan import pscan_flow as pscan_triton
    from GridSamplePScan_Torch import pscan_flow as pscan_torch

def benchmark(func, name, *args):
    torch.cuda.synchronize()
    start = time.time()
    out = func(*args)
    torch.cuda.synchronize()
    end = time.time()
    print(f"[{name}] Forward Time: {(end - start) * 1000:.2f} ms")
    return out

def main():
    torch.manual_seed(1017)
    device = "cuda"
    
    B, L, C, H, W = 2, 16, 32, 128, 128
    print(f"Config: B={B}, L={L}, C={C}, H={H}, W={W}")

    flows = torch.randn(B, L, 2, H, W, device=device, dtype=torch.float32) * 0.1
    images = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    
    flows.requires_grad_(True)
    images.requires_grad_(True)
    
    flows_ref = flows.clone().detach().requires_grad_(True)
    images_ref = images.clone().detach().requires_grad_(True)

    print("\n--- Forward Correctness & Speed ---")
    out_triton = benchmark(pscan_triton, "Triton", flows, images)
    out_torch = benchmark(pscan_torch, "Torch ", flows_ref, images_ref)

    diff_max = (out_triton - out_torch).abs().max().item()
    diff_mean = (out_triton - out_torch).abs().mean().item()
    print(f"Forward Max Diff:  {diff_max:.2e}")
    print(f"Forward Mean Diff: {diff_mean:.2e}")
    
    print("\n--- Backward Correctness & Speed ---")
    loss_triton = out_triton.sum()
    loss_torch = out_torch.sum()

    torch.cuda.synchronize()
    start = time.time()
    loss_triton.backward()
    torch.cuda.synchronize()
    print(f"[Triton] Backward Time: {(time.time() - start) * 1000:.2f} ms")

    torch.cuda.synchronize()
    start = time.time()
    loss_torch.backward()
    torch.cuda.synchronize()
    print(f"[Torch ] Backward Time: {(time.time() - start) * 1000:.2f} ms")

    grad_flow_diff = (flows.grad - flows_ref.grad).abs().max().item()
    grad_img_diff = (images.grad - images_ref.grad).abs().max().item()

    print(f"Grad Flow Max Diff: {grad_flow_diff:.2e}")
    print(f"Grad Img  Max Diff: {grad_img_diff:.2e}")

    status = "PASS" if diff_max < 1e-3 and grad_flow_diff < 1e-2 else "FAIL"
    print(f"\n>> Check Result: {status}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("CUDA not available")

