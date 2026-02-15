import time
import torch

from PScan import pscan


def sequential_pscan_diag(a_diag, x_diag):
    n, length, channels, _ = x_diag.shape
    y = torch.zeros_like(x_diag)
    acc = torch.zeros((n, channels), dtype=x_diag.dtype, device=x_diag.device)
    for t in range(length):
        acc = a_diag[:, t, :, 0] * acc + x_diag[:, t, :, 0]
        y[:, t, :, 0] = acc
    return y


def check_forward_diag():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, length, channels = 2, 256, 256
    a = torch.randn(n, length, channels, 1, dtype=torch.complex64, device=device) * 0.1
    x = torch.randn(n, length, channels, 1, dtype=torch.complex64, device=device) * 0.1
    y_seq = sequential_pscan_diag(a, x)
    y_par = pscan(a, x)
    max_diff = (y_seq - y_par).abs().max().item()
    mean_diff = (y_seq - y_par).abs().mean().item()
    return max_diff, mean_diff


def benchmark():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, length, channels = 64, 256, 256
    a = torch.randn(n, length, channels, 1, dtype=torch.complex64, device=device) * 0.05
    x = torch.randn(n, length, channels, 1, dtype=torch.complex64, device=device) * 0.05
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    y = pscan(a, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return y, t1 - t0


if __name__ == "__main__":
    md, ad = check_forward_diag()
    y, dt = benchmark()
    print(f"max_diff={md:.3e} mean_diff={ad:.3e} time={dt:.3f}s shape={tuple(y.shape)}")
