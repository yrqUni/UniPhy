import sys
import torch

from pscan import pscan


def ref_scan(A, X):
    A, X = torch.broadcast_tensors(A, X)
    Y = torch.empty_like(X)
    y = torch.zeros_like(X[:, 0])
    L = X.shape[1]
    for t in range(L):
        y = A[:, t] * y + X[:, t]
        Y[:, t] = y
    return Y


def make_stable_AX(B, L, device, complex_dtype=torch.complex64, decay_min=0.90, decay_max=0.999):
    torch.manual_seed(0)
    phase = (2.0 * torch.pi) * torch.rand(B, L, device=device, dtype=torch.float32)
    decay = decay_min + (decay_max - decay_min) * torch.rand(B, L, device=device, dtype=torch.float32)
    A = decay * (torch.cos(phase) + 1j * torch.sin(phase))
    Xr = torch.randn(B, L, device=device, dtype=torch.float32)
    Xi = torch.randn(B, L, device=device, dtype=torch.float32)
    X = Xr + 1j * Xi
    return A.to(complex_dtype), X.to(complex_dtype)


def max_rel_err(a, b, eps=1e-8):
    diff = (a - b).abs().max().item()
    denom = b.abs().max().item() + eps
    return diff, diff / denom


def forward_check(device, B=8, L=257):
    A, X = make_stable_AX(B, L, device=device)
    Y_ref = ref_scan(A, X)
    Y_tri = pscan(A, X)
    if not torch.isfinite(Y_tri.real).all() or not torch.isfinite(Y_tri.imag).all():
        return float("nan"), float("nan"), True
    abs_err, rel_err = max_rel_err(Y_tri, Y_ref)
    return abs_err, rel_err, False


def broadcast_check(device, B=6, L=129):
    torch.manual_seed(2)
    A, _ = make_stable_AX(B, L, device=device)
    _, X1 = make_stable_AX(1, L, device=device)
    Y_ref = ref_scan(A, X1)
    Y_tri = pscan(A, X1)
    abs_err, rel_err = max_rel_err(Y_tri, Y_ref)
    return abs_err, rel_err


def backward_check(device, B=4, L=256):
    torch.manual_seed(1)
    A, X = make_stable_AX(B, L, device=device)
    A = A.detach().clone().requires_grad_(True)
    X = X.detach().clone().requires_grad_(True)

    Y_tri = pscan(A, X)
    loss = (Y_tri.real.square().mean() + Y_tri.imag.square().mean())
    loss.backward()
    dA_tri = A.grad.detach().clone()
    dX_tri = X.grad.detach().clone()

    A2 = A.detach().clone().requires_grad_(True)
    X2 = X.detach().clone().requires_grad_(True)
    Y_ref = ref_scan(A2, X2)
    loss2 = (Y_ref.real.square().mean() + Y_ref.imag.square().mean())
    loss2.backward()
    dA_ref = A2.grad.detach().clone()
    dX_ref = X2.grad.detach().clone()

    abs_dA, rel_dA = max_rel_err(dA_tri, dA_ref)
    abs_dX, rel_dX = max_rel_err(dX_tri, dX_ref)
    return (abs_dA, rel_dA), (abs_dX, rel_dX)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for Triton pscan test.")
        sys.exit(1)

    device = "cuda"
    torch.set_printoptions(precision=6, sci_mode=False)

    print("forward check (stable A, complex64):")
    abs_err, rel_err, has_nan = forward_check(device=device, B=8, L=257)
    print(f"  max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}  has_nan={has_nan}")

    print("forward check (broadcast shapes):")
    abs_err, rel_err = broadcast_check(device=device, B=6, L=129)
    print(f"  max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}")

    print("backward check (compare to reference autograd):")
    (abs_dA, rel_dA), (abs_dX, rel_dX) = backward_check(device=device, B=4, L=256)
    print(f"  dA: max_abs_err={abs_dA:.3e}  max_rel_err={rel_dA:.3e}")
    print(f"  dX: max_abs_err={abs_dX:.3e}  max_rel_err={rel_dX:.3e}")

    print("forward check (longer L, stable):")
    abs_err, rel_err, has_nan = forward_check(device=device, B=2, L=2048)
    print(f"  max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}  has_nan={has_nan}")

    print("all done")


if __name__ == "__main__":
    main()

