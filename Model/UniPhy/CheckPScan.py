import torch
import time
from PScan import pscan


def sequential_pscan_diag(A, X):
    B, L, C, D, D2 = X.shape
    A_mat = torch.zeros(B, L, C, D, D, dtype=A.dtype, device=A.device)
    for i in range(D):
        A_mat[..., i, i] = A[..., i]

    Y_list = [X[:, 0]]
    for t in range(1, L):
        Y_t = torch.einsum("bcij,bcjk->bcik", A_mat[:, t], Y_list[t - 1]) + X[:, t]
        Y_list.append(Y_t)
    return torch.stack(Y_list, dim=1)


def sequential_pscan_mat(A, X):
    B, L, C, D, D2 = X.shape
    if A.ndim == 4:
        A_mat = torch.zeros(B, L, C, D, D, dtype=A.dtype, device=A.device)
        for i in range(D):
            A_mat[..., i, i] = A[..., i]
        A = A_mat

    Y_list = [X[:, 0]]
    for t in range(1, L):
        Y_t = torch.einsum("bcij,bcjk->bcik", A[:, t], Y_list[t - 1]) + X[:, t]
        Y_list.append(Y_t)
    return torch.stack(Y_list, dim=1)


def check_forward_diag():
    print("=" * 60)
    print("Testing Forward Pass (Diagonal Mode)")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_diag(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"Shape A: {A.shape}")
    print(f"Shape X: {X.shape}")
    print(f"Shape Y: {Y_par.shape}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")

    passed = max_diff < 1e-4
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_forward_mat():
    print("=" * 60)
    print("Testing Forward Pass (Matrix Mode)")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_mat(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"Shape A: {A.shape}")
    print(f"Shape X: {X.shape}")
    print(f"Shape Y: {Y_par.shape}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")

    passed = max_diff < 1e-4
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_backward_diag():
    print("=" * 60)
    print("Testing Backward Pass (Diagonal Mode)")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A_data = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X_data = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    A_seq = A_data.clone().requires_grad_(True)
    X_seq = X_data.clone().requires_grad_(True)
    A_par = A_data.clone().requires_grad_(True)
    X_par = X_data.clone().requires_grad_(True)

    Y_seq = sequential_pscan_diag(A_seq, X_seq)
    loss_seq = Y_seq.abs().pow(2).sum()
    loss_seq.backward()

    Y_par = pscan(A_par, X_par)
    loss_par = Y_par.abs().pow(2).sum()
    loss_par.backward()

    dA_max_diff = (A_seq.grad - A_par.grad).abs().max().item()
    dA_mean_diff = (A_seq.grad - A_par.grad).abs().mean().item()
    dX_max_diff = (X_seq.grad - X_par.grad).abs().max().item()
    dX_mean_diff = (X_seq.grad - X_par.grad).abs().mean().item()

    print(f"dA Max Difference: {dA_max_diff:.2e}")
    print(f"dA Mean Difference: {dA_mean_diff:.2e}")
    print(f"dX Max Difference: {dX_max_diff:.2e}")
    print(f"dX Mean Difference: {dX_mean_diff:.2e}")

    passed = dA_max_diff < 1e-3 and dX_max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_backward_mat():
    print("=" * 60)
    print("Testing Backward Pass (Matrix Mode)")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A_data = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X_data = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    A_seq = A_data.clone().requires_grad_(True)
    X_seq = X_data.clone().requires_grad_(True)
    A_par = A_data.clone().requires_grad_(True)
    X_par = X_data.clone().requires_grad_(True)

    Y_seq = sequential_pscan_mat(A_seq, X_seq)
    loss_seq = Y_seq.abs().pow(2).sum()
    loss_seq.backward()

    Y_par = pscan(A_par, X_par)
    loss_par = Y_par.abs().pow(2).sum()
    loss_par.backward()

    dA_max_diff = (A_seq.grad - A_par.grad).abs().max().item()
    dA_mean_diff = (A_seq.grad - A_par.grad).abs().mean().item()
    dX_max_diff = (X_seq.grad - X_par.grad).abs().max().item()
    dX_mean_diff = (X_seq.grad - X_par.grad).abs().mean().item()

    print(f"dA Max Difference: {dA_max_diff:.2e}")
    print(f"dA Mean Difference: {dA_mean_diff:.2e}")
    print(f"dX Max Difference: {dX_max_diff:.2e}")
    print(f"dX Mean Difference: {dX_mean_diff:.2e}")

    passed = dA_max_diff < 1e-3 and dX_max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_various_shapes():
    print("=" * 60)
    print("Testing Various Shapes")
    print("=" * 60)

    shapes = [
        (1, 8, 1, 2),
        (2, 16, 4, 2),
        (4, 32, 8, 2),
        (1, 64, 2, 2),
        (2, 128, 4, 2),
    ]

    all_passed = True
    for B, L, C, D in shapes:
        torch.manual_seed(42)

        A_diag = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
        A_mat = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
        X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

        Y_seq_diag = sequential_pscan_diag(A_diag, X)
        Y_par_diag = pscan(A_diag, X)
        diff_diag = (Y_seq_diag - Y_par_diag).abs().max().item()

        Y_seq_mat = sequential_pscan_mat(A_mat, X)
        Y_par_mat = pscan(A_mat, X)
        diff_mat = (Y_seq_mat - Y_par_mat).abs().max().item()

        passed_diag = diff_diag < 1e-4
        passed_mat = diff_mat < 1e-4
        passed = passed_diag and passed_mat

        print(f"Shape (B={B}, L={L}, C={C}, D={D}): Diag={diff_diag:.2e} Mat={diff_mat:.2e} Passed={passed}")
        all_passed = all_passed and passed

    print()
    return all_passed


def check_long_sequence():
    print("=" * 60)
    print("Testing Long Sequence")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 512, 4, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.2
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_mat(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"Shape A: {A.shape}")
    print(f"Shape X: {X.shape}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")

    passed = max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_4d_input():
    print("=" * 60)
    print("Testing 4D Input (Auto Squeeze)")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda")

    Y_par = pscan(A, X)

    print(f"Shape A: {A.shape}")
    print(f"Shape X: {X.shape}")
    print(f"Shape Y: {Y_par.shape}")

    passed = Y_par.shape == X.shape
    print(f"Output shape matches input: {passed}")
    print()
    return passed


def benchmark():
    print("=" * 60)
    print("Benchmark")
    print("=" * 60)

    torch.manual_seed(42)
    B, L, C, D = 8, 256, 16, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    for _ in range(3):
        _ = pscan(A, X)

    torch.cuda.synchronize()
    n_iters = 100

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = sequential_pscan_mat(A, X)
    torch.cuda.synchronize()
    seq_time = (time.time() - start) / n_iters * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = pscan(A, X)
    torch.cuda.synchronize()
    par_time = (time.time() - start) / n_iters * 1000

    print(f"Shape: B={B}, L={L}, C={C}, D={D}")
    print(f"Sequential Time: {seq_time:.3f} ms")
    print(f"Parallel Time: {par_time:.3f} ms")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    print()


def main():
    print("=" * 60)
    print("PScan Correctness Check")
    print("=" * 60)
    print()

    results = {}

    results["forward_diag"] = check_forward_diag()
    results["forward_mat"] = check_forward_mat()
    results["backward_diag"] = check_backward_diag()
    results["backward_mat"] = check_backward_mat()
    results["various_shapes"] = check_various_shapes()
    results["long_sequence"] = check_long_sequence()
    results["4d_input"] = check_4d_input()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print()

    benchmark()

    return all_passed


if __name__ == "__main__":
    main()
    