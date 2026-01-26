import torch
import torch.nn as nn
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from UniPhyOps import RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork
from PScan import pscan


def print_sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def compare(name, t1, t2, atol=1e-5):
    diff = (t1 - t2).abs().max().item()
    status = "PASS" if diff < atol else "FAIL"
    print(f"{name}: diff={diff:.2e} [{status}]")
    return diff < atol


def check_basis_invertibility():
    print_sep("Basis Invertibility")
    torch.manual_seed(42)

    dim = 64
    basis = ComplexSVDTransform(dim).cuda()
    x = torch.randn(4, 16, dim, dtype=torch.complex64, device="cuda")

    h = basis.encode(x)
    x_rec = basis.decode(h)

    print(f"Dimension: {dim}")
    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {h.shape}")
    print(f"Reconstruction Error: {(x - x_rec).abs().max().item():.2e}")
    print(f"DFT Weight (sigmoid): {torch.sigmoid(basis.dft_weight).item():.4f}")

    return compare("Invertibility", x, x_rec, atol=1e-4)


def check_eigenvalue_bounds():
    print_sep("Eigenvalue Bounded Growth")
    torch.manual_seed(42)

    dim = 64
    max_rate = 0.3
    prop = TemporalPropagator(dim, max_growth_rate=max_rate).cuda()

    lam = prop._get_effective_lambda()

    print(f"Dimension: {dim}")
    print(f"Max Growth Rate Config: {max_rate}")
    print(f"Actual Max Real Part: {lam.real.max().item():.4f}")
    print(f"Actual Min Real Part: {lam.real.min().item():.4f}")

    bounded = (lam.real.abs() <= max_rate).all().item()
    print(f"Bounded in [-{max_rate}, {max_rate}]: {bounded}")

    return bounded


def check_ffn():
    print_sep("FFN Complex Multiplication")
    torch.manual_seed(42)

    ffn = UniPhyFeedForwardNetwork(32, 2, 4).cuda()
    x = torch.randn(2, 32, 8, 8, dtype=torch.complex64, device="cuda")

    with torch.no_grad():
        y1 = ffn(x)
        y2 = ffn(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y1.shape}")

    diff = (y1 - y2).abs().max().item()
    print(f"Deterministic Diff: {diff:.2e}")

    return compare("Determinism", y1, y2)


def check_flux_tracker():
    print_sep("GlobalFluxTracker Gate Output")
    torch.manual_seed(42)

    tracker = GlobalFluxTracker(64).cuda()
    prev_state = torch.randn(2, 64, dtype=torch.complex64, device="cuda")
    x_t = torch.randn(2, 64, dtype=torch.complex64, device="cuda")

    new_state, source, gate = tracker.forward_step(prev_state, x_t)

    print(f"Input State Shape: {prev_state.shape}")
    print(f"New State Shape: {new_state.shape}")
    print(f"Source Shape: {source.shape}")
    print(f"Gate Shape: {gate.shape}")
    print(f"Gate Range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")

    valid = gate.min() >= 0 and gate.max() <= 1
    return valid


def check_riemannian_clifford():
    print_sep("RiemannianCliffordConv2d")
    torch.manual_seed(42)

    conv = RiemannianCliffordConv2d(32, 32, 3, 1, 16, 16).cuda()
    x = torch.randn(2, 32, 16, 16, device="cuda")

    y = conv(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    print(f"Metric Scale: {conv.metric_scale.item():.4f}")
    print(f"Viscosity Scale: {conv.viscosity_scale.item():.4f}")

    return y.shape == x.shape


def check_pscan():
    print_sep("PScan Correctness")
    torch.manual_seed(42)

    B, L, C, D = 2, 8, 4, 2
    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_pscan = pscan(A, X)

    Y_loop = torch.zeros_like(X)
    acc = torch.zeros(B, C, D, 1, dtype=torch.complex64, device="cuda")
    for t in range(L):
        acc = acc * A[:, t].unsqueeze(-1) + X[:, t]
        Y_loop[:, t] = acc

    print(f"A Shape: {A.shape}")
    print(f"X Shape: {X.shape}")
    print(f"Y Shape: {Y_pscan.shape}")

    return compare("PScan vs Loop", Y_pscan, Y_loop, atol=1e-5)


def check_block_consistency():
    print_sep("UniPhyBlock: forward vs forward_step")
    torch.manual_seed(42)

    B, T, D, H, W = 1, 4, 32, 8, 8

    block = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")
    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")

    with torch.no_grad():
        z_par, _, _ = block(x, h_init, dt, flux_init)

    z_ser_list = []
    h, flux = h_init.clone(), flux_init.clone()
    with torch.no_grad():
        for t in range(T):
            z_t, h, flux = block.forward_step(x[:, t], h, dt[t], flux)
            z_ser_list.append(z_t)
    z_ser = torch.stack(z_ser_list, dim=1)

    print(f"Parallel Shape: {z_par.shape}")
    print(f"Serial Shape: {z_ser.shape}")

    passed = compare("Block Consistency", z_par, z_ser, atol=1e-4)

    if not passed:
        for t in range(T):
            diff = (z_par[:, t] - z_ser[:, t]).abs().max().item()
            print(f"  t={t}: diff={diff:.2e}")

    return passed


def check_model_consistency():
    print_sep("Model Parallel vs Serial")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33, sde_mode="ode"
    ).cuda().eval()

    x = torch.randn(1, 5, 2, 33, 33, device="cuda") * 0.1
    dt = torch.ones(5, device="cuda")

    with torch.no_grad():
        out_par = model(x, dt)
        out_ser = model.forward_rollout(x[:, 0], [1.0] * 5)

    print(f"Parallel Shape: {out_par.shape}")
    print(f"Serial Shape: {out_ser.shape}")

    passed = compare("Model Consistency", out_par, out_ser, atol=1e-2)

    print(f"Parallel Mean: {out_par.mean().item():.6f}")
    print(f"Serial Mean: {out_ser.mean().item():.6f}")

    return passed


def check_full_forward():
    print_sep("Full Model Forward")
    torch.manual_seed(42)

    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33
    ).cuda().eval()

    x = torch.randn(1, 4, 2, 33, 33, device="cuda")
    dt = torch.ones(4, device="cuda")

    with torch.no_grad():
        out = model(x, dt)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Output Mean: {out.mean().item():.4f}")
    print(f"Output Std: {out.std().item():.4f}")

    return out.shape == (1, 4, 2, 33, 33)


def run_all_checks():
    print("=" * 60)
    print("  UniPhy Model Check Suite")
    print("=" * 60)

    results = {}

    results["basis"] = check_basis_invertibility()
    results["eigenvalue"] = check_eigenvalue_bounds()
    results["ffn"] = check_ffn()
    results["flux_tracker"] = check_flux_tracker()
    results["clifford"] = check_riemannian_clifford()
    results["pscan"] = check_pscan()
    results["block"] = check_block_consistency()
    results["forward"] = check_full_forward()
    results["consistency"] = check_model_consistency()

    print_sep("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


if __name__ == "__main__":
    run_all_checks()
    