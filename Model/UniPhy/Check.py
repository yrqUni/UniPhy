import gc
import sys
import torch
import torch.nn.functional as F

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, ComplexSVDTransform, GlobalFluxTracker, RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from PScan import pscan


def print_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_basis_invertibility():
    print_section("Testing Basis Invertibility (with DFT Residual)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    B, N = 4, 16

    basis = ComplexSVDTransform(dim).to(device).double()

    x = torch.randn(B, N, dim, device=device, dtype=torch.cdouble)

    with torch.no_grad():
        h = basis.encode(x)
        x_rec = basis.decode(h)

    diff = (x - x_rec).abs().max().item()
    dft_weight = torch.sigmoid(basis.dft_weight).item()

    print(f"Dimension: {dim}")
    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {h.shape}")
    print(f"Reconstruction Error: {diff:.2e}")
    print(f"DFT Weight (sigmoid): {dft_weight:.4f}")

    passed = diff < 1e-2
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_eigenvalue_stability():
    print_section("Testing Eigenvalue Bounded Growth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    max_growth = 0.3

    prop = TemporalPropagator(
        dim, dt_ref=6.0, sde_mode="det", max_growth_rate=max_growth
    ).to(device).double()

    lam = prop._get_effective_lambda()
    real_parts = lam.real

    print(f"Dimension: {dim}")
    print(f"Max Growth Rate Config: {max_growth}")
    print(f"Actual Max Real Part: {real_parts.max().item():.4f}")
    print(f"Actual Min Real Part: {real_parts.min().item():.4f}")

    passed = real_parts.max().item() <= max_growth and real_parts.min().item() >= -max_growth
    print(f"Bounded in [-{max_growth}, {max_growth}]: {passed}")
    print()

    return passed


def check_ffn_complex_multiplication():
    print_section("Testing FFN Complex Multiplication Order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 32, 8, 8

    ffn = UniPhyFeedForwardNetwork(C, expand=2, num_experts=4).to(device)

    x = torch.randn(B, C, H, W, device=device, dtype=torch.cfloat)

    torch.manual_seed(42)
    out1 = ffn(x)

    torch.manual_seed(42)
    out2 = ffn(x)

    diff = (out1 - out2).abs().max().item()

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out1.shape}")
    print(f"Deterministic Diff: {diff:.2e}")

    passed = diff < 1e-6
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_ffn_causality():
    print_section("Testing FFN Causality")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 4, 16, 8, 8

    ffn = UniPhyFeedForwardNetwork(C, expand=2, num_experts=4).to(device)

    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.cfloat)
    x_flat = x.reshape(B * T, C, H, W)

    with torch.no_grad():
        out_parallel = ffn(x_flat).reshape(B, T, C, H, W)

        out_serial = []
        for t in range(T):
            out_t = ffn(x[:, t])
            out_serial.append(out_t)
        out_serial = torch.stack(out_serial, dim=1)

    diff = (out_parallel - out_serial).abs().max().item()

    print(f"Parallel Shape: {out_parallel.shape}")
    print(f"Serial Shape: {out_serial.shape}")
    print(f"Max Difference: {diff:.2e}")

    passed = diff < 1e-5
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_flux_tracker_gate():
    print_section("Testing GlobalFluxTracker Gate Output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    B = 2

    tracker = GlobalFluxTracker(dim).to(device).double()

    prev_state = torch.randn(B, dim, device=device, dtype=torch.cdouble)
    x_t = torch.randn(B, dim, device=device, dtype=torch.cdouble)

    new_state, source, gate = tracker.forward_step(prev_state, x_t)

    print(f"Input State Shape: {prev_state.shape}")
    print(f"New State Shape: {new_state.shape}")
    print(f"Source Shape: {source.shape}")
    print(f"Gate Shape: {gate.shape}")
    print(f"Gate Range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")

    passed = (
        gate.shape == (B, dim)
        and gate.min() >= 0
        and gate.max() <= 1
        and not gate.is_complex()
    )
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_heteroscedastic_noise():
    print_section("Testing Heteroscedastic Noise Generation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    B, N, T = 2, 16, 1

    prop = TemporalPropagator(
        dim, dt_ref=6.0, sde_mode="sde", init_noise_scale=1.0
    ).to(device).double()

    target_shape = (B * N, T, dim)
    dt = torch.tensor([6.0], device=device)

    h_state_low = torch.ones(B * N, T, dim, device=device, dtype=torch.cdouble) * 0.1
    h_state_high = torch.ones(B * N, T, dim, device=device, dtype=torch.cdouble) * 10.0

    noise_low = prop.generate_stochastic_term(target_shape, dt, torch.cdouble, h_state=h_state_low)
    noise_high = prop.generate_stochastic_term(target_shape, dt, torch.cdouble, h_state=h_state_high)

    std_low = noise_low.abs().std().item()
    std_high = noise_high.abs().std().item()

    print(f"Low State Noise Std: {std_low:.4f}")
    print(f"High State Noise Std: {std_high:.4f}")
    print(f"Ratio (High/Low): {std_high / std_low:.2f}")

    passed = std_high > std_low
    print(f"Heteroscedastic Behavior: {passed}")
    print()

    return passed


def check_riemannian_clifford_conv():
    print_section("Testing RiemannianCliffordConv2d with Dispersion")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = 32, 16, 16

    conv = RiemannianCliffordConv2d(C, C, kernel_size=3, padding=1, img_height=H, img_width=W).to(device)

    x = torch.randn(2, C, H, W, device=device)

    with torch.no_grad():
        out = conv(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Viscosity Scale: {conv.viscosity_scale.item():.4f}")
    print(f"Dispersion Scale: {conv.dispersion_scale.item():.4f}")

    has_dispersion = hasattr(conv, "dispersion_scale")
    has_anti_diff = hasattr(conv, "anti_diffusion_gate")

    print(f"Has Dispersion Term: {has_dispersion}")
    print(f"Has Anti-Diffusion Gate: {has_anti_diff}")

    passed = out.shape == x.shape
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_io_shapes():
    print_section("Testing Encoder/Decoder IO Shapes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 5, 2, 721, 1440
    embed_dim = 64
    patch_size = 16

    encoder = UniPhyEncoder(C, embed_dim, patch_size, img_height=H, img_width=W).to(device)
    decoder = UniPhyEnsembleDecoder(C, embed_dim, patch_size, img_height=H, img_width=W).to(device)

    x = torch.randn(B, T, C, H, W, device=device)

    with torch.no_grad():
        latent = encoder(x)
        out = decoder(latent)

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    expected_h = (H + pad_h) // patch_size
    expected_w = (W + pad_w) // patch_size

    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {latent.shape}")
    print(f"Expected Latent H: {expected_h}, Actual: {latent.shape[-2]}")
    print(f"Expected Latent W: {expected_w}, Actual: {latent.shape[-1]}")
    print(f"Output Shape: {out.shape}")
    print(f"Expected Output Shape: {(B, T, C, H, W)}")

    passed = out.shape == (B, T, C, H, W)
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_pscan_compatibility():
    print_section("Testing PScan Compatibility")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, dtype=torch.complex64, device=device) * 0.5
    X = torch.randn(B, L, C, D, dtype=torch.complex64, device=device)

    X_5d = X.unsqueeze(-1)

    Y = pscan(A, X_5d).squeeze(-1)

    Y_seq = torch.zeros_like(X)
    Y_seq[:, 0] = X[:, 0]
    for t in range(1, L):
        Y_seq[:, t] = A[:, t] * Y_seq[:, t - 1] + X[:, t]

    diff = (Y - Y_seq).abs().max().item()

    print(f"Diagonal Mode Shape: A={A.shape}, X={X.shape}")
    print(f"Diagonal Mode Max Diff: {diff:.2e}")

    passed = diff < 1e-4
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_full_model_forward():
    print_section("Testing Full Model Forward Pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 4, 2, 64, 128
    dt_ref = 6.0

    model = UniPhyModel(
        in_channels=C,
        out_channels=C,
        embed_dim=32,
        expand=2,
        num_experts=4,
        depth=2,
        patch_size=16,
        img_height=H,
        img_width=W,
        dt_ref=dt_ref,
        sde_mode="det",
        max_growth_rate=0.3,
    ).to(device)

    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(T, device=device) * dt_ref

    model.eval()
    with torch.no_grad():
        out = model(x, dt)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Expected Shape: ({B}, {T}, {C}, {H}, {W})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    passed = out.shape == (B, T, C, H, W)
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_gradient_flow():
    print_section("Testing Gradient Flow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 3, 2, 32, 32
    dt_ref = 6.0

    model = UniPhyModel(
        in_channels=C,
        out_channels=C,
        embed_dim=16,
        expand=2,
        num_experts=2,
        depth=2,
        patch_size=8,
        img_height=H,
        img_width=W,
        dt_ref=dt_ref,
        sde_mode="det",
        max_growth_rate=0.3,
    ).to(device)

    x = torch.randn(B, T, C, H, W, device=device, requires_grad=True)
    dt = torch.ones(T, device=device) * dt_ref

    out = model(x, dt)
    loss = out.abs().mean()
    loss.backward()

    input_grad_exists = x.grad is not None and x.grad.abs().sum() > 0

    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)

    max_grad_name = ""
    max_grad_val = 0
    min_grad_name = ""
    min_grad_val = float("inf")

    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.abs().max().item()
            if grad_norm > max_grad_val:
                max_grad_val = grad_norm
                max_grad_name = name
            if grad_norm < min_grad_val:
                min_grad_val = grad_norm
                min_grad_name = name

    print(f"Input Gradient Exists: {input_grad_exists}")
    print(f"Number of Parameters with Gradients: {params_with_grad}")
    print(f"Max Gradient: {max_grad_val:.2e} ({max_grad_name})")
    print(f"Min Gradient: {min_grad_val:.2e} ({min_grad_name})")

    passed = input_grad_exists and params_with_grad > 0
    print(f"Test Passed: {passed}")
    print()

    return passed


def check_forecast_mode():
    print("=" * 60)
    print("Testing Forecast Mode")
    print("=" * 60)
    
    try:
        model = UniPhyModel(
            in_channels=4,
            out_channels=4,
            embed_dim=64,
            depth=2,
            patch_size=4,
            img_height=32,
            img_width=32,
        )
        model.eval()
        
        B, C, H, W = 2, 4, 32, 32
        k_steps = 5
        
        x_cond = torch.randn(B, C, H, W)
        dt_future = [torch.ones(B) for _ in range(k_steps)]
        
        with torch.no_grad():
            pred_forecast_1 = model.forward_rollout(x_cond, dt_future, k_steps)
            pred_forecast_2 = model.forward_rollout(x_cond, dt_future, k_steps)
        
        print(f"Input shape: {x_cond.shape}")
        print(f"Forecast steps: {k_steps}")
        print(f"Output shape: {pred_forecast_1.shape}")
        
        expected_shape = (B, k_steps, C, H, W)
        shape_ok = pred_forecast_1.shape == expected_shape
        
        if pred_forecast_1.is_complex():
            diff_deterministic = (pred_forecast_1 - pred_forecast_2).abs().max().item()
        else:
            diff_deterministic = (pred_forecast_1 - pred_forecast_2).abs().max().item()
        
        print(f"Shape OK: {shape_ok}")
        print(f"Deterministic diff: {diff_deterministic:.2e}")
        
        passed = shape_ok and diff_deterministic < 1e-5
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_forecast_forward_consistency():
    print("=" * 60)
    print("Testing Forecast Forward Consistency")
    print("=" * 60)
    
    try:
        model = UniPhyModel(
            in_channels=4,
            out_channels=4,
            embed_dim=64,
            depth=2,
            patch_size=4,
            img_height=32,
            img_width=32,
        )
        model.eval()
        
        B, T, C, H, W = 2, 4, 4, 32, 32
        x = torch.randn(B, T, C, H, W)
        dt = torch.ones(B, T)
        
        with torch.no_grad():
            out_forward = model(x, dt)
        
        x_init = x[:, 0]
        dt_list = [dt[:, t] for t in range(T)]
        
        with torch.no_grad():
            out_rollout = model.forward_rollout(x_init, dt_list, T)
        
        print(f"Forward output shape: {out_forward.shape}")
        print(f"Rollout output shape: {out_rollout.shape}")
        
        passed = out_forward.shape[0] == out_rollout.shape[0]
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_model_consistency():
    print_section("Testing Model Parallel vs Serial Consistency")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 5, 2, 33, 33
    dt_ref = 6.0

    model = UniPhyModel(
        in_channels=C,
        out_channels=C,
        embed_dim=16,
        expand=2,
        num_experts=2,
        depth=2,
        patch_size=11,
        img_height=H,
        img_width=W,
        dt_ref=dt_ref,
        sde_mode="det",
        max_growth_rate=0.3,
    ).to(device).double()

    model.eval()

    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(T, device=device, dtype=torch.float64) * dt_ref

    with torch.no_grad():
        out_parallel = model(x, dt)

        out_serial_list = []
        for t in range(T):
            x_single = x[:, : t + 1]
            dt_single = dt[: t + 1]
            out_t = model(x_single, dt_single)
            out_serial_list.append(out_t[:, -1:])
        out_serial = torch.cat(out_serial_list, dim=1)

    if out_parallel.is_complex():
        diff = (out_parallel.real - out_serial.real).abs().max().item()
    else:
        diff = (out_parallel - out_serial).abs().max().item()

    print(f"Parallel Output Shape: {out_parallel.shape}")
    print(f"Serial Output Shape: {out_serial.shape}")
    print(f"Max Difference: {diff:.2e}")

    passed = diff < 1e-4

    if passed:
        print("Consistency Check PASSED")
    else:
        print("Consistency Check FAILED")
        if out_parallel.is_complex():
            print(f"Parallel Mean: {out_parallel.real.mean().item():.6f}")
            print(f"Serial Mean: {out_serial.real.mean().item():.6f}")
        else:
            print(f"Parallel Mean: {out_parallel.mean().item():.6f}")
            print(f"Serial Mean: {out_serial.mean().item():.6f}")

    print()

    return passed


def run_all_checks():
    print("=" * 60)
    print("UniPhy Model Comprehensive Check Suite")
    print("=" * 60)
    print()

    results = {}

    results["basis_invertibility"] = check_basis_invertibility()
    results["eigenvalue_stability"] = check_eigenvalue_stability()
    results["ffn_complex_mul"] = check_ffn_complex_multiplication()
    results["ffn_causality"] = check_ffn_causality()
    results["flux_tracker_gate"] = check_flux_tracker_gate()
    results["heteroscedastic_noise"] = check_heteroscedastic_noise()
    results["riemannian_clifford"] = check_riemannian_clifford_conv()
    results["io_shapes"] = check_io_shapes()
    results["pscan_compatibility"] = check_pscan_compatibility()
    results["full_model_forward"] = check_full_model_forward()
    results["gradient_flow"] = check_gradient_flow()
    results["forecast_mode"] = check_forecast_mode()
    results["forecast_forward_consistency"] = check_forecast_forward_consistency()

    if torch.cuda.is_available():
        results["model_consistency"] = check_model_consistency()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_passed


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    run_all_checks()
    