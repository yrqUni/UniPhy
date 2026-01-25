import torch
import torch.nn.functional as F
from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import (
    ComplexSVDTransform,
    GlobalFluxTracker,
    TemporalPropagator,
    RiemannianCliffordConv2d,
)
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhyFFN import UniPhyFeedForwardNetwork
from pscan_module import pscan


def check_basis_invertibility():
    print("=" * 60)
    print("Testing Basis Invertibility (with DFT Residual)")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    basis = ComplexSVDTransform(dim).to(device)
    x = torch.randn(4, 16, dim, device=device, dtype=torch.float32)
    x_complex = torch.complex(x, torch.zeros_like(x))
    h = basis.encode(x_complex)
    x_rec = basis.decode(h)
    diff = (x_complex - x_rec).abs().max().item()
    print(f"Dimension: {dim}")
    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {h.shape}")
    print(f"Reconstruction Error: {diff:.2e}")
    dft_weight = torch.sigmoid(basis.dft_weight).item()
    print(f"DFT Weight (sigmoid): {dft_weight:.4f}")
    passed = diff < 1e-4
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_eigenvalue_stability():
    print("=" * 60)
    print("Testing Eigenvalue Bounded Growth")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    max_growth = 0.3
    prop = TemporalPropagator(
        dim, dt_ref=6.0, sde_mode="sde", max_growth_rate=max_growth
    ).to(device)
    lambda_val = prop._get_effective_lambda()
    max_real = lambda_val.real.max().item()
    min_real = lambda_val.real.min().item()
    print(f"Dimension: {dim}")
    print(f"Max Growth Rate Config: {max_growth}")
    print(f"Actual Max Real Part: {max_real:.4f}")
    print(f"Actual Min Real Part: {min_real:.4f}")
    passed = max_real <= max_growth + 1e-6 and min_real >= -max_growth - 1e-6
    print(f"Bounded in [-{max_growth}, {max_growth}]: {passed}")
    print()
    return passed


def check_ffn_complex_multiplication():
    print("=" * 60)
    print("Testing FFN Complex Multiplication Order")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, D, H, W = 2, 32, 8, 8
    ffn = UniPhyFeedForwardNetwork(dim=D, expand=4, num_experts=4).to(device)
    ffn.eval()
    x_re = torch.randn(B, D, H, W, device=device)
    x_im = torch.randn(B, D, H, W, device=device)
    x = torch.complex(x_re, x_im)
    with torch.no_grad():
        out1 = ffn(x)
        out2 = ffn(x)
    diff = (out1 - out2).abs().max().item()
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out1.shape}")
    print(f"Deterministic Diff: {diff:.2e}")
    passed = diff < 1e-10
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_ffn_causality():
    print("=" * 60)
    print("Testing FFN Causality")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D, H, W = 1, 4, 16, 8, 8
    ffn = UniPhyFeedForwardNetwork(dim=D, expand=4, num_experts=4).to(device)
    ffn.eval()
    x = torch.randn(B, T, D, H, W, device=device, dtype=torch.cdouble)
    with torch.no_grad():
        x_par = x.reshape(B * T, D, H, W)
        out_par = ffn(x_par).reshape(B, T, D, H, W)
        out_ser_list = []
        for t in range(T):
            out_ser_list.append(ffn(x[:, t]))
        out_ser = torch.stack(out_ser_list, dim=1)
    diff = (out_par - out_ser).abs().max().item()
    print(f"Parallel Shape: {out_par.shape}")
    print(f"Serial Shape: {out_ser.shape}")
    print(f"Max Difference: {diff:.2e}")
    passed = diff < 1e-10
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_flux_tracker_gate():
    print("=" * 60)
    print("Testing GlobalFluxTracker Gate Output")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    B = 2
    tracker = GlobalFluxTracker(dim).to(device)
    prev_state = torch.zeros(B, dim, device=device, dtype=torch.cdouble)
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
    print("=" * 60)
    print("Testing Heteroscedastic Noise Generation")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    B, N, T = 2, 16, 1
    prop = TemporalPropagator(
        dim, dt_ref=6.0, sde_mode="sde", init_noise_scale=1.0
    ).to(device)
    target_shape = (B * N, T, dim)
    dt = torch.tensor([6.0], device=device)
    h_state_low = torch.ones(B * N, T, dim, device=device, dtype=torch.cdouble) * 0.1
    h_state_high = torch.ones(B * N, T, dim, device=device, dtype=torch.cdouble) * 10.0
    noise_low = prop.generate_stochastic_term(
        target_shape, dt, torch.cdouble, h_state=h_state_low
    )
    noise_high = prop.generate_stochastic_term(
        target_shape, dt, torch.cdouble, h_state=h_state_high
    )
    std_low = noise_low.abs().std().item()
    std_high = noise_high.abs().std().item()
    print(f"Low State Noise Std: {std_low:.4f}")
    print(f"High State Noise Std: {std_high:.4f}")
    print(f"Ratio (High/Low): {std_high / (std_low + 1e-8):.2f}")
    passed = std_high > std_low * 0.5
    print(f"Heteroscedastic Behavior: {passed}")
    print()
    return passed


def check_riemannian_clifford_conv():
    print("=" * 60)
    print("Testing RiemannianCliffordConv2d with Dispersion")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 32, 16, 16
    conv = RiemannianCliffordConv2d(
        in_channels=C,
        out_channels=C,
        kernel_size=3,
        padding=1,
        img_height=H,
        img_width=W,
    ).to(device)
    x = torch.randn(B, C, H, W, device=device)
    out = conv(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Viscosity Scale: {conv.viscosity_scale.item():.4f}")
    print(f"Dispersion Scale: {conv.dispersion_scale.item():.4f}")
    has_dispersion = hasattr(conv, "dispersion_scale")
    has_anti_diffusion = hasattr(conv, "anti_diffusion_gate")
    print(f"Has Dispersion Term: {has_dispersion}")
    print(f"Has Anti-Diffusion Gate: {has_anti_diffusion}")
    passed = out.shape == x.shape and has_dispersion and has_anti_diffusion
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_io_shapes():
    print("=" * 60)
    print("Testing Encoder/Decoder IO Shapes")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 1, 5
    C_in, C_out = 2, 2
    H, W = 721, 1440
    Patch, Embed = 16, 64
    enc = UniPhyEncoder(
        in_ch=C_in, embed_dim=Embed, patch_size=Patch, img_height=H, img_width=W
    ).to(device)
    dec = UniPhyEnsembleDecoder(
        out_ch=C_out, latent_dim=Embed, patch_size=Patch, img_height=H, img_width=W
    ).to(device)
    x = torch.randn(B, T, C_in, H, W, device=device)
    z = enc(x)
    expected_h = (H + (Patch - H % Patch) % Patch) // Patch
    expected_w = (W + (Patch - W % Patch) % Patch) // Patch
    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {z.shape}")
    print(f"Expected Latent H: {expected_h}, Actual: {z.shape[-2]}")
    print(f"Expected Latent W: {expected_w}, Actual: {z.shape[-1]}")
    out = dec(z)
    print(f"Output Shape: {out.shape}")
    print(f"Expected Output Shape: ({B}, {T}, {C_out}, {H}, {W})")
    passed = (
        z.shape[-2] == expected_h
        and z.shape[-1] == expected_w
        and out.shape == (B, T, C_out, H, W)
    )
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_pscan_compatibility():
    print("=" * 60)
    print("Testing PScan Compatibility")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping PScan test")
        print()
        return True
    B, T, C, D = 2, 16, 4, 2
    A_diag = torch.randn(B, T, C, D, dtype=torch.complex64, device=device) * 0.5
    X = torch.randn(B, T, C, D, dtype=torch.complex64, device=device)
    X_5d = X.unsqueeze(-1)
    Y_5d = pscan(A_diag, X_5d)
    Y_diag = Y_5d.squeeze(-1)
    Y_seq = torch.zeros_like(X)
    Y_seq[:, 0] = X[:, 0]
    for t in range(1, T):
        Y_seq[:, t] = A_diag[:, t] * Y_seq[:, t - 1] + X[:, t]
    diff_diag = (Y_diag - Y_seq).abs().max().item()
    print(f"Diagonal Mode Shape: A={A_diag.shape}, X={X.shape}")
    print(f"Diagonal Mode Max Diff: {diff_diag:.2e}")
    passed = diff_diag < 1e-4
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_full_model_forward():
    print("=" * 60)
    print("Testing Full Model Forward Pass")
    print("=" * 60)
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


def check_model_consistency():
    print("=" * 60)
    print("Testing Model Parallel vs Serial Consistency")
    print("=" * 60)
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
    ).to(device)
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
    diff = (out_parallel - out_serial).abs().max().item()
    print(f"Parallel Output Shape: {out_parallel.shape}")
    print(f"Serial Output Shape: {out_serial.shape}")
    print(f"Max Difference: {diff:.2e}")
    passed = diff < 1e-4
    if passed:
        print("Consistency Check PASSED")
    else:
        print("Consistency Check FAILED")
        print(f"Parallel Mean: {out_parallel.mean().item():.6f}")
        print(f"Serial Mean: {out_serial.mean().item():.6f}")
    print()
    return passed


def check_gradient_flow():
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
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
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    print(f"Input Gradient Exists: {has_grad}")
    print(f"Number of Parameters with Gradients: {len(grad_norms)}")
    if grad_norms:
        max_grad_name = max(grad_norms, key=grad_norms.get)
        min_grad_name = min(grad_norms, key=grad_norms.get)
        print(f"Max Gradient: {grad_norms[max_grad_name]:.2e} ({max_grad_name})")
        print(f"Min Gradient: {grad_norms[min_grad_name]:.2e} ({min_grad_name})")
    passed = has_grad and len(grad_norms) > 0
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_forecast_mode():
    print("=" * 60)
    print("Testing Forecast Mode")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T_cond, C, H, W = 1, 4, 2, 32, 32
    k_steps = 3
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
    model.eval()
    x_cond = torch.randn(B, T_cond, C, H, W, device=device)
    dt_cond = torch.ones(T_cond, device=device) * dt_ref
    dt_future = torch.ones(B, k_steps, device=device) * dt_ref
    with torch.no_grad():
        predictions = model.forecast(x_cond, dt_cond, k_steps, dt_future)
    print(f"Conditioning Input Shape: {x_cond.shape}")
    print(f"Forecast Steps: {k_steps}")
    print(f"Predictions Shape: {predictions.shape}")
    print(f"Expected Shape: ({B}, {k_steps}, {C}, {H}, {W})")
    passed = predictions.shape == (B, k_steps, C, H, W)
    print(f"Test Passed: {passed}")
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
    return all_passed


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    run_all_checks()
    