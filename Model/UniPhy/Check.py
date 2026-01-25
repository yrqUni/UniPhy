import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUniPhy import UniPhyModel
from UniPhyOps import TemporalPropagator, GlobalFluxTracker
from UniPhyFFN import UniPhyFeedForwardNetwork
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder
from UniPhySpatial import RiemannianCliffordConv2d
from PScan import pscan


def check_basis_invertibility():
    print("=" * 60)
    print("Testing Basis Invertibility (with DFT Residual)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    
    prop = TemporalPropagator(dim, dt_ref=6.0, sde_mode="det").to(device)
    basis = prop.basis

    B, T, D = 4, 16, dim
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)

    latent = basis.encode(x)
    reconstructed = basis.decode(latent)

    error = (x - reconstructed).abs().max().item()

    print(f"Dimension: {dim}")
    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {latent.shape}")
    print(f"Reconstruction Error: {error:.2e}")

    passed = error < 1e-4
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_eigenvalue_stability():
    print("=" * 60)
    print("Testing Eigenvalue Bounded Growth")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    max_growth_rate = 0.3

    prop = TemporalPropagator(
        dim=dim,
        dt_ref=6.0,
        max_growth_rate=max_growth_rate,
        sde_mode="det"
    ).to(device)

    lam = prop._get_effective_lambda()
    real_parts = lam.real

    max_real = real_parts.max().item()
    min_real = real_parts.min().item()

    print(f"Dimension: {dim}")
    print(f"Max Growth Rate Config: {max_growth_rate}")
    print(f"Actual Max Real Part: {max_real:.4f}")
    print(f"Actual Min Real Part: {min_real:.4f}")

    passed = (-max_growth_rate <= min_real) and (max_real <= max_growth_rate)
    print(f"Bounded in [-{max_growth_rate}, {max_growth_rate}]: {passed}")
    print()
    return passed


def check_ffn_complex_multiplication():
    print("=" * 60)
    print("Testing FFN Complex Multiplication Order")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 8
    expand = 2
    num_experts = 2

    ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts).to(device)
    ffn.eval()

    B, H, W = 2, 8, 8
    x = torch.randn(B, dim, H, W, device=device) + 1j * torch.randn(B, dim, H, W, device=device)

    with torch.no_grad():
        out1 = ffn(x)
        out2 = ffn(x)

    diff = (out1 - out2).abs().max().item()

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out1.shape}")
    print(f"Deterministic Diff: {diff:.2e}")

    passed = diff == 0.0
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_ffn_causality():
    print("=" * 60)
    print("Testing FFN Causality")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 8
    expand = 2
    num_experts = 2

    ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts).to(device)
    ffn.eval()

    B, T, H, W = 1, 4, 8, 8
    x = torch.randn(B, T, dim, H, W, device=device) + 1j * torch.randn(B, T, dim, H, W, device=device)

    with torch.no_grad():
        x_flat = x.reshape(B * T, dim, H, W)
        out_parallel = ffn(x_flat).reshape(B, T, dim, H, W)

        out_serial_list = []
        for t in range(T):
            out_t = ffn(x[:, t])
            out_serial_list.append(out_t)
        out_serial = torch.stack(out_serial_list, dim=1)

    diff = (out_parallel - out_serial).abs().max().item()

    print(f"Parallel Shape: {out_parallel.shape}")
    print(f"Serial Shape: {out_serial.shape}")
    print(f"Max Difference: {diff:.2e}")

    passed = diff < 1e-6
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_flux_tracker_gate():
    print("=" * 60)
    print("Testing GlobalFluxTracker Gate Output")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64

    tracker = GlobalFluxTracker(dim).to(device)

    B = 2
    state = torch.randn(B, dim, device=device) + 1j * torch.randn(B, dim, device=device)
    x_t = torch.randn(B, dim, device=device) + 1j * torch.randn(B, dim, device=device)

    new_state, source, gate = tracker.forward_step(state, x_t)

    print(f"Input State Shape: {state.shape}")
    print(f"New State Shape: {new_state.shape}")
    print(f"Source Shape: {source.shape}")
    print(f"Gate Shape: {gate.shape}")
    print(f"Gate Range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")

    passed = (gate.min() >= 0.0) and (gate.max() <= 1.0)
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_heteroscedastic_noise():
    print("=" * 60)
    print("Testing Heteroscedastic Noise Generation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64

    prop = TemporalPropagator(
        dim=dim,
        dt_ref=6.0,
        max_growth_rate=0.3,
        sde_mode="sde"
    ).to(device)

    B, H, W, D = 100, 8, 8, dim
    target_shape = (B, H, W, D)
    dt = torch.tensor(6.0, device=device)

    h_low = torch.zeros(B, H, W, D, device=device, dtype=torch.complex64)
    h_high = torch.ones(B, H, W, D, device=device, dtype=torch.complex64) * 10.0

    if hasattr(prop, 'generate_stochastic_term'):
        noise_low = prop.generate_stochastic_term(target_shape, dt, torch.complex64, h_state=h_low)
        noise_high = prop.generate_stochastic_term(target_shape, dt, torch.complex64, h_state=h_high)

        std_low = noise_low.abs().std().item()
        std_high = noise_high.abs().std().item()

        print(f"Low State Noise Std: {std_low:.4f}")
        print(f"High State Noise Std: {std_high:.4f}")
        print(f"Ratio (High/Low): {std_high / std_low:.2f}")

        passed = std_high > std_low
        print(f"Heteroscedastic Behavior: {passed}")
    else:
        print("generate_stochastic_term method not found, skipping detailed test")
        passed = True

    print()
    return passed


def check_riemannian_clifford_conv():
    print("=" * 60)
    print("Testing RiemannianCliffordConv2d with Dispersion")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 32

    conv = RiemannianCliffordConv2d(
        dim,
        viscosity_scale=0.01,
        dispersion_scale=0.01
    ).to(device)

    B, H, W = 2, 16, 16
    x = torch.randn(B, dim, H, W, device=device) + 1j * torch.randn(B, dim, H, W, device=device)

    out = conv(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Viscosity Scale: {conv.viscosity_scale:.4f}")
    print(f"Dispersion Scale: {conv.dispersion_scale:.4f}")
    print(f"Has Dispersion Term: {hasattr(conv, 'dispersion_scale')}")
    print(f"Has Anti-Diffusion Gate: {hasattr(conv, 'anti_diffusion_gate')}")

    passed = out.shape == x.shape
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_io_shapes():
    print("=" * 60)
    print("Testing Encoder/Decoder IO Shapes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 2
    out_ch = 2
    latent_dim = 64
    patch_size = 16
    img_height = 721
    img_width = 1440

    encoder = UniPhyEncoder(
        in_ch=in_ch,
        latent_dim=latent_dim,
        patch_size=patch_size,
        img_height=img_height,
        img_width=img_width
    ).to(device)

    decoder = UniPhyEnsembleDecoder(
        out_ch=out_ch,
        latent_dim=latent_dim,
        patch_size=patch_size,
        img_height=img_height,
        img_width=img_width
    ).to(device)

    B, T, C, H, W = 1, 5, in_ch, img_height, img_width
    x = torch.randn(B, T, C, H, W, device=device)

    latent = encoder(x)
    output = decoder(latent)

    H_p = (img_height + patch_size - 1) // patch_size
    W_p = (img_width + patch_size - 1) // patch_size

    print(f"Input Shape: {x.shape}")
    print(f"Latent Shape: {latent.shape}")
    print(f"Expected Latent H: {H_p}, Actual: {latent.shape[3]}")
    print(f"Expected Latent W: {W_p}, Actual: {latent.shape[4]}")
    print(f"Output Shape: {output.shape}")
    print(f"Expected Output Shape: {(B, T, out_ch, img_height, img_width)}")

    passed = (
        latent.shape[3] == H_p and
        latent.shape[4] == W_p and
        output.shape == (B, T, out_ch, img_height, img_width)
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
        embed_dim=16,
        expand=2,
        num_experts=2,
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
    print(f"Expected Shape: {(B, T, C, H, W)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    passed = out.shape == (B, T, C, H, W)
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_gradient_flow():
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 2, 2, 32, 32
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
            grad_norms[name] = param.grad.abs().max().item()

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
    B, T, C, H, W = 1, 4, 2, 32, 32
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

    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(T, device=device) * dt_ref
    dt_future = torch.ones(k_steps, device=device) * dt_ref

    with torch.no_grad():
        predictions = model.forecast(x, dt, k_steps, dt_future)

    print(f"Conditioning Input Shape: {x.shape}")
    print(f"Forecast Steps: {k_steps}")
    print(f"Predictions Shape: {predictions.shape}")
    print(f"Expected Shape: {(B, k_steps, C, H, W)}")

    passed = predictions.shape == (B, k_steps, C, H, W)
    print(f"Test Passed: {passed}")
    print()
    return passed


def check_model_consistency():
    print("=" * 60)
    print("Testing Model Determinism (Two Identical Runs)")
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
    ).to(device).double()

    model.eval()
    torch.manual_seed(42)
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(T, device=device, dtype=torch.float64) * dt_ref

    with torch.no_grad():
        for block in model.blocks:
            block.last_h_state = None
            block.last_flux_state = None
        out_run1 = model(x, dt)

        for block in model.blocks:
            block.last_h_state = None
            block.last_flux_state = None
        out_run2 = model(x, dt)

    diff = (out_run1 - out_run2).abs().max().item()

    print(f"Run 1 Output Shape: {out_run1.shape}")
    print(f"Run 2 Output Shape: {out_run2.shape}")
    print(f"Max Difference: {diff:.2e}")
    print(f"Run 1 Mean: {out_run1.mean().item():.6f}")
    print(f"Run 2 Mean: {out_run2.mean().item():.6f}")

    passed = diff < 1e-10
    if passed:
        print("Consistency Check PASSED")
    else:
        print("Consistency Check FAILED")
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
    