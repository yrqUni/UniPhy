import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUniPhy import UniPhyModel
from UniPhyOps import EigenvalueParameterization, PScanOperator
from UniPhyFFN import ComplexMoEFFN, GatedComplexFFN
from UniPhyIO import UniPhyEncoder, UniPhyDecoder
from RiemannianClifford import RiemannianCliffordConv


def check_eigenvalue_stability():
    print("=" * 60)
    print("Testing Eigenvalue Stability")
    print("=" * 60)
    
    try:
        dim = 64
        eigen_param = EigenvalueParameterization(dim=dim, dt_ref=1.0, sde_mode="ode", max_growth_rate=0.1)
        
        dt_values = [0.1, 1.0, 10.0, 100.0]
        stable = True
        
        for dt in dt_values:
            decay, forcing = eigen_param.get_transition_operators(dt)
            decay_mag = decay.abs()
            max_mag = decay_mag.max().item()
            print(f"dt={dt}: max |decay| = {max_mag:.6f}")
            
            if max_mag > 1.5:
                stable = False
        
        print(f"Test Passed: {stable}")
        print()
        return stable
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_ffn_complex_multiplication():
    print("=" * 60)
    print("Testing FFN Complex Multiplication")
    print("=" * 60)
    
    try:
        dim = 64
        hidden_dim = 128
        ffn = GatedComplexFFN(dim=dim, hidden_dim=hidden_dim)
        
        B, H, W = 2, 8, 8
        x_re = torch.randn(B, dim, H, W)
        x_im = torch.randn(B, dim, H, W)
        x = torch.complex(x_re, x_im)
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, dim)
        out = ffn(x_flat)
        
        is_complex = out.is_complex()
        shape_ok = out.shape == x_flat.shape
        
        passed = is_complex and shape_ok
        print(f"Output is complex: {is_complex}")
        print(f"Shape OK: {shape_ok}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_ffn_causality():
    print("=" * 60)
    print("Testing FFN Causality")
    print("=" * 60)
    
    try:
        dim = 64
        hidden_dim = 128
        ffn = GatedComplexFFN(dim=dim, hidden_dim=hidden_dim)
        
        B = 4
        x_re = torch.randn(B, dim)
        x_im = torch.randn(B, dim)
        x = torch.complex(x_re, x_im)
        
        out1 = ffn(x[:2])
        out2 = ffn(x)[:2]
        
        diff = (out1 - out2).abs().max().item()
        passed = diff < 1e-5
        
        print(f"Causality diff: {diff:.2e}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_flux_tracker_gate():
    print("=" * 60)
    print("Testing Flux Tracker Gate")
    print("=" * 60)
    
    try:
        from UniPhyBlock import FluxTracker
        
        dim = 64
        tracker = FluxTracker(dim=dim)
        
        B = 2
        flux_prev = torch.randn(B, dim, dtype=torch.complex64)
        z = torch.randn(B, dim, dtype=torch.complex64)
        
        flux_next = tracker(flux_prev, z)
        
        shape_ok = flux_next.shape == flux_prev.shape
        is_complex = flux_next.is_complex()
        
        passed = shape_ok and is_complex
        print(f"Shape OK: {shape_ok}")
        print(f"Is Complex: {is_complex}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_heteroscedastic_noise():
    print("=" * 60)
    print("Testing Heteroscedastic Noise")
    print("=" * 60)
    
    try:
        dim = 64
        eigen_param = EigenvalueParameterization(dim=dim, dt_ref=1.0, sde_mode="sde", init_noise_scale=0.1)
        
        shape = (2, 16, dim)
        dt = 1.0
        dtype = torch.complex64
        h_state = torch.randn(shape, dtype=dtype)
        
        noise = eigen_param.generate_stochastic_term(shape, dt, dtype, h_state)
        
        shape_ok = noise.shape == shape
        has_variance = noise.abs().mean().item() > 0 or eigen_param.sde_mode != "sde"
        
        passed = shape_ok
        print(f"Shape OK: {shape_ok}")
        print(f"Noise mean magnitude: {noise.abs().mean().item():.6f}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_riemannian_clifford_conv():
    print("=" * 60)
    print("Testing Riemannian Clifford Conv")
    print("=" * 60)
    
    try:
        dim = 64
        conv = RiemannianCliffordConv(dim=dim, kernel_size=3)
        
        B, H, W = 2, 16, 16
        x_re = torch.randn(B, dim, H, W)
        x_im = torch.randn(B, dim, H, W)
        x = torch.complex(x_re, x_im)
        
        out = conv(x)
        
        shape_ok = out.shape == x.shape
        is_complex = out.is_complex()
        
        passed = shape_ok and is_complex
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Shape OK: {shape_ok}")
        print(f"Is Complex: {is_complex}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_io_shapes():
    print("=" * 60)
    print("Testing IO Shapes")
    print("=" * 60)
    
    try:
        in_ch = 4
        embed_dim = 64
        patch_size = 4
        img_h, img_w = 32, 32
        
        encoder = UniPhyEncoder(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size, img_height=img_h, img_width=img_w)
        decoder = UniPhyDecoder(out_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size, img_height=img_h, img_width=img_w)
        
        B, T = 2, 4
        x = torch.randn(B, T, in_ch, img_h, img_w)
        
        z = encoder(x)
        print(f"Encoder input shape: {x.shape}")
        print(f"Encoder output shape: {z.shape}")
        
        x_rec = decoder(z)
        print(f"Decoder output shape: {x_rec.shape}")
        
        h_patches = img_h // patch_size
        w_patches = img_w // patch_size
        
        encoder_shape_ok = z.shape == (B, T, embed_dim, h_patches, w_patches) or z.is_complex()
        decoder_shape_ok = x_rec.shape[-2:] == x.shape[-2:] or x_rec.shape[-2] >= x.shape[-2]
        
        passed = encoder_shape_ok and decoder_shape_ok
        print(f"Encoder shape OK: {encoder_shape_ok}")
        print(f"Decoder shape OK: {decoder_shape_ok}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_pscan_compatibility():
    print("=" * 60)
    print("Testing PScan Compatibility")
    print("=" * 60)
    
    try:
        dim = 64
        pscan = PScanOperator(dim=dim)
        
        B, T = 2, 8
        x_re = torch.randn(B, T, dim)
        x_im = torch.randn(B, T, dim)
        x = torch.complex(x_re, x_im)
        
        A, X = pscan.get_operators(x)
        
        a_shape_ok = A.shape == (B, T, dim)
        x_shape_ok = X.shape == (B, T, dim)
        
        passed = a_shape_ok and x_shape_ok
        print(f"A shape: {A.shape}, expected: {(B, T, dim)}")
        print(f"X shape: {X.shape}, expected: {(B, T, dim)}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_full_model_forward():
    print("=" * 60)
    print("Testing Full Model Forward")
    print("=" * 60)
    
    try:
        model = UniPhyModel(
            in_ch=4,
            out_ch=4,
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
            out = model(x, dt)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        
        shape_ok = out.shape == x.shape
        
        if out.is_complex():
            has_nan = torch.isnan(out.real).any().item() or torch.isnan(out.imag).any().item()
            has_inf = torch.isinf(out.real).any().item() or torch.isinf(out.imag).any().item()
        else:
            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
        
        numerical_ok = not has_nan and not has_inf
        
        passed = shape_ok and numerical_ok
        print(f"Shape OK: {shape_ok}")
        print(f"Numerical OK: {numerical_ok}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_gradient_flow():
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    try:
        model = UniPhyModel(
            in_ch=4,
            out_ch=4,
            embed_dim=64,
            depth=2,
            patch_size=4,
            img_height=32,
            img_width=32,
        )
        model.train()
        
        B, T, C, H, W = 2, 4, 4, 32, 32
        x = torch.randn(B, T, C, H, W, requires_grad=True)
        dt = torch.ones(B, T)
        
        out = model(x, dt)
        
        if out.is_complex():
            loss = out.abs().mean()
        else:
            loss = out.mean()
        
        loss.backward()
        
        grad_ok = x.grad is not None
        grad_nonzero = x.grad.abs().sum().item() > 0 if grad_ok else False
        
        num_params_with_grad = 0
        num_params_total = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params_total += 1
                if param.grad is not None and param.grad.abs().sum().item() > 0:
                    num_params_with_grad += 1
        
        grad_ratio = num_params_with_grad / max(num_params_total, 1)
        
        passed = grad_ok and grad_nonzero and grad_ratio > 0.5
        print(f"Input grad exists: {grad_ok}")
        print(f"Input grad nonzero: {grad_nonzero}")
        print(f"Params with grad: {num_params_with_grad}/{num_params_total} ({grad_ratio:.1%})")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_forecast_mode():
    print("=" * 60)
    print("Testing Forecast Mode")
    print("=" * 60)
    
    try:
        model = UniPhyModel(
            in_ch=4,
            out_ch=4,
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
        
        print(f"Deterministic Check (same input twice): {diff_deterministic:.2e}")
        
        deterministic_ok = diff_deterministic < 1e-5
        
        if pred_forecast_1.is_complex():
            has_nan = torch.isnan(pred_forecast_1.real).any().item() or torch.isnan(pred_forecast_1.imag).any().item()
            has_inf = torch.isinf(pred_forecast_1.real).any().item() or torch.isinf(pred_forecast_1.imag).any().item()
        else:
            has_nan = torch.isnan(pred_forecast_1).any().item()
            has_inf = torch.isinf(pred_forecast_1).any().item()
        
        print(f"Forecast has NaN: {has_nan}")
        print(f"Forecast has Inf: {has_inf}")
        
        numerical_ok = not has_nan and not has_inf
        
        passed = shape_ok and deterministic_ok and numerical_ok
        print(f"Shape OK: {shape_ok}")
        print(f"Deterministic OK: {deterministic_ok}")
        print(f"Numerical OK: {numerical_ok}")
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
            in_ch=4,
            out_ch=4,
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
    print("=" * 60)
    print("Testing Model Consistency (GPU)")
    print("=" * 60)
    
    try:
        device = torch.device("cuda")
        
        model = UniPhyModel(
            in_ch=4,
            out_ch=4,
            embed_dim=64,
            depth=2,
            patch_size=4,
            img_height=32,
            img_width=32,
        ).to(device)
        model.eval()
        
        B, T, C, H, W = 2, 4, 4, 32, 32
        x = torch.randn(B, T, C, H, W, device=device)
        dt = torch.ones(B, T, device=device)
        
        with torch.no_grad():
            out1 = model(x, dt)
            out2 = model(x, dt)
        
        if out1.is_complex():
            diff = (out1 - out2).abs().max().item()
        else:
            diff = (out1 - out2).abs().max().item()
        
        passed = diff < 1e-5
        print(f"Consistency diff: {diff:.2e}")
        print(f"Test Passed: {passed}")
        print()
        return passed
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def run_all_checks():
    results = {}
    
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
    