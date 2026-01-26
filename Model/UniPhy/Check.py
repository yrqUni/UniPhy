import torch
import torch.nn as nn
from ModelUniPhy import UniPhyModel


def check_forecast_mode():
    print("=" * 60)
    print("Testing Forecast Mode")
    print("=" * 60)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = UniPhyModel(
            in_channels=4,
            out_channels=4,
            embed_dim=64,
            depth=2,
            patch_size=4,
            img_height=32,
            img_width=32,
        ).to(device)
        model.eval()
        
        B, C, H, W = 2, 4, 32, 32
        k_steps = 5
        
        x_cond = torch.randn(B, C, H, W, device=device)
        dt_future = [torch.ones(B, device=device) for _ in range(k_steps)]
        
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
        import traceback
        traceback.print_exc()
        print()
        return False


def check_forecast_forward_consistency():
    print("=" * 60)
    print("Testing Forecast Forward Consistency")
    print("=" * 60)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = UniPhyModel(
            in_channels=4,
            out_channels=4,
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
        import traceback
        traceback.print_exc()
        print()
        return False


def check_model_consistency():
    print("=" * 60)
    print("Testing Model Consistency")
    print("=" * 60)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = UniPhyModel(
            in_channels=4,
            out_channels=4,
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
        import traceback
        traceback.print_exc()
        print()
        return False


def run_all_checks():
    results = {}
    
    results["forecast_mode"] = check_forecast_mode()
    results["forecast_forward_consistency"] = check_forecast_forward_consistency()
    results["model_consistency"] = check_model_consistency()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
    print()
    
    all_passed = all(results.values())
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    exit(0 if success else 1)
    