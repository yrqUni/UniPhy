import sys
import torch
import torch.nn as nn

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel


def create_model(device):
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        expand=2,
        num_experts=2,
        depth=2,
        patch_size=4,
        img_height=32,
        img_width=32,
        dt_ref=6.0,
        sde_mode="sde",
        init_noise_scale=1.0,
        max_growth_rate=0.3,
    ).to(device)
    return model


def test_single_forward():
    print("=" * 60)
    print("Test 1: Single Forward + Backward")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        out = model(x, dt)
        
        if out.is_complex():
            out_real = out.real
        else:
            out_real = out
        
        loss = (out_real - target).abs().mean()
        loss.backward()
        
        print(f"Output shape: {out.shape}")
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_double_forward():
    print("=" * 60)
    print("Test 2: Double Forward + Backward")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        out1 = model(x, dt)
        out2 = model(x, dt)
        
        if out1.is_complex():
            out1_real = out1.real
            out2_real = out2.real
        else:
            out1_real = out1
            out2_real = out2
        
        loss = (out1_real - target).abs().mean() + (out2_real - target).abs().mean()
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_ensemble_forward():
    print("=" * 60)
    print("Test 3: Ensemble Forward + Backward")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        ensemble_preds = []
        for _ in range(3):
            out = model(x, dt)
            if out.is_complex():
                ensemble_preds.append(out.real)
            else:
                ensemble_preds.append(out)
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        ensemble_mean = ensemble_stack.mean(dim=0)
        
        loss = (ensemble_mean - target).abs().mean()
        loss.backward()
        
        print(f"Ensemble shape: {ensemble_stack.shape}")
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_ensemble_with_clone():
    print("=" * 60)
    print("Test 4: Ensemble Forward with Clone")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        ensemble_preds = []
        for _ in range(3):
            out = model(x.clone(), dt.clone())
            if out.is_complex():
                ensemble_preds.append(out.real.clone())
            else:
                ensemble_preds.append(out.clone())
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        ensemble_mean = ensemble_stack.mean(dim=0)
        
        loss = (ensemble_mean - target).abs().mean()
        loss.backward()
        
        print(f"Ensemble shape: {ensemble_stack.shape}")
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_with_anomaly_detection():
    print("=" * 60)
    print("Test 5: Anomaly Detection")
    print("=" * 60)
    
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        ensemble_preds = []
        for i in range(3):
            out = model(x, dt)
            if out.is_complex():
                ensemble_preds.append(out.real)
            else:
                ensemble_preds.append(out)
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        ensemble_mean = ensemble_stack.mean(dim=0)
        
        loss = (ensemble_mean - target).abs().mean()
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        torch.autograd.set_detect_anomaly(False)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        torch.autograd.set_detect_anomaly(False)
        return False


def test_det_mode():
    print("=" * 60)
    print("Test 6: Deterministic Mode (no SDE)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        expand=2,
        num_experts=2,
        depth=2,
        patch_size=4,
        img_height=32,
        img_width=32,
        dt_ref=6.0,
        sde_mode="det",
        init_noise_scale=0.0,
        max_growth_rate=0.3,
    ).to(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    target = torch.randn(B, T, C, H, W, device=device)
    
    try:
        ensemble_preds = []
        for _ in range(3):
            out = model(x, dt)
            if out.is_complex():
                ensemble_preds.append(out.real)
            else:
                ensemble_preds.append(out)
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        ensemble_mean = ensemble_stack.mean(dim=0)
        
        loss = (ensemble_mean - target).abs().mean()
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_gradient_accumulation():
    print("=" * 60)
    print("Test 7: Gradient Accumulation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    B, T, C, H, W = 2, 4, 4, 32, 32
    
    try:
        optimizer.zero_grad()
        
        for step in range(4):
            x = torch.randn(B, T, C, H, W, device=device)
            dt = torch.ones(B, T, device=device) * 0.1
            target = torch.randn(B, T, C, H, W, device=device)
            
            out = model(x, dt)
            if out.is_complex():
                out_real = out.real
            else:
                out_real = out
            
            loss = (out_real - target).abs().mean() / 4
            loss.backward()
        
        optimizer.step()
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_full_training_step():
    print("=" * 60)
    print("Test 8: Full Training Step (like train.py)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    B, T, C, H, W = 2, 8, 4, 32, 32
    data = torch.randn(B, T, C, H, W, device=device)
    dt_data = torch.ones(B, T, device=device) * 0.1
    
    ensemble_size = 3
    ensemble_weight = 0.5
    
    try:
        x_input = data[:, :-1]
        x_target = data[:, 1:]
        dt_input = dt_data[:, :-1]
        
        out = model(x_input, dt_input)
        
        if out.is_complex():
            out_real = out.real
        else:
            out_real = out
        
        l1_loss = (out_real - x_target).abs().mean()
        
        ensemble_preds = [out_real]
        for _ in range(ensemble_size - 1):
            out_ens = model(x_input, dt_input)
            if out_ens.is_complex():
                ensemble_preds.append(out_ens.real)
            else:
                ensemble_preds.append(out_ens)
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        mae = (ensemble_stack - x_target.unsqueeze(0)).abs().mean(dim=0)
        crps_loss = mae.mean()
        
        loss = (1 - ensemble_weight) * l1_loss + ensemble_weight * crps_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"L1 Loss: {l1_loss.item():.4f}")
        print(f"CRPS Loss: {crps_loss.item():.4f}")
        print(f"Total Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("UniPhy Debug Script - Inplace Operation Check")
    print("=" * 60 + "\n")
    
    results = {}
    
    results["single_forward"] = test_single_forward()
    print()
    
    results["double_forward"] = test_double_forward()
    print()
    
    results["ensemble_forward"] = test_ensemble_forward()
    print()
    
    results["ensemble_with_clone"] = test_ensemble_with_clone()
    print()
    
    results["anomaly_detection"] = test_with_anomaly_detection()
    print()
    
    results["det_mode"] = test_det_mode()
    print()
    
    results["gradient_accumulation"] = test_gradient_accumulation()
    print()
    
    results["full_training_step"] = test_full_training_step()
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
    
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("All tests passed!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed tests: {failed}")
        print("\nPossible fixes:")
        print("1. Add .clone() after slicing operations")
        print("2. Add .clone() after .real accessor")
        print("3. Check inplace operations in model modules")
        print("4. Try sde_mode='det' to isolate SDE-related issues")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
    