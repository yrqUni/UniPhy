import sys
import torch
import traceback

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel


def test_large_model():
    print("=" * 60)
    print("Test: Large Model (embed_dim=512) - Same as train.yaml")
    print("=" * 60)
    
    device = torch.device("cuda")
    
    model = UniPhyModel(
        in_channels=30,
        out_channels=30,
        embed_dim=512,
        expand=4,
        num_experts=8,
        depth=8,
        patch_size=32,
        img_height=721,
        img_width=1440,
        dt_ref=6.0,
        sde_mode="sde",
        init_noise_scale=1.0,
        max_growth_rate=0.3,
    ).to(device)
    model.train()
    
    B, T, C, H, W = 1, 4, 30, 721, 1440
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 6.0
    target = torch.randn(B, T, C, H, W, device=device)
    
    print("Testing single forward + backward...")
    try:
        out = model(x, dt)
        if out.is_complex():
            out_real = out.real
        else:
            out_real = out
        loss = (out_real - target).abs().mean()
        loss.backward()
        print("Single forward: PASSED")
    except Exception as e:
        print(f"Single forward: FAILED - {e}")
        traceback.print_exc()
        return False
    
    model.zero_grad()
    
    print("\nTesting ensemble forward + backward...")
    try:
        ensemble_preds = []
        for i in range(3):
            print(f"  Ensemble member {i+1}/3...")
            out = model(x, dt)
            if out.is_complex():
                ensemble_preds.append(out.real)
            else:
                ensemble_preds.append(out)
        
        ensemble_stack = torch.stack(ensemble_preds, dim=0)
        ensemble_mean = ensemble_stack.mean(dim=0)
        loss = (ensemble_mean - target).abs().mean()
        loss.backward()
        print("Ensemble forward: PASSED")
    except Exception as e:
        print(f"Ensemble forward: FAILED - {e}")
        traceback.print_exc()
        return False
    
    return True


def test_medium_model():
    print("=" * 60)
    print("Test: Medium Model (embed_dim=256)")
    print("=" * 60)
    
    device = torch.device("cuda")
    
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=256,
        expand=4,
        num_experts=4,
        depth=4,
        patch_size=32,
        img_height=128,
        img_width=256,
        dt_ref=6.0,
        sde_mode="sde",
        init_noise_scale=1.0,
        max_growth_rate=0.3,
    ).to(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 128, 256
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 6.0
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
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED - {e}")
        traceback.print_exc()
        return False


def test_with_anomaly():
    print("=" * 60)
    print("Test: Anomaly Detection on Medium Model")
    print("=" * 60)
    
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device("cuda")
    
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=512,
        expand=4,
        num_experts=4,
        depth=4,
        patch_size=16,
        img_height=64,
        img_width=128,
        dt_ref=6.0,
        sde_mode="sde",
        init_noise_scale=1.0,
        max_growth_rate=0.3,
    ).to(device)
    model.train()
    
    B, T, C, H, W = 2, 4, 4, 64, 128
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 6.0
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
        print("PASSED")
        torch.autograd.set_detect_anomaly(False)
        return True
    except Exception as e:
        print(f"FAILED - {e}")
        traceback.print_exc()
        torch.autograd.set_detect_anomaly(False)
        return False


def main():
    print("\n" + "=" * 60)
    print("UniPhy Large Model Debug")
    print("=" * 60 + "\n")
    
    results = {}
    
    results["medium_model"] = test_medium_model()
    print()
    
    torch.cuda.empty_cache()
    
    results["anomaly_512dim"] = test_with_anomaly()
    print()
    
    torch.cuda.empty_cache()
    
    try:
        results["large_model"] = test_large_model()
    except torch.cuda.OutOfMemoryError:
        print("SKIPPED - Out of Memory")
        results["large_model"] = None
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED (OOM)"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
    