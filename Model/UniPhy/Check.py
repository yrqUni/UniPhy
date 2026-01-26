import torch
import torch.nn as nn
from ModelUniPhy import UniPhyModel

def sequential_forward(model, x, dt):
    B, T, C, H, W = x.shape
    device = x.device
    
    outputs = []
    states = model._init_states(B, device, torch.complex64)
    
    for t in range(T):
        x_t = x[:, t:t+1]
        z = model.encoder(x_t)
        
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z, h_next, flux_next = block(z, h_prev, dt[:, t:t+1] if dt.ndim > 1 else dt, flux_prev)
            states[i] = (h_next, flux_next)
        
        out = model.decoder(z)
        if out.shape[-2] != H or out.shape[-1] != W:
            out = out[..., :H, :W]
        outputs.append(out.squeeze(1))
    
    return torch.stack(outputs, dim=1)

def check_forward_consistency():
    print("=" * 60)
    print("Testing Forward Consistency (Serial vs Parallel)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
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
    
    B, T, C, H, W = 2, 8, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = sequential_forward(model, x, dt)
    
    max_diff = (out_parallel - out_serial).abs().max().item()
    mean_diff = (out_parallel - out_serial).abs().mean().item()
    
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed

def check_long_sequence():
    print("=" * 60)
    print("Testing Long Sequence Consistency")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=32,
        depth=1,
        patch_size=4,
        img_height=32,
        img_width=32,
    ).to(device)
    model.eval()
    
    B, T, C, H, W = 1, 32, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.05
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = sequential_forward(model, x, dt)
    
    max_diff = (out_parallel - out_serial).abs().max().item()
    mean_diff = (out_parallel - out_serial).abs().mean().item()
    
    print(f"Sequence Length: {T}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed

def check_variable_dt():
    print("=" * 60)
    print("Testing Variable dt Consistency")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(456)
    
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
    
    B, T, C, H, W = 2, 16, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device) * 0.2 + 0.05
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = sequential_forward(model, x, dt)
    
    max_diff = (out_parallel - out_serial).abs().max().item()
    mean_diff = (out_parallel - out_serial).abs().mean().item()
    
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed

def check_batch_consistency():
    print("=" * 60)
    print("Testing Batch Consistency")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(789)
    
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
    
    B, T, C, H, W = 8, 4, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = sequential_forward(model, x, dt)
    
    max_diff = (out_parallel - out_serial).abs().max().item()
    mean_diff = (out_parallel - out_serial).abs().mean().item()
    
    print(f"Batch Size: {B}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-3
    print(f"Test Passed: {passed}")
    print()
    return passed

def run_all_checks():
    results = {}
    
    results["forward_consistency"] = check_forward_consistency()
    results["long_sequence"] = check_long_sequence()
    results["variable_dt"] = check_variable_dt()
    results["batch_consistency"] = check_batch_consistency()
    
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

