import torch
import torch.nn as nn
from ModelUniPhy import UniPhyModel

def sequential_forward(model, x, dt):
    B, T, C, H, W = x.shape
    device = x.device
    outputs = []
    
    z_all = model.encoder(x)
    dtype = z_all.dtype if z_all.dtype.is_complex else torch.complex64
    states = model._init_states(B, device, dtype)
    
    for t in range(T):
        z_t = z_all[:, t]
        dt_t = dt[:, t] if dt.ndim > 1 else dt[t] if dt.ndim == 1 else dt
        
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt_t, flux_prev)
            states[i] = (h_next, flux_next)
        
        out = model.decoder(z_t.unsqueeze(1)).squeeze(1)
        if out.shape[-2] != H or out.shape[-1] != W:
            out = out[..., :H, :W]
        outputs.append(out)
    
    return torch.stack(outputs, dim=1)

def manual_rollout_reference(model, x_context, dt_context, dt_list):
    B, T_in, C, H, W = x_context.shape
    device = x_context.device
    
    z_ctx = model.encoder(x_context)
    dtype = z_ctx.dtype if z_ctx.dtype.is_complex else torch.complex64
    states = model._init_states(B, device, dtype)
    
    if isinstance(dt_context, float):
        dt_ctx_tensor = torch.full((B, T_in), dt_context, device=device)
    elif dt_context.ndim == 0:
        dt_ctx_tensor = dt_context.expand(B, T_in)
    else:
        dt_ctx_tensor = dt_context

    for i, block in enumerate(model.blocks):
        h_prev, flux_prev = states[i]
        z_ctx, h_final, flux_final = block(z_ctx, h_prev, dt_ctx_tensor, flux_prev)
        states[i] = (h_final, flux_final)

    z_curr = z_ctx[:, -1]
    preds = []
    
    for dt_k in dt_list:
        new_states = []
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_curr, h_next, flux_next = block.forward_step(z_curr, h_prev, dt_k, flux_prev)
            new_states.append((h_next, flux_next))
        
        states = new_states
        
        pred = model.decoder(z_curr.unsqueeze(1)).squeeze(1)
        if pred.shape[-2] != model.img_height or pred.shape[-1] != model.img_width:
            pred = pred[..., :model.img_height, :model.img_width]
        preds.append(pred)
        
    return torch.stack(preds, dim=1)

def check_forward_consistency():
    print("=" * 60)
    print("Testing Forward Consistency (Serial Step vs Parallel Scan)")
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
        sde_mode="det",
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

def check_rollout_consistency():
    print("=" * 60)
    print("Testing Rollout Consistency (Manual vs Built-in)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(999)
    
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depth=2,
        patch_size=4,
        img_height=32,
        img_width=32,
        sde_mode="det",
    ).to(device)
    model.eval()
    
    B, T_in, T_pred, C, H, W = 2, 8, 4, 4, 32, 32
    x_context = torch.randn(B, T_in, C, H, W, device=device)
    dt_context = 1.0
    dt_list = [torch.tensor(0.5, device=device) for _ in range(T_pred)]
    
    with torch.no_grad():
        out_builtin = model.forward_rollout(x_context, dt_context, dt_list)
        out_manual = manual_rollout_reference(model, x_context, dt_context, dt_list)
    
    max_diff = (out_builtin - out_manual).abs().max().item()
    mean_diff = (out_builtin - out_manual).abs().mean().item()
    
    print(f"Context Len: {T_in}, Pred Len: {T_pred}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-4
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
        sde_mode="det",
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
        sde_mode="det",
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
        sde_mode="det",
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
    results["rollout_consistency"] = check_rollout_consistency()
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
    
    return all(results.values())

if __name__ == "__main__":
    success = run_all_checks()
    exit(0 if success else 1)
    