import torch
import torch.nn as nn
from ModelUniPhy import UniPhyModel


def full_serial_inference(model, x_context, dt_context, dt_list):
    B, T_in = x_context.shape[0], x_context.shape[1]
    device = x_context.device

    z_all = model.encoder(x_context)
    dtype = z_all.dtype if z_all.dtype.is_complex else torch.complex64
    states = model._init_states(B, device, dtype)

    if isinstance(dt_context, (float, int)):
        dt_ctx_val = float(dt_context)
    else:
        dt_ctx_val = dt_context[0].item() if dt_context.ndim > 0 else dt_context.item()

    z_curr = None

    for t in range(T_in):
        z_in = z_all[:, t]
        dt_t = torch.tensor(dt_ctx_val, device=device).float()

        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_in, h_next, flux_next = block.forward_step(
                z_in, h_prev, dt_t, flux_prev
            )
            states[i] = (h_next, flux_next)
        
        z_curr = z_in

    preds = []
    for dt_k in dt_list:
        new_states = []
        for i, block in enumerate(model.blocks):
            h_prev, flux_prev = states[i]
            z_curr, h_next, flux_next = block.forward_step(
                z_curr, h_prev, dt_k, flux_prev
            )
            new_states.append((h_next, flux_next))
        
        states = new_states
        
        pred = model.decoder(z_curr.unsqueeze(1)).squeeze(1)
        if pred.shape[-2] != model.img_height or pred.shape[-1] != model.img_width:
            pred = pred[..., :model.img_height, :model.img_width]
        preds.append(pred)

    return torch.stack(preds, dim=1)


def check_forward_consistency():
    print("=" * 60)
    print("Test 1: Parallel Scan vs Serial Loop (Pure Sequence)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    model = UniPhyModel(
        in_channels=4, out_channels=4, embed_dim=64, depth=2, 
        patch_size=4, img_height=32, img_width=32, sde_mode="det"
    ).to(device)
    model.eval()
    
    B, T, C, H, W = 2, 8, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.ones(B, T, device=device) * 0.1
    
    with torch.no_grad():
        out_parallel = model(x, dt)
    
    out_serial = []
    z_all = model.encoder(x)
    states = model._init_states(B, device, z_all.dtype)
    
    for t in range(T):
        z_t = z_all[:, t]
        dt_t = dt[:, t]
        for i, block in enumerate(model.blocks):
            h, f = states[i]
            z_t, h_n, f_n = block.forward_step(z_t, h, dt_t, f)
            states[i] = (h_n, f_n)
        out = model.decoder(z_t.unsqueeze(1)).squeeze(1)
        out_serial.append(out[..., :32, :32])
    
    out_serial = torch.stack(out_serial, dim=1)
    
    max_diff = (out_parallel - out_serial).abs().max().item()
    print(f"Max Difference: {max_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"Result: {'PASSED' if passed else 'FAILED'}\n")
    return passed


def check_context_equivalence():
    print("=" * 60)
    print("Test 2: Optimized Rollout vs Full Serial Inference")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1024)
    
    model = UniPhyModel(
        in_channels=4, out_channels=4, embed_dim=64, depth=2, 
        patch_size=4, img_height=32, img_width=32, sde_mode="det"
    ).to(device)
    model.eval()
    
    B, T_in, T_pred, C, H, W = 2, 8, 4, 4, 32, 32
    x_context = torch.randn(B, T_in, C, H, W, device=device)
    dt_context = 1.0
    dt_list = [torch.tensor(0.5, device=device) for _ in range(T_pred)]
    
    with torch.no_grad():
        out_optimized = model.forward_rollout(x_context, dt_context, dt_list)
        out_full_serial = full_serial_inference(
            model, x_context, dt_context, dt_list
        )
    
    max_diff = (out_optimized - out_full_serial).abs().max().item()
    mean_diff = (out_optimized - out_full_serial).abs().mean().item()
    
    print(f"Context Len: {T_in}, Pred Len: {T_pred}")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    passed = max_diff < 1e-4
    print(f"Result: {'PASSED' if passed else 'FAILED'}\n")
    return passed


def run_all_checks():
    r1 = check_forward_consistency()
    r2 = check_context_equivalence()
    
    print("=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Forward Consistency (Math):   {'PASSED' if r1 else 'FAILED'}")
    print(f"Rollout Logic (Full Serial):  {'PASSED' if r2 else 'FAILED'}")
    
    return r1 and r2


if __name__ == "__main__":
    success = run_all_checks()
    exit(0 if success else 1)
    