import torch
import torch.nn as nn
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from UniPhyOps import RiemannianCliffordConv2d
from UniPhyFFN import UniPhyFeedForwardNetwork
from PScan import pscan


def sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def cmp(name, t1, t2, atol=1e-5):
    diff = (t1 - t2).abs().max().item()
    ok = diff < atol
    print(f"{name}: diff={diff:.2e} [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        print(f"  t1: mean={t1.real.mean().item():.4f} std={t1.real.std().item():.4f}")
        print(f"  t2: mean={t2.real.mean().item():.4f} std={t2.real.std().item():.4f}")
    return ok


def debug_basis():
    sep("Basis Transform")
    torch.manual_seed(42)
    
    basis = ComplexSVDTransform(32).cuda()
    x = torch.randn(2, 4, 8, 8, 32, dtype=torch.complex64, device="cuda")
    
    h = basis.encode(x)
    x_rec = basis.decode(h)
    
    cmp("Invertibility", x, x_rec, atol=1e-4)


def debug_pscan():
    sep("PScan")
    torch.manual_seed(42)
    B, L, C, D = 2, 8, 4, 2
    
    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")
    
    Y_pscan = pscan(A, X)
    
    Y_loop = torch.zeros_like(X)
    acc = torch.zeros(B, C, D, 1, dtype=torch.complex64, device="cuda")
    for t in range(L):
        acc = acc * A[:, t].unsqueeze(-1) + X[:, t]
        Y_loop[:, t] = acc
    
    passed = cmp("PScan vs Loop", Y_pscan, Y_loop)
    
    if not passed:
        print("\n  Per-timestep:")
        for t in range(min(4, L)):
            d = (Y_pscan[:, t] - Y_loop[:, t]).abs().max().item()
            print(f"    t={t}: {d:.2e}")


def debug_flux_tracker():
    sep("FluxTracker")
    torch.manual_seed(42)
    
    tracker = GlobalFluxTracker(32).cuda()
    
    flux = torch.zeros(2, 32, dtype=torch.complex64, device="cuda")
    x_seq = torch.randn(2, 4, 32, dtype=torch.complex64, device="cuda")
    
    flux_list = []
    gate_list = []
    for t in range(4):
        flux, _, gate = tracker.forward_step(flux, x_seq[:, t])
        flux_list.append(flux.clone())
        gate_list.append(gate.clone())
    
    print(f"Flux shape: {flux.shape}")
    print(f"Gate shape: {gate.shape}")
    print(f"Gate range: [{gate_list[-1].min():.4f}, {gate_list[-1].max():.4f}]")
    print(f"Gate is real: {not gate_list[-1].is_complex()}")


def debug_block_interface():
    sep("Block Interface Check")
    
    block = UniPhyBlock(
        dim=32, expand=2, num_experts=4,
        img_height=8, img_width=8,
        dt_ref=1.0, sde_mode="ode"
    ).cuda()
    
    import inspect
    
    forward_sig = inspect.signature(block.forward)
    step_sig = inspect.signature(block.forward_step)
    
    print(f"forward() params: {list(forward_sig.parameters.keys())}")
    print(f"forward_step() params: {list(step_sig.parameters.keys())}")
    
    forward_params = len(forward_sig.parameters)
    step_params = len(step_sig.parameters)
    
    print(f"\nforward() has {forward_params} params")
    print(f"forward_step() has {step_params} params")
    
    if forward_params != step_params:
        print("\n!!! MISMATCH: forward() and forward_step() have different signatures !!!")
        print("This is likely the root cause of parallel/serial inconsistency")


def debug_block_detailed():
    sep("Block Detailed Analysis")
    torch.manual_seed(42)
    
    B, T, D, H, W = 1, 4, 32, 8, 8
    
    block = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()
    
    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")
    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")
    
    print(f"Input x shape: {x.shape}")
    print(f"dt shape: {dt.shape}")
    print(f"h_init shape: {h_init.shape}")
    print(f"flux_init shape: {flux_init.shape}")
    
    try:
        with torch.no_grad():
            result = block(x, h_init, dt, flux_init)
        
        if isinstance(result, tuple):
            print(f"\nforward() returns tuple of {len(result)} elements")
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
                else:
                    print(f"  [{i}]: {type(r)}")
            z_par = result[0]
        else:
            print(f"\nforward() returns single tensor: {result.shape}")
            z_par = result
            
    except TypeError as e:
        print(f"\nforward() call failed: {e}")
        print("Trying with 2 arguments (z, dt)...")
        
        with torch.no_grad():
            result = block(x, dt)
        
        if isinstance(result, tuple):
            z_par = result[0]
        else:
            z_par = result
        print(f"Result shape: {z_par.shape}")
    
    print("\n--- Serial Mode ---")
    z_ser_list = []
    h, flux = h_init.clone(), flux_init.clone()
    
    with torch.no_grad():
        for t in range(T):
            z_t, h, flux = block.forward_step(x[:, t], h, dt[t], flux)
            z_ser_list.append(z_t)
            print(f"  t={t}: z_t shape={z_t.shape}")
    
    z_ser = torch.stack(z_ser_list, dim=1)
    print(f"\nSerial output shape: {z_ser.shape}")
    
    if z_par.shape == z_ser.shape:
        cmp("Block parallel vs serial", z_par, z_ser, atol=1e-3)
    else:
        print(f"Shape mismatch: parallel={z_par.shape}, serial={z_ser.shape}")


def debug_model_forward():
    sep("Model Forward Analysis")
    torch.manual_seed(42)
    
    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33, sde_mode="ode"
    ).cuda().eval()
    
    x = torch.randn(1, 4, 2, 33, 33, device="cuda") * 0.1
    dt = torch.ones(4, device="cuda")
    
    print(f"Input shape: {x.shape}")
    print(f"dt shape: {dt.shape}")
    print(f"h_patches: {model.h_patches}, w_patches: {model.w_patches}")
    print(f"embed_dim: {model.embed_dim}")
    print(f"depth: {model.depth}")
    
    with torch.no_grad():
        z = model.encoder(x)
        print(f"\nEncoder output shape: {z.shape}")
        print(f"Encoder output dtype: {z.dtype}")
        
        for i, block in enumerate(model.blocks):
            print(f"\n--- Block {i} ---")
            
            import inspect
            sig = inspect.signature(block.forward)
            params = list(sig.parameters.keys())
            print(f"forward() params: {params}")
            
            if len(params) == 2:
                z = block(z, dt)
            elif len(params) == 4:
                B = z.shape[0]
                h_init = torch.zeros(
                    B * model.h_patches * model.w_patches, 1, model.embed_dim,
                    device="cuda", dtype=z.dtype
                )
                flux_init = torch.zeros(B, model.embed_dim, device="cuda", dtype=z.dtype)
                z, _, _ = block(z, h_init, dt, flux_init)
            
            if isinstance(z, tuple):
                z = z[0]
            print(f"Block {i} output shape: {z.shape}")
        
        out = model.decoder(z)
        print(f"\nDecoder output shape: {out.shape}")


def debug_model_comparison():
    sep("Model Parallel vs Serial")
    torch.manual_seed(42)
    
    model = UniPhyModel(
        in_channels=2, out_channels=2, embed_dim=32,
        expand=2, num_experts=4, depth=2, patch_size=8,
        img_height=33, img_width=33, sde_mode="ode"
    ).cuda().eval()
    
    x = torch.randn(1, 4, 2, 33, 33, device="cuda") * 0.1
    dt = torch.ones(4, device="cuda")
    
    with torch.no_grad():
        out_par = model(x, dt)
    
    print(f"Parallel output shape: {out_par.shape}")
    print(f"Parallel mean: {out_par.mean().item():.4f}")
    print(f"Parallel std: {out_par.std().item():.4f}")
    
    with torch.no_grad():
        out_ser = model.forward_rollout(x[:, 0], [1.0] * 4)
    
    print(f"\nSerial output shape: {out_ser.shape}")
    print(f"Serial mean: {out_ser.mean().item():.4f}")
    print(f"Serial std: {out_ser.std().item():.4f}")
    
    if out_par.shape == out_ser.shape:
        cmp("Model parallel vs serial", out_par, out_ser, atol=1e-2)
        
        print("\nPer-timestep analysis:")
        for t in range(out_par.shape[1]):
            d = (out_par[:, t] - out_ser[:, t]).abs().max().item()
            print(f"  t={t}: diff={d:.2e}")
    else:
        print(f"\nShape mismatch!")


def main():
    print("=" * 60)
    print("  UniPhy Debug Suite")
    print("=" * 60)
    
    debug_basis()
    debug_pscan()
    debug_flux_tracker()
    debug_block_interface()
    debug_block_detailed()
    debug_model_forward()
    debug_model_comparison()
    
    print("\n" + "=" * 60)
    print("  Debug Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
    