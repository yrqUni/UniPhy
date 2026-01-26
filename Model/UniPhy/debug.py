import torch
import torch.nn as nn
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from PScan import pscan


def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def compare_tensors(name, t1, t2, atol=1e-5):
    if t1.is_complex():
        diff = (t1 - t2).abs().max().item()
    else:
        diff = (t1 - t2).abs().max().item()
    
    status = "✓ PASS" if diff < atol else "✗ FAIL"
    print(f"{name}: max_diff = {diff:.6e} {status}")
    
    if diff >= atol:
        print(f"  t1 mean: {t1.real.mean().item():.6f}, t2 mean: {t2.real.mean().item():.6f}")
        print(f"  t1 std: {t1.real.std().item():.6f}, t2 std: {t2.real.std().item():.6f}")
    
    return diff < atol


def debug_basis_transform():
    print_separator("Debug: Basis Transform (encode/decode)")
    
    torch.manual_seed(42)
    dim = 32
    B, T, H, W = 2, 4, 8, 8
    
    basis = ComplexSVDTransform(dim).cuda()
    
    x_seq = torch.randn(B, T, H, W, dim, dtype=torch.complex64, device="cuda")
    x_single_list = [x_seq[:, t] for t in range(T)]
    
    h_seq = basis.encode(x_seq)
    h_single_list = [basis.encode(x_single_list[t]) for t in range(T)]
    h_single_stacked = torch.stack(h_single_list, dim=1)
    
    compare_tensors("Basis encode (seq vs single)", h_seq, h_single_stacked)
    
    x_recon_seq = basis.decode(h_seq)
    x_recon_single_list = [basis.decode(h_single_list[t]) for t in range(T)]
    x_recon_single_stacked = torch.stack(x_recon_single_list, dim=1)
    
    compare_tensors("Basis decode (seq vs single)", x_recon_seq, x_recon_single_stacked)
    compare_tensors("Basis invertibility", x_seq, x_recon_seq, atol=1e-4)


def debug_flux_tracker():
    print_separator("Debug: FluxTracker (parallel vs serial)")
    
    torch.manual_seed(42)
    dim = 32
    B, T = 2, 4
    
    tracker = GlobalFluxTracker(dim).cuda()
    
    x_mean_seq = torch.randn(B, T, dim, dtype=torch.complex64, device="cuda")
    
    A, X = tracker.get_operators(x_mean_seq)
    print(f"A shape: {A.shape}, X shape: {X.shape}")
    
    flux_parallel = pscan(A.unsqueeze(-1), X.unsqueeze(-1)).squeeze(-1)
    source_parallel = tracker.project(flux_parallel)
    
    flux_serial_list = []
    source_serial_list = []
    gate_serial_list = []
    
    flux_prev = torch.zeros(B, dim, dtype=torch.complex64, device="cuda")
    
    for t in range(T):
        x_t = x_mean_seq[:, t]
        flux_next, source, gate = tracker.forward_step(flux_prev, x_t)
        flux_serial_list.append(flux_next)
        source_serial_list.append(source)
        gate_serial_list.append(gate)
        flux_prev = flux_next
    
    flux_serial = torch.stack(flux_serial_list, dim=1)
    source_serial = torch.stack(source_serial_list, dim=1)
    
    compare_tensors("Flux state (parallel vs serial)", flux_parallel, flux_serial)
    compare_tensors("Source (parallel vs serial)", source_parallel, source_serial)
    
    print("\nFlux details:")
    for t in range(T):
        diff = (flux_parallel[:, t] - flux_serial[:, t]).abs().max().item()
        print(f"  t={t}: flux diff = {diff:.6e}")


def debug_pscan_vs_loop():
    print_separator("Debug: PScan vs Sequential Loop")
    
    torch.manual_seed(42)
    B, T, D = 4, 8, 32
    
    A = torch.randn(B, T, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, T, D, 1, dtype=torch.complex64, device="cuda")
    
    Y_pscan = pscan(A, X)
    
    Y_loop = torch.zeros_like(X)
    acc = torch.zeros(B, D, 1, dtype=torch.complex64, device="cuda")
    
    for t in range(T):
        acc = acc * A[:, t].unsqueeze(-1) + X[:, t]
        Y_loop[:, t] = acc
    
    compare_tensors("PScan vs Loop", Y_pscan, Y_loop)
    
    print("\nPer-timestep comparison:")
    for t in range(T):
        diff = (Y_pscan[:, t] - Y_loop[:, t]).abs().max().item()
        print(f"  t={t}: diff = {diff:.6e}")


def debug_transition_operators():
    print_separator("Debug: Transition Operators")
    
    torch.manual_seed(42)
    dim = 32
    
    prop = TemporalPropagator(dim, dt_ref=1.0, sde_mode="ode").cuda()
    
    dt_scalar = torch.tensor(1.0, device="cuda")
    dt_1d = torch.tensor([1.0, 1.0, 1.0, 1.0], device="cuda")
    dt_2d = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], device="cuda")
    
    decay_s, forcing_s = prop.get_transition_operators(dt_scalar)
    decay_1d, forcing_1d = prop.get_transition_operators(dt_1d)
    decay_2d, forcing_2d = prop.get_transition_operators(dt_2d)
    
    print(f"Scalar dt -> decay shape: {decay_s.shape}, forcing shape: {forcing_s.shape}")
    print(f"1D dt -> decay shape: {decay_1d.shape}, forcing shape: {forcing_1d.shape}")
    print(f"2D dt -> decay shape: {decay_2d.shape}, forcing shape: {forcing_2d.shape}")
    
    compare_tensors("decay (scalar vs 1d[0])", decay_s, decay_1d[0])


def debug_block_forward_vs_step():
    print_separator("Debug: UniPhyBlock forward() vs forward_step()")
    
    torch.manual_seed(42)
    
    B, T, D, H, W = 1, 4, 32, 8, 8
    
    block = UniPhyBlock(
        dim=D,
        expand=2,
        num_experts=4,
        img_height=H,
        img_width=W,
        dt_ref=1.0,
        sde_mode="ode",
        init_noise_scale=0.0,
        max_growth_rate=0.3,
    ).cuda()
    block.eval()
    
    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")
    
    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")
    
    with torch.no_grad():
        z_parallel, h_parallel, flux_parallel = block(x, h_init, dt, flux_init)
    
    z_serial_list = []
    h_prev = h_init.clone()
    flux_prev = flux_init.clone()
    
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t]
            dt_t = dt[t]
            
            z_t, h_next, flux_next = block.forward_step(x_t, h_prev, dt_t, flux_prev)
            z_serial_list.append(z_t)
            h_prev = h_next
            flux_prev = flux_next
    
    z_serial = torch.stack(z_serial_list, dim=1)
    
    print(f"Parallel output shape: {z_parallel.shape}")
    print(f"Serial output shape: {z_serial.shape}")
    
    compare_tensors("Block output (parallel vs serial)", z_parallel, z_serial, atol=1e-4)
    
    print("\nPer-timestep comparison:")
    for t in range(T):
        diff = (z_parallel[:, t] - z_serial[:, t]).abs().max().item()
        status = "✓" if diff < 1e-4 else "✗"
        print(f"  t={t}: diff = {diff:.6e} {status}")
    
    print("\nDiagnostics:")
    print(f"  Parallel mean: {z_parallel.real.mean().item():.6f}")
    print(f"  Serial mean: {z_serial.real.mean().item():.6f}")
    print(f"  Parallel std: {z_parallel.real.std().item():.6f}")
    print(f"  Serial std: {z_serial.real.std().item():.6f}")


def debug_spatial_processing():
    print_separator("Debug: Spatial Processing Consistency")
    
    torch.manual_seed(42)
    
    B, T, D, H, W = 1, 4, 32, 8, 8
    
    block = UniPhyBlock(
        dim=D,
        expand=2,
        num_experts=4,
        img_height=H,
        img_width=W,
        dt_ref=1.0,
        sde_mode="ode",
    ).cuda()
    block.eval()
    
    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    
    with torch.no_grad():
        x_5d_processed = block._spatial_process(x)
    
    x_4d_processed_list = []
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t]
            x_t_processed = block._spatial_process(x_t)
            x_4d_processed_list.append(x_t_processed)
    
    x_4d_stacked = torch.stack(x_4d_processed_list, dim=1)
    
    compare_tensors("Spatial process (5D vs 4D stacked)", x_5d_processed, x_4d_stacked)


def debug_temporal_decode():
    print_separator("Debug: Temporal Decode Consistency")
    
    torch.manual_seed(42)
    
    B, T, D, H, W = 1, 4, 32, 8, 8
    
    block = UniPhyBlock(
        dim=D,
        expand=2,
        num_experts=4,
        img_height=H,
        img_width=W,
        dt_ref=1.0,
        sde_mode="ode",
    ).cuda()
    block.eval()
    
    x = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    
    with torch.no_grad():
        x_5d_decoded = block._temporal_decode(x)
    
    x_4d_decoded_list = []
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t]
            x_t_decoded = block._temporal_decode(x_t)
            x_4d_decoded_list.append(x_t_decoded)
    
    x_4d_stacked = torch.stack(x_4d_decoded_list, dim=1)
    
    compare_tensors("Temporal decode (5D vs 4D stacked)", x_5d_decoded, x_4d_stacked)


def debug_full_model():
    print_separator("Debug: Full Model Parallel vs Serial")
    
    torch.manual_seed(42)
    
    B, T, C, H, W = 1, 4, 2, 33, 33
    
    model = UniPhyModel(
        in_channels=C,
        out_channels=C,
        embed_dim=32,
        expand=2,
        num_experts=4,
        depth=2,
        patch_size=8,
        img_height=H,
        img_width=W,
        dt_ref=1.0,
        sde_mode="ode",
        init_noise_scale=0.0,
    ).cuda()
    model.eval()
    
    x = torch.randn(B, T, C, H, W, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")
    
    with torch.no_grad():
        out_parallel = model(x, dt)
    
    with torch.no_grad():
        z = model.encoder(x)
        
        base_dtype = z.dtype if z.dtype.is_complex else torch.complex64
        
        h_prev = None
        flux_prev = None
        
        for i, block in enumerate(model.blocks):
            if i == 0:
                h_init = torch.zeros(
                    B * model.h_patches * model.w_patches, 1, model.embed_dim,
                    device="cuda", dtype=base_dtype
                )
                flux_init = torch.zeros(B, model.embed_dim, device="cuda", dtype=base_dtype)
                z, h_prev, flux_prev = block(z, h_init, dt, flux_init)
            else:
                z, h_prev, flux_prev = block(z, h_prev, dt, flux_prev)
        
        out_manual_parallel = model.decoder(z)
    
    compare_tensors("Full model (forward vs manual parallel)", out_parallel, out_manual_parallel)
    
    with torch.no_grad():
        x_init = x[:, 0]
        dt_list = [dt[t].item() for t in range(T)]
        out_serial = model.forward_rollout(x_init, dt_list, num_steps=T)
    
    print(f"\nParallel output shape: {out_parallel.shape}")
    print(f"Serial output shape: {out_serial.shape}")
    
    compare_tensors("Full model (parallel vs serial)", out_parallel, out_serial, atol=1e-3)
    
    print("\nPer-timestep comparison:")
    for t in range(T):
        diff = (out_parallel[:, t] - out_serial[:, t]).abs().max().item()
        status = "✓" if diff < 1e-3 else "✗"
        print(f"  t={t}: diff = {diff:.6e} {status}")


def main():
    print("\n" + "=" * 70)
    print("  UniPhy Model Debug Suite")
    print("=" * 70)
    
    debug_basis_transform()
    debug_pscan_vs_loop()
    debug_transition_operators()
    debug_flux_tracker()
    debug_spatial_processing()
    debug_temporal_decode()
    debug_block_forward_vs_step()
    debug_full_model()
    
    print("\n" + "=" * 70)
    print("  Debug Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()