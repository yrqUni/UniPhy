import torch
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from PScan import pscan


def print_sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def compare(name, t1, t2, atol=1e-5):
    diff = (t1 - t2).abs().max().item()
    status = "PASS" if diff < atol else "FAIL"
    print(f"{name}: diff={diff:.2e} [{status}]")
    if diff >= atol:
        print(f"  t1: mean={t1.real.mean():.4f}, std={t1.real.std():.4f}")
        print(f"  t2: mean={t2.real.mean():.4f}, std={t2.real.std():.4f}")
    return diff < atol


def debug_basis():
    print_sep("Basis Transform")
    torch.manual_seed(42)
    
    basis = ComplexSVDTransform(32).cuda()
    x = torch.randn(2, 4, 8, 8, 32, dtype=torch.complex64, device="cuda")
    
    h = basis.encode(x)
    x_rec = basis.decode(h)
    
    compare("Invertibility", x, x_rec, atol=1e-4)


def debug_pscan():
    print_sep("PScan")
    torch.manual_seed(42)
    B, L, C, D = 2, 8, 4, 32
    
    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")
    
    Y_pscan = pscan(A, X)
    
    Y_loop = torch.zeros_like(X)
    acc = torch.zeros(B, C, D, 1, dtype=torch.complex64, device="cuda")
    for t in range(L):
        acc = acc * A[:, t].unsqueeze(-1) + X[:, t]
        Y_loop[:, t] = acc
    
    passed = compare("PScan vs Loop", Y_pscan, Y_loop)
    
    if not passed:
        print("\n  Per-timestep analysis:")
        for t in range(min(3, L)):
            diff_t = (Y_pscan[:, t] - Y_loop[:, t]).abs().max().item()
            print(f"    t={t}: diff={diff_t:.2e}")


def debug_flux_tracker():
    print_sep("FluxTracker")
    torch.manual_seed(42)
    
    tracker = GlobalFluxTracker(32).cuda()
    x_mean = torch.randn(2, 4, 32, dtype=torch.complex64, device="cuda")
    
    flux_list, gate_list = [], []
    flux = torch.zeros(2, 32, dtype=torch.complex64, device="cuda")
    
    for t in range(4):
        flux, _, gate = tracker.forward_step(flux, x_mean[:, t])
        flux_list.append(flux)
        gate_list.append(gate)
    
    print(f"Flux shape: {flux.shape}")
    print(f"Gate range: [{gate_list[-1].min():.4f}, {gate_list[-1].max():.4f}]")


def debug_block():
    print_sep("UniPhyBlock: forward vs forward_step")
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
    
    with torch.no_grad():
        z_par, _, _ = block(x, h_init, dt, flux_init)
    
    z_ser_list = []
    h, flux = h_init.clone(), flux_init.clone()
    with torch.no_grad():
        for t in range(T):
            z_t, h, flux = block.forward_step(x[:, t], h, dt[t], flux)
            z_ser_list.append(z_t)
    z_ser = torch.stack(z_ser_list, dim=1)
    
    passed = compare("Block output", z_par, z_ser, atol=1e-3)
    
    if not passed:
        print("\n  Per-timestep analysis:")
        for t in range(T):
            diff = (z_par[:, t] - z_ser[:, t]).abs().max().item()
            print(f"    t={t}: diff={diff:.2e}")


def debug_model():
    print_sep("Full Model")
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
        out_ser = model.forward_rollout(x[:, 0], [1.0] * 4)
    
    compare("Model output", out_par, out_ser, atol=1e-2)


def main():
    print("=" * 60)
    print("  Debug Suite")
    print("=" * 60)
    
    debug_basis()
    debug_pscan()
    debug_flux_tracker()
    debug_block()
    debug_model()
    
    print("\n" + "=" * 60)
    print("  Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
    