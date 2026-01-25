import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from UniPhyOps import TemporalPropagator, ComplexSVDTransform, GlobalFluxTracker
from PScan import pscan


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    if tensor.is_complex():
        print(f"{name}: shape={tuple(tensor.shape)}, "
              f"abs_max={tensor.abs().max().item():.6e}, "
              f"abs_mean={tensor.abs().mean().item():.6e}")
    else:
        print(f"{name}: shape={tuple(tensor.shape)}, "
              f"max={tensor.max().item():.6e}, "
              f"min={tensor.min().item():.6e}")


def check_pscan_recurrence():
    print_section("Step 1: Verify PScan Recurrence Formula")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    B, T, C, D = 2, 5, 4, 2

    A = torch.randn(B, T, C, D, dtype=torch.cdouble, device=device) * 0.5
    X = torch.randn(B, T, C, D, dtype=torch.cdouble, device=device)

    X_5d = X.unsqueeze(-1)
    Y_pscan = pscan(A, X_5d).squeeze(-1)

    Y_serial = torch.zeros_like(X)
    Y_serial[:, 0] = X[:, 0]
    for t in range(1, T):
        Y_serial[:, t] = A[:, t] * Y_serial[:, t - 1] + X[:, t]

    print("PScan recurrence: Y[0] = X[0], Y[t] = A[t] * Y[t-1] + X[t]")
    print()

    for t in range(T):
        diff = (Y_pscan[:, t] - Y_serial[:, t]).abs().max().item()
        print(f"  t={t}: Diff={diff:.2e}")

    total_diff = (Y_pscan - Y_serial).abs().max().item()
    print(f"\nTotal Max Diff: {total_diff:.2e}")
    print(f"PScan Formula Verified: {total_diff < 1e-6}")

    return total_diff < 1e-6


def check_transition_operators():
    print_section("Step 2: Check Transition Operators")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    D = 4
    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()

    dt_values = [0.1, 0.5, 1.0, 2.0]

    print("Checking op_decay and op_forcing for different dt values:")
    print()

    for dt_val in dt_values:
        dt = torch.tensor([[dt_val]], device=device, dtype=torch.float64)
        op_decay, op_forcing = prop.get_transition_operators(dt)

        print(f"dt={dt_val}:")
        print_tensor_stats("  op_decay", op_decay)
        print_tensor_stats("  op_forcing", op_forcing)
        print()


def check_parallel_vs_serial_simplified():
    print_section("Step 3: Simplified Parallel vs Serial PScan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    B, T, H, W, D = 1, 5, 4, 4, 4

    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()

    forcing = torch.randn(B, T, H, W, D, dtype=torch.cdouble, device=device)
    dt = torch.ones(T, device=device, dtype=torch.float64)

    print("--- Computing Operators Per Timestep ---")
    op_decay_list = []
    op_forcing_list = []
    for t in range(T):
        dt_t = dt[t:t+1]
        op_d, op_f = prop.get_transition_operators(dt_t)
        op_d = op_d.squeeze()
        op_f = op_f.squeeze()
        op_decay_list.append(op_d)
        op_forcing_list.append(op_f)
        if t == 0:
            print_tensor_stats(f"  op_decay[{t}]", op_d)
            print_tensor_stats(f"  op_forcing[{t}]", op_f)

    op_decay = torch.stack(op_decay_list, dim=0).unsqueeze(0)
    op_forcing = torch.stack(op_forcing_list, dim=0).unsqueeze(0)

    op_decay = op_decay.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
    op_forcing = op_forcing.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)

    print("\nInput shapes:")
    print(f"  forcing: {forcing.shape}")
    print(f"  op_decay: {op_decay.shape}")
    print(f"  op_forcing: {op_forcing.shape}")

    print("\n--- PARALLEL MODE ---")

    X_time = forcing * op_forcing
    A_time = op_decay

    A_flat = A_time.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)
    X_flat = X_time.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, T, D, 1)

    print(f"  A_flat shape: {A_flat.shape}")
    print(f"  X_flat shape: {X_flat.shape}")

    Y_flat = pscan(A_flat, X_flat)
    Y_parallel = Y_flat.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)

    print_tensor_stats("  Y_parallel", Y_parallel)

    print("\n--- SERIAL MODE ---")

    h_serial = torch.zeros(B, H, W, D, dtype=torch.cdouble, device=device)
    Y_serial_list = []

    for t in range(T):
        forcing_t = forcing[:, t] * op_forcing[:, t]

        if t == 0:
            h_serial = forcing_t
        else:
            h_serial = op_decay[:, t] * h_serial + forcing_t

        Y_serial_list.append(h_serial.clone())

    Y_serial = torch.stack(Y_serial_list, dim=1)
    print_tensor_stats("  Y_serial", Y_serial)

    print("\n--- COMPARISON ---")

    for t in range(T):
        diff = (Y_parallel[:, t] - Y_serial[:, t]).abs().max().item()
        print(f"  t={t}: Diff={diff:.2e}")

    total_diff = (Y_parallel - Y_serial).abs().max().item()
    print(f"\nTotal Max Difference: {total_diff:.2e}")
    print(f"Match: {total_diff < 1e-5}")

    return total_diff


def check_flux_tracker_consistency():
    print_section("Step 4: Flux Tracker Parallel vs Serial")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    B, T, D = 1, 5, 4

    flux_tracker = GlobalFluxTracker(D).to(device).double()

    x_mean = torch.randn(B, T, D, dtype=torch.cdouble, device=device)

    print("--- PARALLEL ---")
    A_flux, X_flux = flux_tracker.get_operators(x_mean)
    print_tensor_stats("  A_flux", A_flux)
    print_tensor_stats("  X_flux", X_flux)

    A_flux_5d = A_flux.unsqueeze(-1)
    X_flux_5d = X_flux.unsqueeze(-1)
    flux_parallel = pscan(A_flux_5d, X_flux_5d).squeeze(-1)
    print_tensor_stats("  flux_parallel", flux_parallel)

    print("\n--- SERIAL ---")
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    flux_serial_list = []

    decay = flux_tracker._get_decay()
    print_tensor_stats("  decay", decay)

    for t in range(T):
        x_t = x_mean[:, t]
        x_cat = torch.cat([x_t.real, x_t.imag], dim=-1)
        x_in = flux_tracker.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        flux_input = torch.complex(x_re, x_im)

        if t == 0:
            flux_state = flux_input
        else:
            flux_state = decay * flux_state + flux_input

        flux_serial_list.append(flux_state.clone())

    flux_serial = torch.stack(flux_serial_list, dim=1)
    print_tensor_stats("  flux_serial", flux_serial)

    print("\n--- COMPARISON ---")
    for t in range(T):
        diff = (flux_parallel[:, t] - flux_serial[:, t]).abs().max().item()
        print(f"  t={t}: Diff={diff:.2e}")

    total_diff = (flux_parallel - flux_serial).abs().max().item()
    print(f"\nTotal flux diff: {total_diff:.2e}")

    return total_diff


def check_model_forward_step():
    print_section("Step 5: Check UniPhyBlock forward vs forward_step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    from ModelUniPhy import UniPhyBlock

    B, T, D, H, W = 1, 3, 4, 4, 4

    block = UniPhyBlock(
        dim=D, expand=2, num_experts=2,
        img_height=H, img_width=W,
        sde_mode="det"
    ).to(device).double()
    block.eval()

    x = torch.randn(B, T, D, H, W, dtype=torch.cdouble, device=device)
    dt = torch.ones(T, dtype=torch.float64, device=device)

    print("--- PARALLEL FORWARD ---")
    with torch.no_grad():
        out_parallel = block(x, dt)
    print_tensor_stats("  out_parallel", out_parallel)

    print("\n--- SERIAL FORWARD_STEP ---")
    h_state = torch.zeros(B * H * W, 1, D, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    out_serial_list = []

    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t]
            out_t, h_state, flux_state = block.forward_step(x_t, h_state, dt[t], flux_state)
            out_serial_list.append(out_t)

    out_serial = torch.stack(out_serial_list, dim=1)
    print_tensor_stats("  out_serial", out_serial)

    print("\n--- COMPARISON ---")
    for t in range(T):
        diff = (out_parallel[:, t] - out_serial[:, t]).abs().max().item()
        print(f"  t={t}: Diff={diff:.2e}")

    total_diff = (out_parallel - out_serial).abs().max().item()
    print(f"\nTotal diff: {total_diff:.2e}")

    return total_diff


def trace_model_internals():
    print_section("Step 6: Trace Model Internals Step by Step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    from ModelUniPhy import UniPhyBlock

    B, T, D, H, W = 1, 3, 4, 4, 4

    block = UniPhyBlock(
        dim=D, expand=2, num_experts=2,
        img_height=H, img_width=W,
        sde_mode="det"
    ).to(device).double()
    block.eval()

    x = torch.randn(B, T, D, H, W, dtype=torch.cdouble, device=device)
    dt = torch.ones(T, dtype=torch.float64, device=device)

    print("--- TRACING PARALLEL forward() ---")

    x_flat = x.reshape(B * T, D, H, W)
    x_real = torch.cat([x_flat.real, x_flat.imag], dim=1)
    x_norm = block.norm_spatial(x_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_spatial = block.spatial_cliff(x_norm)

    x_re, x_im = torch.chunk(x_spatial, 2, dim=1)
    x_spatial_complex = torch.complex(x_re, x_im)
    x_spatial_5d = x_spatial_complex.reshape(B, T, D, H, W)

    x_perm = x_spatial_5d.permute(0, 1, 3, 4, 2)
    x_eigen_par = block.prop.basis.encode(x_perm)
    print_tensor_stats("x_eigen (parallel)", x_eigen_par)

    x_mean_par = x_eigen_par.mean(dim=(2, 3))
    print_tensor_stats("x_mean (parallel)", x_mean_par)

    A_flux, X_flux = block.prop.flux_tracker.get_operators(x_mean_par)
    print_tensor_stats("A_flux", A_flux)
    print_tensor_stats("X_flux", X_flux)

    flux_seq = pscan(A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)).squeeze(-1)
    print_tensor_stats("flux_seq (parallel)", flux_seq)

    source_seq = block.prop.flux_tracker.project(flux_seq)
    print_tensor_stats("source_seq", source_seq)

    gate_seq = torch.sigmoid(
        block.prop.flux_tracker.gate_net(
            torch.cat([flux_seq.real, flux_seq.imag], dim=-1)
        )
    )
    print_tensor_stats("gate_seq", gate_seq)

    print("\n--- TRACING SERIAL forward_step() ---")

    h_state = torch.zeros(B * H * W, 1, D, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)

    for t in range(min(T, 2)):
        print(f"\n  t={t}:")
        x_t = x[:, t]

        x_real_t = torch.cat([x_t.real, x_t.imag], dim=1)
        x_norm_t = block.norm_spatial(x_real_t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spatial_t = block.spatial_cliff(x_norm_t)

        x_s_re, x_s_im = torch.chunk(x_spatial_t, 2, dim=1)
        x_spatial_complex_t = torch.complex(x_s_re, x_s_im)

        x_perm_t = x_spatial_complex_t.permute(0, 2, 3, 1)
        x_eigen_t = block.prop.basis.encode(x_perm_t)
        print_tensor_stats(f"    x_eigen_t", x_eigen_t)

        x_mean_t = x_eigen_t.mean(dim=(1, 2))
        print_tensor_stats(f"    x_mean_t", x_mean_t)

        par_x_mean = x_mean_par[:, t]
        diff_x_mean = (x_mean_t - par_x_mean).abs().max().item()
        print(f"    x_mean diff from parallel: {diff_x_mean:.2e}")

        flux_next, source_t, gate_t = block.prop.flux_tracker.forward_step(flux_state, x_mean_t)
        print_tensor_stats(f"    flux_next", flux_next)
        print_tensor_stats(f"    source_t", source_t)
        print_tensor_stats(f"    gate_t", gate_t)

        par_flux = flux_seq[:, t]
        diff_flux = (flux_next - par_flux).abs().max().item()
        print(f"    flux diff from parallel: {diff_flux:.2e}")

        par_source = source_seq[:, t]
        diff_source = (source_t - par_source).abs().max().item()
        print(f"    source diff from parallel: {diff_source:.2e}")

        par_gate = gate_seq[:, t]
        diff_gate = (gate_t - par_gate).abs().max().item()
        print(f"    gate diff from parallel: {diff_gate:.2e}")

        flux_state = flux_next


def check_forward_step_recurrence():
    print_section("Step 7: Check forward_step Recurrence Formula")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    D = 4
    B, H, W = 1, 4, 4

    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()

    print("Checking TemporalPropagator.forward_step implementation...")

    if not hasattr(prop, 'forward_step'):
        print("  WARNING: forward_step not found in TemporalPropagator!")
        print("  This may cause inconsistency between parallel and serial modes.")
        return False

    import inspect
    sig = inspect.signature(prop.forward_step)
    print(f"  forward_step signature: {sig}")

    source = inspect.getsource(prop.forward_step)
    print(f"\n  forward_step source code:")
    for line in source.split('\n')[:20]:
        print(f"    {line}")

    return True


def diagnose_issue():
    print_section("Step 8: Diagnose Root Cause")

    print("""
Based on the analysis, let's check the key difference:

PARALLEL forward() in ModelUniPhy.py [4]:
=========================================
1. forcing = x_eigen * gate + source * (1 - gate)
2. u_t = forcing * op_forcing
3. A_time = op_decay
4. Y = pscan(A_time, u_t)
   -> Y[0] = u_t[0] = forcing[0] * op_forcing[0]
   -> Y[t] = A[t] * Y[t-1] + u_t[t]

SERIAL forward_step() in ModelUniPhy.py [4]:
============================================
Calls self.prop.forward_step() which is in UniPhyOps.py [8]

The question is: Does prop.forward_step use the same formula?

Let's check if h_prev is encoded/decoded correctly.
""")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    D = 4
    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()

    print("Checking if basis encode/decode is invertible...")
    x_test = torch.randn(1, 4, 4, D, dtype=torch.cdouble, device=device)
    x_encoded = prop.basis.encode(x_test)
    x_decoded = prop.basis.decode(x_encoded)
    diff = (x_test - x_decoded).abs().max().item()
    print(f"  encode->decode diff: {diff:.2e}")

    if diff > 1e-5:
        print("  WARNING: basis encode/decode is not invertible!")
        print("  This could cause issues in forward_step.")


def main():
    print("=" * 70)
    print("DETAILED PSCAN PARALLEL VS SERIAL DEBUGGER")
    print("=" * 70)

    results = {}

    results['pscan_formula'] = check_pscan_recurrence()

    check_transition_operators()

    results['parallel_vs_serial'] = check_parallel_vs_serial_simplified()

    results['flux_tracker'] = check_flux_tracker_consistency()

    results['model_forward'] = check_model_forward_step()

    trace_model_internals()

    check_forward_step_recurrence()

    diagnose_issue()

    print_section("SUMMARY")

    for name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
        else:
            status = f"diff={result:.2e}, {'PASS' if result < 1e-3 else 'FAIL'}"
        print(f"  {name}: {status}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    