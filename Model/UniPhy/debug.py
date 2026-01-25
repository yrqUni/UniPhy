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

    dt_expanded = dt.view(1, T, 1, 1, 1).expand(B, T, H, W, 1)
    op_decay, op_forcing = prop.get_transition_operators(dt_expanded)

    op_decay = op_decay.expand(B, T, H, W, D)
    op_forcing = op_forcing.expand(B, T, H, W, D)

    print("Input shapes:")
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


def trace_forward_step_internals():
    print_section("Step 6: Trace forward_step Internals")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    D = 4
    B, H, W = 1, 4, 4

    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()

    x_spatial = torch.randn(B, H, W, D, dtype=torch.cdouble, device=device)
    h_prev = torch.zeros(B, H, W, D, dtype=torch.cdouble, device=device)
    flux_prev = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    dt = torch.tensor(1.0, device=device, dtype=torch.float64)

    x_eigen = prop.basis.encode(x_spatial)
    print_tensor_stats("x_eigen", x_eigen)

    x_mean = x_eigen.mean(dim=(1, 2))
    print_tensor_stats("x_mean", x_mean)

    flux_next, source, gate = prop.flux_tracker.forward_step(flux_prev, x_mean)
    print_tensor_stats("flux_next", flux_next)
    print_tensor_stats("source", source)
    print_tensor_stats("gate", gate)

    dt_expanded = dt.view(1, 1, 1, 1)
    op_decay, op_forcing = prop.get_transition_operators(dt_expanded)
    print_tensor_stats("op_decay", op_decay)
    print_tensor_stats("op_forcing", op_forcing)

    source_exp = source.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
    gate_exp = gate.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)

    forcing = x_eigen * gate_exp + source_exp * (1 - gate_exp)
    print_tensor_stats("forcing", forcing)

    h_next = h_prev * op_decay + forcing * op_forcing
    print_tensor_stats("h_next (t=0, h_prev=0)", h_next)

    print("\nExpected for t=0 with h_prev=0:")
    print("  h_next = 0 * op_decay + forcing * op_forcing = forcing * op_forcing")
    expected = forcing * op_forcing
    diff = (h_next - expected).abs().max().item()
    print(f"  Diff from expected: {diff:.2e}")


def compare_forward_formulas():
    print_section("Step 7: Compare Forward Formulas in Detail")

    print("""
PARALLEL forward() 中的公式 [4]:
================================
1. x_eigen = basis.encode(x_spatial)
2. A_flux, X_flux = flux_tracker.get_operators(x_mean)
3. flux_seq = pscan(A_flux, X_flux)
4. source_seq = flux_tracker.project(flux_seq)
5. gate_seq = sigmoid(gate_net(flux_seq))
6. forcing = x_eigen * gate + source * (1 - gate)
7. u_t = forcing * op_forcing        <-- X for PScan
8. A_time = op_decay                  <-- A for PScan  
9. Y = pscan(A_time, u_t)
   -> Y[0] = u_t[0] = forcing[0] * op_forcing[0]
   -> Y[t] = A[t] * Y[t-1] + u_t[t]
           = op_decay * Y[t-1] + forcing[t] * op_forcing


SERIAL forward_step() 中的公式 [8]:
==================================
1. x_tilde = basis.encode(x_input)
2. flux_next, source, gate = flux_tracker.forward_step(flux_state, x_mean)
3. forcing_term = x_tilde * gate + source * (1 - gate)
4. h_next = h_prev * op_decay + forcing_term * op_forcing

对于 t=0, h_prev=0:
  h[0] = 0 * op_decay + forcing[0] * op_forcing = forcing[0] * op_forcing  ✓

对于 t=1:
  h[1] = h[0] * op_decay + forcing[1] * op_forcing
       = forcing[0] * op_forcing * op_decay + forcing[1] * op_forcing

而 PScan:
  Y[1] = A[1] * Y[0] + X[1]
       = op_decay * (forcing[0] * op_forcing) + forcing[1] * op_forcing
       = forcing[0] * op_forcing * op_decay + forcing[1] * op_forcing  ✓

公式一致！
""")


def check_flux_get_operators_vs_forward_step():
    print_section("Step 8: Check flux_tracker.get_operators vs forward_step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    D = 4
    B, T = 1, 3

    flux_tracker = GlobalFluxTracker(D).to(device).double()

    x_mean_seq = torch.randn(B, T, D, dtype=torch.cdouble, device=device)

    print("--- get_operators (for parallel) ---")
    A_flux, X_flux = flux_tracker.get_operators(x_mean_seq)

    print(f"A_flux shape: {A_flux.shape}")
    print(f"X_flux shape: {X_flux.shape}")

    for t in range(T):
        print(f"  t={t}: A_flux={A_flux[0, t, 0].item():.4f}, X_flux={X_flux[0, t, 0].item():.4f}")

    print("\n--- forward_step (for serial) ---")
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    decay = flux_tracker._get_decay()
    print(f"decay: {decay[0].item():.4f}")

    for t in range(T):
        x_t = x_mean_seq[:, t]

        x_cat = torch.cat([x_t.real, x_t.imag], dim=-1)
        x_in = flux_tracker.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        flux_input = torch.complex(x_re, x_im)

        print(f"  t={t}: flux_input={flux_input[0, 0].item():.4f}, expected X_flux={X_flux[0, t, 0].item():.4f}")

        diff = (flux_input - X_flux[:, t]).abs().max().item()
        print(f"       X_flux diff: {diff:.2e}")

        if t == 0:
            flux_state = flux_input
        else:
            flux_state = decay * flux_state + flux_input


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

    trace_forward_step_internals()

    compare_forward_formulas()

    check_flux_get_operators_vs_forward_step()

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
    