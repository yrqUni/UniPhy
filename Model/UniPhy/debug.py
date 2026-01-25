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
              f"abs_mean={tensor.abs().mean().item():.6e}, "
              f"real_range=[{tensor.real.min().item():.4f}, {tensor.real.max().item():.4f}], "
              f"imag_range=[{tensor.imag.min().item():.4f}, {tensor.imag.max().item():.4f}]")
    else:
        print(f"{name}: shape={tuple(tensor.shape)}, "
              f"max={tensor.max().item():.6e}, "
              f"min={tensor.min().item():.6e}, "
              f"mean={tensor.mean().item():.6e}")


def check_pscan_recurrence():
    print_section("Step 1: Verify PScan Recurrence Formula")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, D = 2, 5, 4
    
    A = torch.randn(B, T, D, dtype=torch.cdouble, device=device) * 0.5
    X = torch.randn(B, T, D, dtype=torch.cdouble, device=device)
    
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
        print(f"  t={t}: PScan={Y_pscan[0, t, 0].item():.6f}, "
              f"Serial={Y_serial[0, t, 0].item():.6f}, Diff={diff:.2e}")
    
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
        dt = torch.tensor(dt_val, device=device, dtype=torch.float64)
        op_decay, op_forcing = prop.get_transition_operators(dt)
        
        print(f"dt={dt_val}:")
        print_tensor_stats("  op_decay", op_decay)
        print_tensor_stats("  op_forcing", op_forcing)
        
        lam = prop._get_effective_lambda()
        expected_decay = torch.exp(lam * dt_val / prop.dt_ref)
        expected_forcing = (expected_decay - 1) / (lam + 1e-8)
        
        decay_diff = (op_decay.squeeze() - expected_decay).abs().max().item()
        forcing_diff = (op_forcing.squeeze() - expected_forcing).abs().max().item()
        print(f"  decay formula verified: {decay_diff:.2e}")
        print(f"  forcing formula verified: {forcing_diff:.2e}")
        print()


def check_parallel_vs_serial_step_by_step():
    print_section("Step 3: Step-by-Step Comparison")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, H, W, D = 1, 5, 4, 4, 4
    
    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()
    basis = ComplexSVDTransform(D).to(device).double()
    flux_tracker = GlobalFluxTracker(D).to(device).double()
    
    x_input = torch.randn(B, T, H, W, D, dtype=torch.cdouble, device=device)
    dt = torch.ones(T, device=device, dtype=torch.float64)
    
    print("Input shape:", x_input.shape)
    print()
    
    print("-" * 70)
    print("PARALLEL MODE")
    print("-" * 70)
    
    x_eigen = basis.encode(x_input)
    print_tensor_stats("x_eigen (after basis.encode)", x_eigen)
    
    x_mean = x_eigen.mean(dim=(2, 3))
    A_flux, X_flux = flux_tracker.get_operators(x_mean)
    flux_seq = pscan(A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)).squeeze(-1)
    source_seq = flux_tracker.project(flux_seq)
    
    print_tensor_stats("flux_seq", flux_seq)
    print_tensor_stats("source_seq", source_seq)
    
    gate_seq = torch.sigmoid(
        flux_tracker.gate_net(torch.cat([flux_seq.real, flux_seq.imag], dim=-1))
    )
    print_tensor_stats("gate_seq", gate_seq)
    
    source_expanded = source_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
    gate_expanded = gate_seq.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, D)
    
    forcing_parallel = x_eigen * gate_expanded + source_expanded * (1 - gate_expanded)
    print_tensor_stats("forcing_parallel", forcing_parallel)
    
    dt_expanded = dt.view(1, T, 1, 1, 1).expand(B, T, H, W, 1)
    op_decay, op_forcing = prop.get_transition_operators(dt_expanded)
    
    op_decay = op_decay.view(B, T, 1, 1, D).expand(B, T, H, W, D)
    op_forcing = op_forcing.view(B, T, 1, 1, D).expand(B, T, H, W, D)
    
    print_tensor_stats("op_decay", op_decay[:, 0, 0, 0, :])
    print_tensor_stats("op_forcing", op_forcing[:, 0, 0, 0, :])
    
    X_time = forcing_parallel * op_forcing
    A_time = op_decay
    
    print_tensor_stats("X_time (forcing * op_forcing)", X_time)
    print_tensor_stats("A_time (op_decay)", A_time)
    
    A_flat = A_time.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
    X_flat = X_time.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D, 1)
    
    Y_flat = pscan(A_flat, X_flat)
    Y_parallel = Y_flat.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)
    
    print_tensor_stats("Y_parallel (pscan output)", Y_parallel)
    
    print()
    print("-" * 70)
    print("SERIAL MODE")
    print("-" * 70)
    
    h_serial = torch.zeros(B, H, W, D, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    Y_serial_list = []
    
    for t in range(T):
        x_t = x_input[:, t]
        x_eigen_t = basis.encode(x_t)
        
        x_mean_t = x_eigen_t.mean(dim=(1, 2))
        
        flux_decay = flux_tracker._get_decay()
        x_cat = torch.cat([x_mean_t.real, x_mean_t.imag], dim=-1)
        x_in = flux_tracker.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        flux_input = torch.complex(x_re, x_im)
        
        if t == 0:
            flux_state = flux_input
        else:
            flux_state = flux_decay * flux_state + flux_input
        
        source_t = flux_tracker.project(flux_state)
        gate_t = torch.sigmoid(
            flux_tracker.gate_net(torch.cat([flux_state.real, flux_state.imag], dim=-1))
        )
        
        source_exp = source_t.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        gate_exp = gate_t.unsqueeze(1).unsqueeze(2).expand(B, H, W, D)
        
        forcing_t = x_eigen_t * gate_exp + source_exp * (1 - gate_exp)
        
        dt_t = dt[t].view(1, 1, 1, 1)
        op_decay_t, op_forcing_t = prop.get_transition_operators(dt_t)
        
        if t == 0:
            h_serial = forcing_t * op_forcing_t
        else:
            h_serial = op_decay_t * h_serial + forcing_t * op_forcing_t
        
        Y_serial_list.append(h_serial.clone())
        
        if t < 3:
            print(f"\nt={t}:")
            print_tensor_stats(f"  x_eigen_t", x_eigen_t)
            print_tensor_stats(f"  flux_state", flux_state)
            print_tensor_stats(f"  forcing_t", forcing_t)
            print_tensor_stats(f"  h_serial", h_serial)
    
    Y_serial = torch.stack(Y_serial_list, dim=1)
    print_tensor_stats("\nY_serial (stacked)", Y_serial)
    
    print()
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    for t in range(T):
        diff = (Y_parallel[:, t] - Y_serial[:, t]).abs().max().item()
        parallel_val = Y_parallel[0, t, 0, 0, 0].item()
        serial_val = Y_serial[0, t, 0, 0, 0].item()
        print(f"t={t}: Parallel={parallel_val:.6f}, Serial={serial_val:.6f}, Diff={diff:.2e}")
    
    total_diff = (Y_parallel - Y_serial).abs().max().item()
    print(f"\nTotal Max Difference: {total_diff:.2e}")
    
    return total_diff


def check_flux_tracker_consistency():
    print_section("Step 4: Flux Tracker Parallel vs Serial")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, D = 1, 5, 4
    
    flux_tracker = GlobalFluxTracker(D).to(device).double()
    
    x_mean = torch.randn(B, T, D, dtype=torch.cdouble, device=device)
    
    print("Parallel flux computation:")
    A_flux, X_flux = flux_tracker.get_operators(x_mean)
    flux_parallel = pscan(A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)).squeeze(-1)
    print_tensor_stats("A_flux", A_flux)
    print_tensor_stats("X_flux", X_flux)
    print_tensor_stats("flux_parallel", flux_parallel)
    
    print("\nSerial flux computation:")
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    flux_serial_list = []
    
    decay = flux_tracker._get_decay()
    print_tensor_stats("decay", decay)
    
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
        
        diff = (flux_parallel[:, t] - flux_state).abs().max().item()
        print(f"t={t}: diff={diff:.2e}")
    
    flux_serial = torch.stack(flux_serial_list, dim=1)
    total_diff = (flux_parallel - flux_serial).abs().max().item()
    print(f"\nTotal flux diff: {total_diff:.2e}")
    
    return total_diff


def check_pscan_initial_condition():
    print_section("Step 5: PScan Initial Condition Analysis")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, D = 1, 5, 4
    
    A = torch.ones(B, T, D, dtype=torch.cdouble, device=device) * 0.9
    X = torch.randn(B, T, D, dtype=torch.cdouble, device=device)
    
    print("Case 1: Standard PScan (Y[0] = X[0])")
    Y_pscan = pscan(A.unsqueeze(-1), X.unsqueeze(-1)).squeeze(-1)
    
    Y_manual = torch.zeros_like(X)
    Y_manual[:, 0] = X[:, 0]
    for t in range(1, T):
        Y_manual[:, t] = A[:, t] * Y_manual[:, t - 1] + X[:, t]
    
    for t in range(T):
        print(f"  t={t}: PScan={Y_pscan[0, t, 0].item():.6f}, "
              f"Manual={Y_manual[0, t, 0].item():.6f}")
    
    print("\nCase 2: Recurrence with h[0] = 0 (h[t] = A*h[t-1] + X[t])")
    h = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    H_list = []
    for t in range(T):
        h = A[:, t] * h + X[:, t]
        H_list.append(h.clone())
    H_recur = torch.stack(H_list, dim=1)
    
    for t in range(T):
        print(f"  t={t}: H_recur={H_recur[0, t, 0].item():.6f}")
    
    print("\nCase 3: Match PScan with h[0]=0 recurrence")
    print("Need: Y[0] = X[0] to match h[0] = A[0]*0 + X[0] = X[0]")
    print("Then: Y[1] = A[1]*Y[0] + X[1] = A[1]*X[0] + X[1]")
    print("But:  h[1] = A[1]*h[0] + X[1] = A[1]*X[0] + X[1]")
    print("So they match!")
    
    diff = (Y_pscan - H_recur).abs().max().item()
    print(f"\nDiff between PScan and h[0]=0 recurrence: {diff:.2e}")
    
    return diff


def check_actual_model_forward():
    print_section("Step 6: Check Actual Model Code Paths")
    
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
    
    print("Running parallel forward...")
    with torch.no_grad():
        out_parallel = block(x, dt)
    print_tensor_stats("out_parallel", out_parallel)
    
    print("\nRunning serial forward...")
    h_state = torch.zeros(B * H * W, 1, D, dtype=torch.cdouble, device=device)
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    out_serial_list = []
    
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t]
            out_t, h_state, flux_state = block.forward_step(x_t, h_state, dt[t], flux_state)
            out_serial_list.append(out_t)
    
    out_serial = torch.stack(out_serial_list, dim=1)
    print_tensor_stats("out_serial", out_serial)
    
    diff = (out_parallel - out_serial).abs().max().item()
    print(f"\nFinal output diff: {diff:.2e}")
    
    return diff


def diagnose_forcing_construction():
    print_section("Step 7: Diagnose Forcing Construction Difference")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, H, W, D = 1, 3, 4, 4, 4
    
    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()
    basis = prop.basis
    flux_tracker = prop.flux_tracker
    
    x_spatial = torch.randn(B, T, H, W, D, dtype=torch.cdouble, device=device)
    dt = torch.ones(T, dtype=torch.float64, device=device)
    
    print("=" * 50)
    print("PARALLEL: forcing computation")
    print("=" * 50)
    
    x_eigen_par = basis.encode(x_spatial)
    x_mean_par = x_eigen_par.mean(dim=(2, 3))
    
    A_flux, X_flux = flux_tracker.get_operators(x_mean_par)
    flux_par = pscan(A_flux.unsqueeze(-1), X_flux.unsqueeze(-1)).squeeze(-1)
    source_par = flux_tracker.project(flux_par)
    gate_par = torch.sigmoid(
        flux_tracker.gate_net(torch.cat([flux_par.real, flux_par.imag], dim=-1))
    )
    
    for t in range(T):
        print(f"\nt={t}:")
        print(f"  x_mean_par: {x_mean_par[0, t, 0].item():.6f}")
        print(f"  A_flux: {A_flux[0, t, 0].item():.6f}")
        print(f"  X_flux: {X_flux[0, t, 0].item():.6f}")
        print(f"  flux_par: {flux_par[0, t, 0].item():.6f}")
        print(f"  source_par: {source_par[0, t, 0].item():.6f}")
        print(f"  gate_par: {gate_par[0, t, 0].item():.6f}")
    
    print("\n" + "=" * 50)
    print("SERIAL: forcing computation")
    print("=" * 50)
    
    flux_state = torch.zeros(B, D, dtype=torch.cdouble, device=device)
    
    for t in range(T):
        x_t = x_spatial[:, t]
        x_eigen_t = basis.encode(x_t)
        x_mean_t = x_eigen_t.mean(dim=(1, 2))
        
        decay = flux_tracker._get_decay()
        x_cat = torch.cat([x_mean_t.real, x_mean_t.imag], dim=-1)
        x_in = flux_tracker.input_mix(x_cat)
        x_re, x_im = torch.chunk(x_in, 2, dim=-1)
        flux_input = torch.complex(x_re, x_im)
        
        if t == 0:
            flux_state = flux_input
        else:
            flux_state = decay * flux_state + flux_input
        
        source_t = flux_tracker.project(flux_state)
        gate_t = torch.sigmoid(
            flux_tracker.gate_net(torch.cat([flux_state.real, flux_state.imag], dim=-1))
        )
        
        print(f"\nt={t}:")
        print(f"  x_mean_t: {x_mean_t[0, 0].item():.6f}")
        print(f"  flux_input: {flux_input[0, 0].item():.6f}")
        print(f"  flux_state: {flux_state[0, 0].item():.6f}")
        print(f"  source_t: {source_t[0, 0].item():.6f}")
        print(f"  gate_t: {gate_t[0, 0].item():.6f}")
        
        par_flux = flux_par[0, t, 0].item()
        ser_flux = flux_state[0, 0].item()
        print(f"  DIFF flux: {abs(par_flux - ser_flux):.2e}")


def check_forward_step_implementation():
    print_section("Step 8: Check forward_step vs forward Formulas")
    
    print("""
Based on code analysis:

PARALLEL forward() [ModelUniPhy.py]:
------------------------------------
1. x_eigen = basis.encode(x_spatial)
2. flux_seq = pscan(A_flux, X_flux)
3. forcing = x_eigen * gate + source * (1 - gate)
4. X_time = forcing * op_forcing
5. A_time = op_decay
6. Y = pscan(A_time, X_time)
   -> Y[0] = X[0] = forcing[0] * op_forcing[0]
   -> Y[t] = A[t] * Y[t-1] + X[t]
           = op_decay * Y[t-1] + forcing[t] * op_forcing

SERIAL forward_step():
----------------------
Need to check what TemporalPropagator.forward_step does!
""")
    
    print("Let's trace TemporalPropagator.forward_step:")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = 4
    prop = TemporalPropagator(D, dt_ref=1.0, sde_mode="det").to(device).double()
    
    print("\nTemporalPropagator attributes:")
    print(f"  basis: {type(prop.basis).__name__}")
    print(f"  flux_tracker: {type(prop.flux_tracker).__name__}")
    
    print("\nCheck if forward_step exists and what it does:")
    if hasattr(prop, 'forward_step'):
        import inspect
        sig = inspect.signature(prop.forward_step)
        print(f"  forward_step signature: {sig}")
    else:
        print("  forward_step NOT FOUND in TemporalPropagator!")
        print("  This might be the issue - serial mode might use different logic")


def main():
    print("=" * 70)
    print("DETAILED PSCAN PARALLEL VS SERIAL DEBUGGER")
    print("=" * 70)
    
    results = {}
    
    results['pscan_formula'] = check_pscan_recurrence()
    
    check_transition_operators()
    
    results['parallel_vs_serial'] = check_parallel_vs_serial_step_by_step()
    
    results['flux_tracker'] = check_flux_tracker_consistency()
    
    results['initial_condition'] = check_pscan_initial_condition()
    
    results['actual_model'] = check_actual_model_forward()
    
    diagnose_forcing_construction()
    
    check_forward_step_implementation()
    
    print_section("SUMMARY")
    
    for name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
        else:
            status = f"diff={result:.2e}"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    