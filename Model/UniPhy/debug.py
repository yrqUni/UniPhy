import torch
import torch.nn as nn
import sys

sys.path.append("/nfs/UniPhy/Model/UniPhy")

from ModelUniPhy import UniPhyModel, UniPhyBlock
from UniPhyOps import TemporalPropagator, GlobalFluxTracker, ComplexSVDTransform
from PScan import pscan


def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def cmp(name, t1, t2, atol=1e-5):
    diff = (t1 - t2).abs().max().item()
    ok = diff < atol
    print(f"{name}: diff={diff:.2e} [{'PASS' if ok else 'FAIL'}]")
    return ok, diff


def debug_single_block_per_timestep():
    sep("Single Block Per-Timestep Analysis")
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

    print("\n--- Parallel Mode ---")
    with torch.no_grad():
        z_par, h_par_final, flux_par_final = block(x, h_init, dt, flux_init)

    print(f"z_par shape: {z_par.shape}")
    print(f"h_par_final shape: {h_par_final.shape}")
    print(f"flux_par_final shape: {flux_par_final.shape}")

    print("\n--- Serial Mode ---")
    z_ser_list = []
    h_ser_list = []
    flux_ser_list = []

    h_prev = h_init.clone()
    flux_prev = flux_init.clone()

    with torch.no_grad():
        for t in range(T):
            z_t, h_next, flux_next = block.forward_step(x[:, t], h_prev, dt[t], flux_prev)
            z_ser_list.append(z_t.clone())
            h_ser_list.append(h_next.clone())
            flux_ser_list.append(flux_next.clone())
            h_prev = h_next
            flux_prev = flux_next

    z_ser = torch.stack(z_ser_list, dim=1)

    print(f"z_ser shape: {z_ser.shape}")

    print("\n--- Per-Timestep Comparison ---")
    for t in range(T):
        _, diff = cmp(f"z[{t}]", z_par[:, t], z_ser_list[t])

    print("\n--- Final State Comparison ---")
    cmp("h_final", h_par_final, h_ser_list[-1])
    cmp("flux_final", flux_par_final, flux_ser_list[-1])


def debug_multi_block_state_flow():
    sep("Multi-Block State Flow Analysis")
    torch.manual_seed(42)

    B, T, D, H, W = 1, 4, 32, 5, 5
    num_blocks = 2

    blocks = nn.ModuleList([
        UniPhyBlock(
            dim=D, expand=2, num_experts=4,
            img_height=H, img_width=W,
            dt_ref=1.0, sde_mode="ode"
        )
        for _ in range(num_blocks)
    ]).cuda().eval()

    z_input = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")

    def init_states():
        states = []
        for _ in range(num_blocks):
            h = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
            f = torch.zeros(B, D, dtype=torch.complex64, device="cuda")
            states.append((h, f))
        return states

    print("\n--- Parallel Mode (Process All T at Once) ---")
    with torch.no_grad():
        states_par = init_states()
        z_par = z_input.clone()

        z_par_per_block = [z_input.clone()]

        for i, block in enumerate(blocks):
            h_prev, flux_prev = states_par[i]
            z_par, h_next, flux_next = block(z_par, h_prev, dt, flux_prev)
            states_par[i] = (h_next, flux_next)
            z_par_per_block.append(z_par.clone())
            print(f"Block {i}: z shape={z_par.shape}, h shape={h_next.shape}")

    print("\n--- Serial Mode (Process One T at a Time) ---")
    with torch.no_grad():
        states_ser = init_states()

        z_ser_per_t = []

        for t in range(T):
            z_t = z_input[:, t].clone()

            for i, block in enumerate(blocks):
                h_prev, flux_prev = states_ser[i]
                z_t, h_next, flux_next = block.forward_step(z_t, h_prev, dt[t], flux_prev)
                states_ser[i] = (h_next, flux_next)

            z_ser_per_t.append(z_t.clone())
            print(f"t={t}: z shape={z_t.shape}")

        z_ser = torch.stack(z_ser_per_t, dim=1)

    print("\n--- Comparison ---")
    print(f"z_par shape: {z_par.shape}")
    print(f"z_ser shape: {z_ser.shape}")

    for t in range(T):
        _, diff = cmp(f"z[{t}] after all blocks", z_par[:, t], z_ser_per_t[t])


def debug_state_between_blocks():
    sep("State Between Blocks Analysis")
    torch.manual_seed(42)

    B, T, D, H, W = 1, 4, 32, 5, 5

    block0 = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    block1 = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    z_input = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")

    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")

    print("\n--- Parallel: Block0 processes all T, then Block1 processes all T ---")
    with torch.no_grad():
        z_after_b0_par, h_b0_final, flux_b0_final = block0(z_input, h_init, dt, flux_init)
        print(f"After Block0: z shape={z_after_b0_par.shape}")
        print(f"Block0 returns h_final (t={T-1}): shape={h_b0_final.shape}")

        z_after_b1_par, h_b1_final, flux_b1_final = block1(z_after_b0_par, h_init, dt, flux_init)
        print(f"After Block1: z shape={z_after_b1_par.shape}")

    print("\n--- Serial: For each t, run Block0 then Block1 ---")
    with torch.no_grad():
        h_b0 = h_init.clone()
        flux_b0 = flux_init.clone()
        h_b1 = h_init.clone()
        flux_b1 = flux_init.clone()

        z_ser_list = []

        for t in range(T):
            z_t = z_input[:, t].clone()

            z_t, h_b0, flux_b0 = block0.forward_step(z_t, h_b0, dt[t], flux_b0)

            z_t, h_b1, flux_b1 = block1.forward_step(z_t, h_b1, dt[t], flux_b1)

            z_ser_list.append(z_t.clone())
            print(f"t={t}: processed through both blocks")

        z_ser = torch.stack(z_ser_list, dim=1)

    print("\n--- Key Insight ---")
    print("Parallel mode: Block1 uses h_init (zeros) for ALL timesteps")
    print("Serial mode: Block1 uses accumulated state from previous timesteps")

    print("\n--- Comparison ---")
    for t in range(T):
        _, diff = cmp(f"z[{t}]", z_after_b1_par[:, t], z_ser_list[t])


def debug_correct_parallel_implementation():
    sep("Correct Parallel Implementation Test")
    torch.manual_seed(42)

    B, T, D, H, W = 1, 4, 32, 5, 5

    block0 = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    block1 = UniPhyBlock(
        dim=D, expand=2, num_experts=4,
        img_height=H, img_width=W,
        dt_ref=1.0, sde_mode="ode"
    ).cuda().eval()

    z_input = torch.randn(B, T, D, H, W, dtype=torch.complex64, device="cuda") * 0.1
    dt = torch.ones(T, device="cuda")

    h_init = torch.zeros(B * H * W, 1, D, dtype=torch.complex64, device="cuda")
    flux_init = torch.zeros(B, D, dtype=torch.complex64, device="cuda")

    print("\n--- Reference: Pure Serial ---")
    with torch.no_grad():
        h_b0 = h_init.clone()
        flux_b0 = flux_init.clone()
        h_b1 = h_init.clone()
        flux_b1 = flux_init.clone()

        z_ref_list = []
        for t in range(T):
            z_t = z_input[:, t].clone()
            z_t, h_b0, flux_b0 = block0.forward_step(z_t, h_b0, dt[t], flux_b0)
            z_t, h_b1, flux_b1 = block1.forward_step(z_t, h_b1, dt[t], flux_b1)
            z_ref_list.append(z_t)

        z_ref = torch.stack(z_ref_list, dim=1)

    print("\n--- Current Parallel (WRONG) ---")
    with torch.no_grad():
        z_wrong, _, _ = block0(z_input, h_init, dt, flux_init)
        z_wrong, _, _ = block1(z_wrong, h_init, dt, flux_init)

    print("\n--- Proposed Fix: Sequential Block Processing ---")
    print("Each block should process the ENTIRE sequence before passing to next block")
    print("But the state passed to block1 should be per-timestep, not just final state")

    print("\n--- Comparison ---")
    for t in range(T):
        _, diff_wrong = cmp(f"Wrong z[{t}]", z_wrong[:, t], z_ref_list[t])

    print("\n--- Root Cause ---")
    print("In parallel mode:")
    print("  - Block0 processes z[0:T] with h_init, outputs h_final (for t=T-1 only)")
    print("  - Block1 processes z[0:T] with h_init (NOT the accumulated state from Block0)")
    print("")
    print("In serial mode:")
    print("  - For each t: Block0(z[t]) -> Block1(z[t]), states accumulate correctly")


def main():
    print("=" * 70)
    print("  Detailed State Propagation Debug")
    print("=" * 70)

    debug_single_block_per_timestep()
    debug_multi_block_state_flow()
    debug_state_between_blocks()
    debug_correct_parallel_implementation()

    print("\n" + "=" * 70)
    print("  Debug Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    