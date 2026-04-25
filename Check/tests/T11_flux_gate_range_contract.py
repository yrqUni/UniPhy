import sys

import torch

from Check.utils import write_result
from Model.UniPhy.UniPhyOps import GlobalFluxTracker

TEST_ID = "T11"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    tracker = GlobalFluxTracker(dim=16, dt_ref=1.0).to(device)
    batch_size, steps = 4, 7
    x_mean_seq = torch.complex(
        torch.randn(batch_size, steps, 16, device=device),
        torch.randn(batch_size, steps, 16, device=device),
    )
    dt_seq = torch.rand(batch_size, steps, device=device) * 1.5 + 0.1
    flux_prev = tracker.get_initial_state(batch_size, device, torch.complex64)
    with torch.no_grad():
        flux_seq = torch.zeros(
            batch_size,
            steps,
            tracker.state_dim,
            device=device,
            dtype=torch.complex64,
        )
        flux_state = flux_prev
        for step in range(steps):
            flux_state, _, _ = tracker.forward_step(
                flux_state,
                x_mean_seq[:, step],
                dt_seq[:, step],
            )
            flux_seq[:, step] = flux_state
        _, gate_seq = tracker.compute_output_seq(flux_seq)
    gate_min = float(gate_seq.min().item())
    gate_max = float(gate_seq.max().item())
    passed = gate_min >= tracker.gate_min - 1e-6 and gate_max <= tracker.gate_max + 1e-6
    max_err = max(tracker.gate_min - gate_min, gate_max - tracker.gate_max, 0.0)
    detail = f"gate_min={gate_min:.6f} gate_max={gate_max:.6f}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
