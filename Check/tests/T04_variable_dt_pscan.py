import sys
from pathlib import Path

from Check.utils import complex_randn, sequential_ssm_recurrence, write_result

import torch
from Model.UniPhy.PScan import pscan
from Model.UniPhy.UniPhyOps import TemporalPropagator

TEST_ID = "T04"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size, steps, dim = 2, 16, 8
    prop = TemporalPropagator(dim=dim, dt_ref=6.0, init_noise_scale=1e-4).to(device)
    prop.eval()
    patterns = {
        "uniform": torch.full((batch_size, steps), 6.0, device=device),
        "random": torch.rand(batch_size, steps, device=device) * 23 + 1,
        "substep": torch.tensor([1, 2, 3, 6] * 4, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size, 1),
    }
    max_err = 0.0
    with torch.no_grad():
        for dt_seq in patterns.values():
            decay_seq, _ = prop.get_transition_operators_seq(dt_seq)
            decay_seq = decay_seq.unsqueeze(-1)
            x_seq = complex_randn((batch_size, steps, dim, 1), device)
            h0 = torch.zeros(batch_size, dim, device=device, dtype=torch.complex64)
            ref = sequential_ssm_recurrence(
                decay_seq.squeeze(-1), x_seq.squeeze(-1), h0
            )
            got = pscan(decay_seq, x_seq).squeeze(-1)
            err = float((ref - got).abs().max().item())
            max_err = max(max_err, err)
    status = "PASS" if max_err < 1e-4 else "FAIL"
    detail = f"max_err={max_err:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
