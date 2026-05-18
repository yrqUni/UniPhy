import sys

import torch

from Check.utils import max_diff, write_result

TEST_ID = "T10"


def sample_linear_sequence(frames, positions):
    rows = []
    for position in positions:
        lower_idx = int(torch.floor(position).item())
        upper_idx = int(torch.ceil(position).item())
        weight = position - lower_idx
        lower = frames[lower_idx]
        upper = frames[upper_idx]
        rows.append(lower * (1.0 - weight) + upper * weight)
    return torch.stack(rows, dim=0)


def compute_dt(positions, dt_ref):
    gaps = (positions[1:] - positions[:-1]) * dt_ref
    return torch.cat([positions.new_tensor([dt_ref]), gaps], dim=0)


def run():
    frames = torch.arange(9 * 3 * 2, dtype=torch.float64).reshape(9, 3, 2)
    positions = torch.tensor([0.0, 0.5, 2.0, 3.25, 5.0], dtype=torch.float64)
    sampled = sample_linear_sequence(frames, positions)
    expected = torch.stack(
        [
            frames[0],
            (frames[0] + frames[1]) * 0.5,
            frames[2],
            frames[3] * 0.75 + frames[4] * 0.25,
            frames[5],
        ],
        dim=0,
    )
    dt = compute_dt(positions, 6.0)
    expected_dt = torch.tensor([6.0, 3.0, 9.0, 7.5, 10.5], dtype=torch.float64)
    err_interp = max_diff(sampled, expected)
    err_dt = max_diff(dt, expected_dt)
    max_err = max(err_interp, err_dt)
    passed = max_err < 1e-12
    detail = f"err_interp={err_interp:.2e} err_dt={err_dt:.2e}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
