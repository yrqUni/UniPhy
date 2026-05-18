import sys

import torch

from Check.utils import build_check_model, make_dt, make_input, max_diff, write_result

TEST_ID = "T16"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = build_check_model(device, torch.float64)
    model.eval()

    x_context = make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = make_dt(2, 4, device, torch.float64)
    dt_ref = torch.ones(2, device=device, dtype=torch.float64)
    direct_dt = torch.full((2,), 4.0, device=device, dtype=torch.float64)

    with torch.no_grad():
        direct = model.forward_rollout(
            x_context,
            dt_context,
            [direct_dt],
            chunk_size=1,
        )
        recursive = model.forward_rollout(
            x_context,
            dt_context,
            [dt_ref, dt_ref, dt_ref, dt_ref],
            chunk_size=4,
        )[:, -1:]

    err = max_diff(direct, recursive)
    passed = err < 1e-10
    detail = f"err_direct_recursive={err:.2e}"
    return ("PASS" if passed else "FAIL"), err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
