import sys

import torch

from Check.utils import build_check_model, full_serial_inference, make_dt, make_input, max_diff, write_result

TEST_ID = "T05"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = build_check_model(device, torch.float64)
    model.eval()

    x_context = make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = make_dt(2, 4, device, torch.float64)
    dt_list = [
        torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
        torch.tensor([0.75, 1.25], device=device, dtype=torch.float64),
    ]

    with torch.no_grad():
        rollout_serial = full_serial_inference(model, x_context, dt_context, dt_list)
        rollout_chunk_1 = model.forward_rollout(
            x_context,
            dt_context,
            dt_list,
            chunk_size=1,
        )
        rollout_chunk_3 = model.forward_rollout(
            x_context,
            dt_context,
            dt_list,
            chunk_size=3,
        )
        rollout_stride = model.forward_rollout(
            x_context,
            dt_context,
            dt_list,
            chunk_size=3,
            output_stride=2,
            output_offset=1,
        )

    err_serial = max_diff(rollout_chunk_1, rollout_serial)
    err_chunk = max_diff(rollout_chunk_1, rollout_chunk_3)
    err_stride = max_diff(rollout_stride, rollout_chunk_1[:, 1::2])

    max_err = max(err_serial, err_chunk, err_stride)
    passed = err_serial < 1e-7 and err_chunk < 1e-10 and err_stride < 1e-10
    detail = (
        f"err_serial={err_serial:.2e} err_chunk={err_chunk:.2e} "
        f"err_stride={err_stride:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
