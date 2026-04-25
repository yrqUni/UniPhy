import sys

import torch

from Check.utils import build_check_model, make_input, write_result

TEST_ID = "T08"


def expect_value_error(fn):
    try:
        fn()
        return False
    except ValueError:
        return True
    except Exception:
        return False


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = build_check_model(device, torch.float64)
    model.eval()

    x = make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_zero = torch.zeros(2, 4, device=device, dtype=torch.float64)
    with torch.no_grad():
        out = model(x, dt_zero)
    err_zero = float((out - x).abs().max().item())

    invalid_results = [
        expect_value_error(
            lambda: model.forward(
                x[:, :1],
                torch.tensor([[-1.0], [-1.0]], device=device, dtype=torch.float64),
            )
        ),
        expect_value_error(
            lambda: model.forward(
                x[:, :1],
                torch.tensor(
                    [[float("nan")], [float("inf")]],
                    device=device,
                    dtype=torch.float64,
                ),
            )
        ),
        expect_value_error(
            lambda: model.forward_rollout(
                x[:, :1],
                torch.tensor([[1.0], [1.0]], device=device, dtype=torch.float64),
                [
                    torch.tensor([0.5, 0.5], device=device, dtype=torch.float64),
                    torch.tensor([-1.0, -1.0], device=device, dtype=torch.float64),
                ],
            )
        ),
        expect_value_error(
            lambda: model._normalize_dt(torch.tensor(1.0), 2, 4, device)
        ),
        expect_value_error(
            lambda: model._normalize_dt(
                torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
                2,
                4,
                device,
            )
        ),
    ]

    passed = err_zero < 1e-12 and all(invalid_results)
    max_err = max(err_zero, float(len(invalid_results) - sum(invalid_results)))
    detail = (
        f"err_zero={err_zero:.2e} invalid_cases={sum(invalid_results)}/"
        f"{len(invalid_results)}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
