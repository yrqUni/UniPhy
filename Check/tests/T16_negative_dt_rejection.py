import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import make_tiny_model, write_result
else:
    from ..utils import make_tiny_model, write_result

import torch

TEST_ID = "T16"


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
    model = make_tiny_model(device)
    x = torch.randn(2, 1, 4, 721, 1440, device=device)
    results = [
        expect_value_error(
            lambda: model.forward(x, torch.tensor([[-1.0], [-1.0]], device=device))
        ),
        expect_value_error(
            lambda: model.forward(x, torch.tensor([[6.0], [-0.001]], device=device))
        ),
        expect_value_error(
            lambda: model.forward(
                x, torch.tensor([[-float("inf")], [-float("inf")]], device=device)
            )
        ),
        expect_value_error(
            lambda: model.forward_rollout(
                x,
                torch.tensor([[6.0], [6.0]], device=device),
                [
                    torch.tensor([6.0, 6.0], device=device),
                    torch.tensor([-1.0, -1.0], device=device),
                ],
            )
        ),
    ]
    passed = all(results)
    max_err = float(4 - sum(results))
    detail = f"sub_cases_passed={sum(results)}/4"
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
