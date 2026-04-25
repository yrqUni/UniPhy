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
        latent_zero = model.encoder(x[:, :1])[:, 0]
        states = model._init_states(2, device, latent_zero.dtype)
        z_zero, h_zero, flux_zero = model.blocks[0].forward_step(
            latent_zero,
            states[0][0],
            torch.zeros(2, device=device, dtype=torch.float64),
            states[0][1],
        )
        rollout_zero = model.forward_rollout(
            x[:, :2],
            dt_zero[:, :2],
            [torch.zeros(2, device=device, dtype=torch.float64)],
        )
    err_zero = float((out - x).abs().max().item())
    err_step = float((z_zero - latent_zero).abs().max().item())
    err_state = float((h_zero - states[0][0]).abs().max().item())
    err_flux = float((flux_zero - states[0][1]).abs().max().item())
    err_rollout = float((rollout_zero[:, 0] - x[:, 1]).abs().max().item())

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

    passed = (
        err_zero < 1e-12
        and err_step < 1e-12
        and err_state < 1e-12
        and err_flux < 1e-12
        and err_rollout < 1e-12
        and all(invalid_results)
    )
    max_err = max(
        err_zero,
        err_step,
        err_state,
        err_flux,
        err_rollout,
        float(len(invalid_results) - sum(invalid_results)),
    )
    detail = (
        f"err_zero={err_zero:.2e} err_step={err_step:.2e} err_state={err_state:.2e} "
        f"err_flux={err_flux:.2e} err_rollout={err_rollout:.2e} "
        f"invalid_cases={sum(invalid_results)}/{len(invalid_results)}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
