import sys

import torch

from Check.utils import build_check_model, make_dt, make_input, manual_forward, max_diff, write_result

TEST_ID = "T04"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = build_check_model(device, torch.float64)
    model.eval()

    x = make_input(2, 5, 4, 32, 32, device, torch.float64)
    dt = make_dt(2, 5, device, torch.float64)
    with torch.no_grad():
        out_model = model(x, dt)
        out_manual = manual_forward(model, x, dt)
    err_model = max_diff(out_model, out_manual)

    x_single = make_input(2, 1, 4, 32, 32, device, torch.float64)
    dt_single = torch.full((2, 1), 0.75, device=device, dtype=torch.float64)
    with torch.no_grad():
        latent = model.encoder(x_single)[:, 0]
        h_prev, flux_prev = model._init_states(2, device, latent.dtype)[0]
        seq_out, seq_h, seq_flux = model.blocks[0].forward(
            latent.unsqueeze(1),
            h_prev,
            dt_single,
            flux_prev,
        )
        step_out, step_h, step_flux = model.blocks[0].forward_step(
            latent,
            h_prev,
            dt_single[:, 0],
            flux_prev,
        )
    err_block = max(
        max_diff(seq_out[:, 0], step_out),
        max_diff(seq_h, step_h),
        max_diff(seq_flux, step_flux),
    )

    max_err = max(err_model, err_block)
    passed = err_model < 1e-10 and err_block < 1e-10
    detail = f"err_model={err_model:.2e} err_block={err_block:.2e}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
