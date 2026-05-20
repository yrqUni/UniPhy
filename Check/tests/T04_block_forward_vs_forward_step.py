import sys

import torch

from Check.utils import (
    build_check_model,
    make_dt,
    make_input,
    max_diff,
    write_result,
)

TEST_ID = "T04"


def _block_pair_err(block, h_prev, flux_prev, latent, dt):
    seq_out, _seq_h, _seq_flux = block.forward(
        latent[:, :2],
        h_prev,
        dt[:, :2],
        flux_prev,
    )
    step_out, _step_h, _step_flux = block.forward_step(
        latent[:, 0],
        latent[:, 1],
        h_prev,
        dt[:, 0],
        dt[:, 1],
        flux_prev,
    )
    return max_diff(seq_out[:, 0], step_out)


def _manual_full_etd2(model, x, dt):
    batch_size, steps = x.shape[:2]
    if steps < 2:
        raise ValueError("need steps >= 2")
    device = x.device
    dt_seq = model._normalize_dt(dt, batch_size, steps, device)
    z_all = model.encoder(x)
    states = model._init_states(batch_size, device, z_all.dtype)

    z_skip = z_all[:, 0]
    latent_running = z_all
    for block_idx, block in enumerate(model.blocks):
        h_prev, flux_prev = states[block_idx]
        seq_out, _h, _flux = block.forward(
            latent_running,
            h_prev,
            dt_seq,
            flux_prev,
        )
        latent_running = seq_out

    out0 = model.decoder(
        model._apply_decoder_skip(
            latent_running[:, 0].unsqueeze(1),
            z_skip.unsqueeze(1),
        )
    )[:, 0]
    return out0


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device, torch.float64)
    model.eval()

    x = make_input(2, 5, 4, 32, 32, device, torch.float64)
    dt = make_dt(2, 5, device, torch.float64)

    with torch.no_grad():
        latent = model.encoder(x)
        h_prev, flux_prev = model._init_states(x.shape[0], device, latent.dtype)[0]
        err_block = _block_pair_err(model.blocks[0], h_prev, flux_prev, latent, dt)

    with torch.no_grad():
        out_seq = model(x, dt)
        out_manual0 = _manual_full_etd2(model, x, dt)
    err_model = max_diff(out_seq[:, 0], out_manual0)

    max_err = max(err_block, err_model)
    passed = err_block < 1e-10 and err_model < 1e-10
    detail = f"err_block={err_block:.2e} err_model_step0={err_model:.2e}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
