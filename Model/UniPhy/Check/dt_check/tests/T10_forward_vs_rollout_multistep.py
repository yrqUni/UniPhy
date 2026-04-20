import sys

import torch

from Model.UniPhy.ModelUniPhy import complex_dtype_for
from dt_check.utils import fit_log_slope, make_tiny_model, write_result

TEST_ID = "T10_forward_vs_rollout_multistep"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size = 2
    model = make_tiny_model(device)
    x_init = torch.randn(batch_size, 1, 4, 721, 1440, device=device)
    dt_6 = torch.full((batch_size,), 6.0, device=device)
    with torch.no_grad():
        z_curr = model.encoder(x_init)[:, 0]
        z_skip = z_curr.clone()
        states = model._init_states(batch_size, device, complex_dtype_for(z_curr.dtype))
        preds_a = []
        for _ in range(24):
            h_prev, flux_prev = states[0]
            z_curr, h_next, flux_next = model.blocks[0].forward_step(
                z_curr, h_prev, dt_6, flux_prev
            )
            states[0] = (h_next, flux_next)
            z_dec = model._apply_decoder_skip(z_curr, z_skip)
            pred = model.decoder(z_dec)
            preds_a.append(pred.real if pred.is_complex() else pred)
        preds_a = torch.stack(preds_a, dim=1)
        dt_list = [dt_6] * 24
        preds_b = model.forward_rollout(x_init, dt_6.unsqueeze(1), dt_list)
        preds_b = preds_b.real if preds_b.is_complex() else preds_b
    points = [1, 4, 8, 24]
    errs = []
    for n in points:
        errs.append(float((preds_a[:, n - 1] - preds_b[:, n - 1]).abs().max().item()))
    max_err = max(errs)
    slope = fit_log_slope(points, [max(err, 1e-30) for err in errs])
    status = "PASS" if max_err < 1e-4 and slope <= 2.0 else "FAIL"
    detail = (
        f"err_1={errs[0]:.2e} err_4={errs[1]:.2e} "
        f"err_8={errs[2]:.2e} err_24={errs[3]:.2e} slope={slope:.2f}"
    )
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
