import sys
from pathlib import Path

from Check.utils import make_tiny_model, write_result

import torch
from Model.UniPhy.ModelUniPhy import complex_dtype_for

TEST_ID = "T09"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size = 2
    model = make_tiny_model(device)
    x = torch.randn(batch_size, 1, 4, 721, 1440, device=device)
    dt = torch.full((batch_size, 1), 6.0, device=device)
    with torch.no_grad():
        z_curr = model.encoder(x)[:, 0]
        states = model._init_states(batch_size, device, complex_dtype_for(z_curr.dtype))
        h_prev, flux_prev = states[0]
        z_seq = z_curr.unsqueeze(1)
        out_seq, h_a, flux_a = model.blocks[0].forward(z_seq, h_prev, dt, flux_prev)
        z_out_a = out_seq[:, 0]
        z_out_b, h_b, flux_b = model.blocks[0].forward_step(
            z_curr, h_prev, dt[:, 0], flux_prev
        )
    err_z = float((z_out_a - z_out_b).abs().max().item())
    err_h = float((h_a - h_b).abs().max().item())
    err_flux = float((flux_a - flux_b).abs().max().item())
    max_err = max(err_z, err_h, err_flux)
    status = "PASS" if max_err < 1e-4 else "FAIL"
    detail = f"err_z={err_z:.2e} err_h={err_h:.2e} err_flux={err_flux:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
