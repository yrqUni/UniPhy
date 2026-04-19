import sys

import torch

from Model.UniPhy.ModelUniPhy import complex_dtype_for
from dt_check.utils import make_tiny_model, write_result

TEST_ID = "T08_rollout_dt_alignment"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size, steps_in = 2, 4
    model = make_tiny_model(device)
    block = model.blocks[0]
    x_context = torch.randn(batch_size, steps_in, 4, 721, 1440, device=device)
    dt_context = torch.full((batch_size, steps_in), 6.0, device=device)
    with torch.no_grad():
        z_ctx_full = model.encoder(x_context)
        z_ctx = z_ctx_full[:, :-1]
        dt_ctx_seq = dt_context[:, 1:]
        dtype = complex_dtype_for(z_ctx_full.dtype)
        h0, flux0 = model._init_states(batch_size, device, dtype)[0]
        _, h_a, _ = block.forward(z_ctx, h0, dt_ctx_seq, flux0)
        z_curr = z_ctx_full[:, 0]
        h_b, flux_b = h0, flux0
        for _ in range(steps_in - 1):
            z_curr, h_b, flux_b = block.forward_step(
                z_curr,
                h_b,
                torch.full((batch_size,), 6.0, device=device),
                flux_b,
            )
    err = float((h_a - h_b).abs().max().item())
    status = "PASS" if err < 1e-4 else "FAIL"
    detail = f"max_err={err:.2e}"
    return status, err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
