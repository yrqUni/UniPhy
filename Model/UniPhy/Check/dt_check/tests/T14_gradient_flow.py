import sys

import torch

from dt_check.utils import make_tiny_model, write_result

TEST_ID = "T14_gradient_flow"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = make_tiny_model(device)
    model.train()
    batch_size = 2
    x_pixels = torch.randn(batch_size, 2, 4, 721, 1440, device=device)
    dt = torch.full((batch_size, 2), 6.0, device=device)
    noise = model.sample_noise(x_pixels)
    out, latent = model.forward(x_pixels, dt, z=noise, return_latent=True)
    out_loss = out.real.square().mean() if out.is_complex() else out.square().mean()
    latent_loss = latent.real.square().mean()
    if latent.is_complex():
        latent_loss = latent_loss + latent.imag.square().mean()
    loss = out_loss + latent_loss
    loss.backward()
    params = {
        "lam_re": model.blocks[0].prop.lam_re,
        "lam_im": model.blocks[0].prop.lam_im,
        "h0_re": model.blocks[0].prop.h0_re,
        "h0_im": model.blocks[0].prop.h0_im,
        "base_noise": model.blocks[0].prop.base_noise,
        "w_re": model.blocks[0].prop.basis.w_re,
        "w_im": model.blocks[0].prop.basis.w_im,
        "w_inv_re": model.blocks[0].prop.basis.w_inv_re,
        "w_inv_im": model.blocks[0].prop.basis.w_inv_im,
        "alpha_logit": model.blocks[0].prop.basis.alpha_logit,
        "decay_re": model.blocks[0].prop.flux_tracker.decay_re,
        "decay_im": model.blocks[0].prop.flux_tracker.decay_im,
        "ft_h0_re": model.blocks[0].prop.flux_tracker.h0_re,
        "ft_h0_im": model.blocks[0].prop.flux_tracker.h0_im,
    }
    failed = [
        name
        for name, param in params.items()
        if param.grad is None or float(param.grad.abs().max().item()) == 0.0
    ]
    max_err = float(len(failed))
    status = "PASS" if not failed else "FAIL"
    detail = f"zero_grad={','.join(failed)}" if failed else "all_params_have_gradients"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
