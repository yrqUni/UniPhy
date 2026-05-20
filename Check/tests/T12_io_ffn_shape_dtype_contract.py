import sys

import torch

from Check.utils import max_diff, write_result
from Model.UniPhy.UniPhyFFN import ComplexLayerNorm2d, UniPhyFeedForwardNetwork
from Model.UniPhy.UniPhyIO import FlexiblePadder, UniPhyEncoder, UniPhyEnsembleDecoder

TEST_ID = "T12"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    padder = FlexiblePadder((4, 3)).to(device)
    x_pad = torch.arange(
        1 * 2 * 5 * 7,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 2, 5, 7)
    padded = padder(x_pad)
    pad_shape_ok = padded.shape[-2:] == (8, 9)
    pad_err = max_diff(padded[..., :5, :7], x_pad)

    encoder = UniPhyEncoder(4, 8, (4, 3), 5, 7).to(device)
    decoder = UniPhyEnsembleDecoder(4, 8, (4, 3), 8, 5, 7).to(device)
    x = torch.randn(2, 3, 4, 5, 7, device=device)
    with torch.no_grad():
        z = encoder(x)
        decoded = decoder(z)
    shape_ok = z.shape == (2, 3, 8, 2, 3) and decoded.shape == x.shape

    layer_norm = ComplexLayerNorm2d(8).to(device)
    real = torch.randn(2, 8, 4, 5, device=device)
    complex_x = torch.complex(real, torch.randn_like(real))
    normed = layer_norm(complex_x)
    norm_real_mean = float(normed.real.mean(dim=1).abs().max().item())
    norm_imag_mean = float(normed.imag.mean(dim=1).abs().max().item())

    ffn = UniPhyFeedForwardNetwork(8, 2).to(device)
    ffn_input = torch.complex(
        torch.randn(2, 8, 4, 5, device=device),
        torch.randn(2, 8, 4, 5, device=device),
    )
    ffn_out = ffn(ffn_input)
    ffn_shape_ok = ffn_out.shape == ffn_input.shape and ffn_out.is_complex()

    max_err = max(pad_err, norm_real_mean, norm_imag_mean)
    passed = (
        pad_shape_ok
        and shape_ok
        and ffn_shape_ok
        and pad_err == 0.0
        and norm_real_mean < 1e-5
        and norm_imag_mean < 1e-5
    )
    detail = (
        f"pad_shape_ok={pad_shape_ok} shape_ok={shape_ok} ffn_shape_ok={ffn_shape_ok} "
        f"pad_err={pad_err:.2e} norm_real_mean={norm_real_mean:.2e} "
        f"norm_imag_mean={norm_imag_mean:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
