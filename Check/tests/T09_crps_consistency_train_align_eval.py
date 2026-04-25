import sys

import torch

from Check.utils import (
    compute_channelwise_crps_error,
    compute_crps_error,
    write_result,
)

TEST_ID = "T09"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    pred_ensemble = torch.randn(4, 2, 3, 5, 6, 7, device=device)
    target = torch.randn(2, 3, 5, 6, 7, device=device)
    err_crps, actual_crps, expected_crps = compute_crps_error(pred_ensemble, target)

    pred_channels = torch.randn(4, 2, 3, 5, 8, 9, device=device)
    target_channels = torch.randn(2, 3, 5, 8, 9, device=device)
    lat_weights = torch.cos(
        torch.deg2rad(torch.linspace(-90, 90, 8, device=device))
    ).clamp_min(1e-6)
    lat_weights = (lat_weights / lat_weights.mean()).view(1, 1, 8, 1)
    err_channel, actual_channel, expected_channel = compute_channelwise_crps_error(
        pred_channels,
        target_channels,
        lat_weights,
    )

    align_terms = []
    for member in range(pred_ensemble.shape[0]):
        align_terms.append((pred_ensemble[member] - target).abs().mean())
    align_forward = torch.stack(align_terms).mean()
    pairwise_total = torch.tensor(0.0, device=device)
    for left in range(pred_ensemble.shape[0]):
        for right in range(left + 1, pred_ensemble.shape[0]):
            pairwise_total = pairwise_total + (
                pred_ensemble[left] - pred_ensemble[right]
            ).abs().mean()
    align_reference = align_forward - pairwise_total / (pred_ensemble.shape[0] ** 2)
    align_err = float((align_reference - expected_crps).abs().item())

    max_err = max(err_crps, err_channel, align_err)
    passed = max_err < 1e-6
    detail = (
        f"err_crps={err_crps:.2e} err_channel={err_channel:.2e} "
        f"align_err={align_err:.2e} actual={float(actual_crps.item()):.6f} "
        f"expected={float(expected_crps.item()):.6f}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
