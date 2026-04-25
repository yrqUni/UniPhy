import sys

import torch

from Check.utils import (
    compute_channelwise_crps_error,
    compute_crps_error,
    write_result,
)
from Exp.ERA5.runtime_config import compute_channelwise_crps, compute_weighted_crps

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
    err_channel, _, _ = compute_channelwise_crps_error(
        pred_channels,
        target_channels,
        lat_weights,
    )

    align_terms = []
    for member in range(pred_channels.shape[0]):
        align_terms.append(
            (lat_weights * (pred_channels[member] - target_channels).abs()).mean()
        )
    align_forward = torch.stack(align_terms).mean()
    pairwise_total = torch.tensor(0.0, device=device)
    for left in range(pred_channels.shape[0]):
        for right in range(left + 1, pred_channels.shape[0]):
            pairwise_total = pairwise_total + (
                lat_weights * (pred_channels[left] - pred_channels[right]).abs()
            ).mean()
    align_reference = align_forward - pairwise_total / (pred_channels.shape[0] ** 2)
    align_actual = compute_weighted_crps(pred_channels, target_channels, lat_weights)
    align_err = float((align_reference - align_actual).abs().item())

    single_pred = torch.randn(1, 2, 3, 5, 8, 9, device=device)
    single_target = torch.randn(2, 3, 5, 8, 9, device=device)
    single_weighted_expected = (
        lat_weights * (single_pred[0] - single_target).abs()
    ).mean()
    single_weighted_actual = compute_weighted_crps(
        single_pred,
        single_target,
        lat_weights,
    )
    single_channel_expected = (
        lat_weights * (single_pred[0] - single_target).abs()
    ).mean(dim=(-2, -1))
    single_channel_actual = compute_channelwise_crps(
        single_pred,
        single_target,
        lat_weights,
    )
    single_err = max(
        float((single_weighted_actual - single_weighted_expected).abs().item()),
        float((single_channel_actual - single_channel_expected).abs().max().item()),
    )

    max_err = max(err_crps, err_channel, align_err, single_err)
    passed = max_err < 1e-6
    detail = (
        f"err_crps={err_crps:.2e} err_channel={err_channel:.2e} "
        f"align_err={align_err:.2e} single_err={single_err:.2e} "
        f"actual={float(actual_crps.item()):.6f} "
        f"expected={float(expected_crps.item()):.6f}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
