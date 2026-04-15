import torch

from Exp.ERA5.align import compute_crps as align_compute_crps
from Exp.ERA5.train import compute_crps as train_compute_crps
from .suite_utils import (
    Recorder,
    build_align_cfg,
    build_model,
    build_train_cfg,
    compute_alignment_metrics,
    compute_stage1_terms,
    default_device,
    finite_number,
    gaussian_crps_closed_form,
    gaussian_quantile_ensemble,
    make_fast_align_cfg,
    make_synthetic_batch,
    run_check,
    seed_all,
)


def check_train_and_align_crps_match_analytic_case():
    device = default_device()
    pred = torch.tensor([[-1.0], [1.0]], device=device).view(2, 1, 1)
    target = torch.tensor([[0.0]], device=device).view(1, 1)
    expected = 0.5
    train_value = float(train_compute_crps(pred, target).item())
    align_value = float(align_compute_crps(pred, target).item())
    if abs(train_value - expected) > 1e-6:
        raise RuntimeError(f"train_crps={train_value}")
    if abs(align_value - expected) > 1e-6:
        raise RuntimeError(f"align_crps={align_value}")
    return (
        f"train={train_value:.6f} align={align_value:.6f} "
        f"expected={expected:.6f}"
    )


def check_quantile_ensemble_matches_gaussian_closed_form():
    device = default_device()
    mu = torch.tensor([[0.2]], device=device, dtype=torch.float64)
    sigma = torch.tensor([[1.3]], device=device, dtype=torch.float64)
    target = torch.tensor([[-0.7]], device=device, dtype=torch.float64)
    ensemble = gaussian_quantile_ensemble(mu, sigma, members=4096, device=device)
    empirical = float(train_compute_crps(ensemble, target).item())
    closed_form = float(gaussian_crps_closed_form(mu, sigma, target).item())
    diff = abs(empirical - closed_form)
    if diff > 2e-3:
        raise RuntimeError(f"gaussian_crps_diff={diff}")
    return (
        f"empirical={empirical:.6f} closed_form={closed_form:.6f} "
        f"diff={diff:.6f}"
    )


def check_stage1_loss_terms_are_finite_and_nonnegative():
    device = default_device()
    cfg = build_train_cfg(patch_grid=2)
    model = build_model(cfg["model"], device)
    model.train()
    data, dt = make_synthetic_batch(
        cfg["model"],
        device,
        batch_size=1,
        time_steps=4,
        scale=0.5,
    )
    terms = compute_stage1_terms(model, cfg, data, dt)
    for key in ["l1", "mse", "crps", "loss"]:
        value = float(terms[key].item())
        if not finite_number(value):
            raise RuntimeError(f"nonfinite_{key}={value}")
        if value < 0.0:
            raise RuntimeError(f"negative_{key}={value}")
    ensemble_size = int(cfg["model"]["ensemble_size"])
    if terms["ensemble_stack"].shape[0] != ensemble_size:
        raise RuntimeError(
            f"ensemble_stack_shape={tuple(terms['ensemble_stack'].shape)}"
        )
    return (
        f"l1={terms['l1'].item():.6f} mse={terms['mse'].item():.6f} "
        f"crps={terms['crps'].item():.6f} loss={terms['loss'].item():.6f}"
    )


def check_stage2_metrics_are_finite_and_nonnegative():
    device = default_device()
    cfg = make_fast_align_cfg(
        build_align_cfg(patch_grid=2),
        condition_steps=2,
        max_target_steps=1,
        sub_steps=(1,),
        max_rollout_steps=1,
        chunk_size=1,
    )
    model = build_model(cfg["model"], device)
    model.train()
    data, dt = make_synthetic_batch(
        cfg["model"],
        device,
        batch_size=1,
        time_steps=3,
        scale=0.5,
    )
    metrics = compute_alignment_metrics(model, cfg, data, dt, lr=1e-4)
    for key in ["loss", "l1", "crps", "rmse", "basis_residual"]:
        value = float(metrics[key])
        if not finite_number(value):
            raise RuntimeError(f"nonfinite_{key}={value}")
        if value < 0.0:
            raise RuntimeError(f"negative_{key}={value}")
    return (
        f"loss={metrics['loss']:.6f} l1={metrics['l1']:.6f} "
        f"crps={metrics['crps']:.6f} rmse={metrics['rmse']:.6f}"
    )


def main():
    seed_all(44)
    recorder = Recorder("3_loss")
    run_check(
        recorder,
        "train_and_align_crps_match_analytic_case",
        check_train_and_align_crps_match_analytic_case,
    )
    run_check(
        recorder,
        "quantile_ensemble_matches_gaussian_closed_form",
        check_quantile_ensemble_matches_gaussian_closed_form,
    )
    run_check(
        recorder,
        "stage1_loss_terms_are_finite_and_nonnegative",
        check_stage1_loss_terms_are_finite_and_nonnegative,
    )
    run_check(
        recorder,
        "stage2_metrics_are_finite_and_nonnegative",
        check_stage2_metrics_are_finite_and_nonnegative,
    )
    recorder.finalize()


if __name__ == "__main__":
    main()
