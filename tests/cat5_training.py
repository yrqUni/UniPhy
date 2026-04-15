import torch

from .suite_utils import (
    Recorder,
    build_model,
    build_train_cfg,
    build_train_optimizer,
    compute_stage1_terms,
    default_device,
    make_synthetic_batch,
    run_check,
    seed_all,
)

REQUIRED_PARAM_NAMES = [
    "encoder.proj.weight",
    "decoder.out_smooth.weight",
]
OPTIONAL_PROP_PARAM_NAMES = [
    "blocks.0.prop.lam_re",
    "blocks.0.prop.basis.w_re",
]


def _capture_params(model, names):
    named = dict(model.named_parameters())
    return {name: named[name].detach().clone() for name in names}



def check_parameters_update_after_single_step():
    device = default_device()
    cfg = build_train_cfg(patch_grid=2)
    model = build_model(cfg["model"], device)
    model.train()
    optimizer = build_train_optimizer(model, cfg)
    tracked_names = REQUIRED_PARAM_NAMES + OPTIONAL_PROP_PARAM_NAMES
    before = _capture_params(model, tracked_names)
    optimizer.zero_grad(set_to_none=True)
    data, dt = make_synthetic_batch(
        cfg["model"],
        device,
        batch_size=1,
        time_steps=4,
        scale=0.5,
    )
    loss = compute_stage1_terms(model, cfg, data, dt)["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite_loss={float(loss.item())}")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 128.0).item())
    if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm <= 0.0:
        raise RuntimeError(f"invalid_grad_norm={grad_norm}")
    optimizer.step()
    after = dict(model.named_parameters())
    deltas = {
        name: float((after[name].detach() - before[name]).abs().sum().item())
        for name in tracked_names
    }
    required_unchanged = [
        name for name in REQUIRED_PARAM_NAMES if deltas[name] == 0.0
    ]
    if required_unchanged:
        raise RuntimeError(f"unchanged_required_parameters={required_unchanged}")
    if not any(deltas[name] > 0.0 for name in OPTIONAL_PROP_PARAM_NAMES):
        raise RuntimeError(
            f"unchanged_propagator_parameters={OPTIONAL_PROP_PARAM_NAMES}"
        )
    return f"grad_norm={grad_norm:.6f} deltas={deltas}"



def check_short_training_run_stays_finite():
    device = default_device()
    cfg = build_train_cfg(patch_grid=2)
    model = build_model(cfg["model"], device)
    model.train()
    optimizer = build_train_optimizer(model, cfg)
    losses = []
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        data, dt = make_synthetic_batch(
            cfg["model"],
            device,
            batch_size=1,
            time_steps=4,
            scale=0.5,
        )
        loss = compute_stage1_terms(model, cfg, data, dt)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite_loss_step_{step}={float(loss.item())}")
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 128.0).item()
        )
        if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm <= 0.0:
            raise RuntimeError(f"invalid_grad_norm_step_{step}={grad_norm}")
        optimizer.step()
        losses.append(float(loss.item()))
    if max(losses) > max(losses[0] * 5.0, losses[0] + 5.0):
        raise RuntimeError(f"training_loss_diverged={losses}")
    return f"losses={losses}"



def main():
    seed_all(46)
    recorder = Recorder("5_training")
    run_check(
        recorder,
        "parameters_update_after_single_step",
        check_parameters_update_after_single_step,
    )
    run_check(
        recorder,
        "short_training_run_stays_finite",
        check_short_training_run_stays_finite,
    )
    recorder.finalize()



if __name__ == "__main__":
    main()
