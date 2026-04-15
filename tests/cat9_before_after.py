import importlib

import torch

from Exp.ERA5.eval_forecast import safe_channel_values
from Exp.ERA5.runtime_config import build_runtime_cfg
from .suite_utils import (
    ALIGN_CFG_PATH,
    TRAIN_CFG_PATH,
    Recorder,
    build_align_cfg,
    build_model,
    build_train_cfg,
    build_train_optimizer,
    compute_alignment_metrics,
    compute_stage1_terms,
    default_device,
    finite_number,
    make_fast_align_cfg,
    make_synthetic_batch,
    run_check,
    seed_all,
)

ENTRYPOINT_MODULES = [
    "Exp.ERA5.train",
    "Exp.ERA5.align",
]


def check_entrypoint_modules_import():
    imported = []
    for name in ENTRYPOINT_MODULES:
        module = importlib.import_module(name)
        if module is None:
            raise RuntimeError(f"failed_import={name}")
        imported.append(name)
    return f"imported_entrypoints={imported}"


def check_entrypoint_main_symbols_exist():
    missing = []
    for name in ENTRYPOINT_MODULES:
        module = importlib.import_module(name)
        if name.endswith("train") and not hasattr(module, "train"):
            missing.append(name)
        if name.endswith("align") and not hasattr(module, "align"):
            missing.append(name)
    if missing:
        raise RuntimeError(f"missing_entry_symbols={missing}")
    return "entry_symbols_present"


def check_train_runtime_config_overrides():
    cfg = build_runtime_cfg(
        str(TRAIN_CFG_PATH),
        data_input_dir="/tmp/era5",
        train_year_range="2001,2002",
        sample_offsets_hours="0,3,9,18",
        epochs=7,
        log_path="/tmp/logs",
        ckpt_dir="/tmp/ckpt",
        ckpt_path="/tmp/ckpt/model.pt",
        max_steps=11,
    )
    expected_offsets = [0.0, 3.0, 9.0, 18.0]
    if cfg["data"]["input_dir"] != "/tmp/era5":
        raise RuntimeError(f"unexpected_input_dir={cfg['data']['input_dir']}")
    if cfg["data"]["year_range"] != [2001, 2002]:
        raise RuntimeError(f"unexpected_year_range={cfg['data']['year_range']}")
    if cfg["data"]["sample_offsets_hours"] != expected_offsets:
        raise RuntimeError(
            f"unexpected_offsets={cfg['data']['sample_offsets_hours']}"
        )
    if int(cfg["train"]["epochs"]) != 7:
        raise RuntimeError(f"unexpected_epochs={cfg['train']['epochs']}")
    if cfg["logging"]["log_path"] != "/tmp/logs":
        raise RuntimeError(f"unexpected_log_path={cfg['logging']['log_path']}")
    if cfg["logging"]["ckpt_dir"] != "/tmp/ckpt":
        raise RuntimeError(f"unexpected_ckpt_dir={cfg['logging']['ckpt_dir']}")
    if cfg["logging"]["ckpt"] != "/tmp/ckpt/model.pt":
        raise RuntimeError(f"unexpected_ckpt={cfg['logging']['ckpt']}")
    if int(cfg["runtime"]["max_steps"]) != 11:
        raise RuntimeError(f"unexpected_max_steps={cfg['runtime']['max_steps']}")
    return "train_runtime_config_overrides_ok"


def check_align_runtime_config_overrides():
    cfg = build_runtime_cfg(
        str(ALIGN_CFG_PATH),
        data_input_dir="/tmp/era5-align",
        train_year_range="2003,2004",
        pretrained_ckpt="/tmp/pretrained.pt",
        max_steps=13,
    )
    if cfg["data"]["input_dir"] != "/tmp/era5-align":
        raise RuntimeError(f"unexpected_input_dir={cfg['data']['input_dir']}")
    if cfg["data"]["year_range"] != [2003, 2004]:
        raise RuntimeError(f"unexpected_year_range={cfg['data']['year_range']}")
    if cfg["alignment"]["pretrained_ckpt"] != "/tmp/pretrained.pt":
        raise RuntimeError(
            f"unexpected_pretrained_ckpt={cfg['alignment']['pretrained_ckpt']}"
        )
    if int(cfg["runtime"]["max_steps"]) != 13:
        raise RuntimeError(f"unexpected_max_steps={cfg['runtime']['max_steps']}")
    return "align_runtime_config_overrides_ok"




def check_safe_channel_values_replaces_nonfinite_metrics():
    values = torch.tensor(
        [[0.5, float("nan"), 0.25], [0.1, 0.2, float("inf")]],
        dtype=torch.float32,
    )
    safe_values, invalid_count = safe_channel_values(values, metric_name="acc")
    if invalid_count != 2:
        raise RuntimeError(f"unexpected_invalid_count={invalid_count}")
    if not torch.isfinite(safe_values).all():
        raise RuntimeError(f"safe_values_still_nonfinite={safe_values}")
    if float(safe_values[0, 1].item()) != -1.0:
        raise RuntimeError(f"unexpected_nan_fill={float(safe_values[0, 1].item())}")
    if float(safe_values[1, 2].item()) != -1.0:
        raise RuntimeError(f"unexpected_inf_fill={float(safe_values[1, 2].item())}")
    return "safe_channel_values_nonfinite_replacement_ok"


def check_stage1_runtime_step_smoke():
    device = default_device()
    cfg = build_train_cfg(patch_grid=2)
    model = build_model(cfg["model"], device)
    model.train()
    optimizer = build_train_optimizer(model, cfg)
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
        raise RuntimeError(f"stage1_nonfinite_loss={float(loss.item())}")
    loss.backward()
    optimizer.step()
    return f"stage1_loss={float(loss.item()):.6f}"



def check_stage2_runtime_step_smoke():
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
    loss = float(metrics["loss"])
    if not finite_number(loss):
        raise RuntimeError(f"stage2_nonfinite_loss={loss}")
    return f"stage2_loss={loss:.6f}"


def main():
    seed_all(50)
    recorder = Recorder("9_runtime")
    run_check(recorder, "entrypoint_modules_import", check_entrypoint_modules_import)
    run_check(
        recorder,
        "entrypoint_main_symbols_exist",
        check_entrypoint_main_symbols_exist,
    )
    run_check(
        recorder,
        "train_runtime_config_overrides",
        check_train_runtime_config_overrides,
    )
    run_check(
        recorder,
        "align_runtime_config_overrides",
        check_align_runtime_config_overrides,
    )
    run_check(
        recorder,
        "safe_channel_values_replaces_nonfinite_metrics",
        check_safe_channel_values_replaces_nonfinite_metrics,
    )
    run_check(recorder, "stage1_runtime_step_smoke", check_stage1_runtime_step_smoke)
    run_check(recorder, "stage2_runtime_step_smoke", check_stage2_runtime_step_smoke)
    recorder.finalize()


if __name__ == "__main__":
    main()
