import importlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from Exp.ERA5.ERA5 import ERA5Dataset
from .suite_utils import (
    ROOT,
    Recorder,
    build_model,
    build_train_cfg,
    default_device,
    list_project_modules,
    run_check,
    seed_all,
)


def check_module_imports():
    modules = list_project_modules(ROOT)
    imported = []
    for name in modules:
        importlib.import_module(name)
        imported.append(name)
    return f"imported_modules={len(imported)}"


def check_public_model_instantiation():
    device = default_device()
    cfg = build_train_cfg(patch_grid=2)
    model = build_model(cfg["model"], device)
    param_count = sum(param.numel() for param in model.parameters())
    if param_count <= 0:
        raise RuntimeError(f"invalid_parameter_count={param_count}")
    return f"device={device} parameters={param_count}"


def check_dataset_instantiation_and_sampling():
    cfg = build_train_cfg(patch_grid=2)
    channels = int(cfg["model"]["in_channels"])
    height = int(cfg["model"]["img_height"])
    width = int(cfg["model"]["img_width"])
    dt_ref = float(cfg["model"]["dt_ref"])
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        year_dir = root / "2000"
        year_dir.mkdir(parents=True, exist_ok=True)
        frames = np.stack(
            [
                np.full(
                    (channels, height, width),
                    fill_value=float(index),
                    dtype=np.float32,
                )
                for index in range(8)
            ],
            axis=0,
        )
        np.save(year_dir / "01.npy", frames)
        dataset = ERA5Dataset(
            input_dir=str(root),
            year_range=[2000, 2000],
            window_size=5,
            sample_k=5,
            look_ahead=1,
            is_train=False,
            dt_ref=dt_ref,
            sample_offsets_hours=[0.0, 3.0, 9.0, 18.0, 24.0],
        )
        if len(dataset) <= 0:
            raise RuntimeError(f"invalid_dataset_length={len(dataset)}")
        data, dt = dataset[0]
        expected_shape = (5, channels, height, width)
        if tuple(data.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected_data_shape={tuple(data.shape)} expected={expected_shape}"
            )
        expected_dt = torch.tensor(
            [dt_ref, 3.0, 6.0, 9.0, 6.0],
            dtype=torch.float32,
        )
        if not torch.allclose(dt, expected_dt):
            raise RuntimeError(f"unexpected_dt={dt.tolist()}")
        means = data.mean(dim=(1, 2, 3))
        expected_means = torch.tensor(
            [0.0, 0.5, 1.5, 3.0, 4.0],
            dtype=torch.float32,
        )
        if not torch.allclose(means, expected_means):
            raise RuntimeError(f"unexpected_interpolation={means.tolist()}")
    return f"dataset_len={len(dataset)} sample_shape={tuple(data.shape)}"


def main():
    seed_all(42)
    recorder = Recorder("1_imports")
    run_check(recorder, "module_imports", check_module_imports)
    run_check(recorder, "public_model_instantiation", check_public_model_instantiation)
    run_check(
        recorder,
        "dataset_instantiation_and_sampling",
        check_dataset_instantiation_and_sampling,
    )
    recorder.finalize()


if __name__ == "__main__":
    main()
