import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from Check.utils import max_diff, write_result
from Exp.ERA5.ERA5 import ERA5Dataset

TEST_ID = "T10"


def run():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        year_dir = root / "2000"
        year_dir.mkdir(parents=True, exist_ok=True)
        frames = np.arange(8 * 2 * 2 * 1, dtype=np.float32).reshape(8, 2, 2, 1)
        np.save(year_dir / "01.npy", frames)

        dataset = ERA5Dataset(
            input_dir=str(root),
            year_range=[2000, 2000],
            window_size=4,
            sample_k=4,
            look_ahead=0,
            is_train=False,
            dt_ref=6.0,
            sample_offsets_hours=[0.0, 3.0, 12.0, 18.0],
        )
        data, dt = dataset[1]

        expected_a = torch.from_numpy(frames[1].copy())
        expected_b = (
            torch.from_numpy(frames[1].copy()) + torch.from_numpy(frames[2].copy())
        ) / 2.0
        expected_c = torch.from_numpy(frames[3].copy())
        expected_d = torch.from_numpy(frames[4].copy())
        err_interp = max(
            max_diff(data[0], expected_a),
            max_diff(data[1], expected_b),
            max_diff(data[2], expected_c),
            max_diff(data[3], expected_d),
        )
        dt_expected = torch.tensor([6.0, 3.0, 9.0, 6.0])
        err_dt = max_diff(dt, dt_expected)

        dataset_eval = ERA5Dataset(
            input_dir=str(root),
            year_range=[2000, 2000],
            window_size=4,
            sample_k=3,
            look_ahead=0,
            is_train=False,
            dt_ref=6.0,
            sample_offsets_hours=None,
        )
        data_eval, dt_eval = dataset_eval[2]
        expected_eval = torch.from_numpy(frames[2:5].copy())
        err_eval = max_diff(data_eval, expected_eval)
        err_eval_dt = max_diff(dt_eval, torch.tensor([6.0, 6.0, 6.0]))

        dataset_train = ERA5Dataset(
            input_dir=str(root),
            year_range=[2000, 2000],
            window_size=4,
            sample_k=3,
            look_ahead=0,
            is_train=True,
            dt_ref=6.0,
            sample_offsets_hours=None,
        )
        dataset_train._sample_offsets = lambda: ([1, 2, 3], True)
        data_train, dt_train = dataset_train[2]
        expected_train = torch.from_numpy(frames[3:6].copy())
        err_train = max_diff(data_train, expected_train)
        err_train_dt = max_diff(dt_train, torch.tensor([6.0, 6.0, 6.0]))

    max_err = max(err_interp, err_dt, err_eval, err_eval_dt, err_train, err_train_dt)
    passed = max_err < 1e-6
    detail = (
        f"err_interp={err_interp:.2e} err_dt={err_dt:.2e} "
        f"err_eval={err_eval:.2e} err_eval_dt={err_eval_dt:.2e} "
        f"err_train={err_train:.2e} err_train_dt={err_train_dt:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
