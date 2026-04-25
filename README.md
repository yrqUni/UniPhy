# UniPhy

UniPhy is a continuous-time probabilistic forecasting model for
physical fields, designed to model irregular temporal dynamics while
preserving stable rollout behavior, calibrated ensemble objectives,
and numerically verifiable operator contracts.

## Highlights

- Continuous-time latent dynamics with explicit transition operators
  and stochastic forcing
- Probabilistic ensemble training for physical-field forecasting
- ERA5 experiment pipeline for training, alignment, and forecast
  evaluation
- Numerical verification suite covering transition operators, scan
  recurrences, rollout consistency, timestep semantics, CRPS
  consistency, data interpolation, and regression baselines

## Repository structure

- `Model/UniPhy/` — model architecture, transition operators, basis
  transforms, encoder/decoder, and scan primitive
- `Exp/ERA5/` — ERA5 training, alignment, evaluation, runtime
  configuration, and dataset loading
- `Check/` — numerical verification suite and result logs

## Requirements

- Python 3.11+
- NVIDIA GPU environment with CUDA support
- PyTorch and project dependencies listed in `requirements.txt`

A Conda environment example is provided in `environment.yaml`.

## Data layout

ERA5 experiments expect NumPy files organized by year:

```text
<data_dir>/
  2000/
    01.npy
    02.npy
    ...
  2001/
    01.npy
    ...
```

Each `.npy` file is loaded as a time-major tensor sequence by
`Exp/ERA5/ERA5.py`.

## Training

Stage I supervised training:

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train --data-input-dir <data_dir>
```

Stage II alignment training:

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align \
  --data-input-dir <data_dir> \
  --pretrained-ckpt <stage1_ckpt>
```

Stage II loads the Stage I checkpoint with strict parameter matching,
so model architecture fields must remain consistent across stages.

## Evaluation

```bash
python -m Exp.ERA5.eval_forecast --checkpoint <ckpt> --data-input-dir <data_dir>
```

The evaluation path reports RMSE, ACC, and CRPS over configurable lead
times.

## Numerical verification

```bash
python -m Check.tests.run_all --log-dir <log_dir> --json-out <json_path>
```

The suite is organized as a consecutive `T01`–`T13` numerical test
set. Coverage includes:

- transition-operator discretization
- scan forward/backward correctness
- flux-state recurrence equivalence
- block and rollout consistency
- basis invertibility
- stochastic scaling
- timestep validation
- CRPS consistency across train/align/eval
- ERA5 interpolation and `dt` semantics
- gate-range constraints
- encoder/decoder and FFN shape/dtype contracts
- numerical regression baseline

See `Check/README.md` for the latest recorded run results.

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title  = {{UniPhy}: Continuous-Time Probabilistic Weather Forecasting},
  year   = {2026},
  url    = {https://github.com/yrqUni/UniPhy}
}
```
