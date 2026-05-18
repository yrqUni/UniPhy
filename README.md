<div align="center">

# UniPhy

**Continuous time probabilistic forecasting for physical fields**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.1-ee4c2c.svg)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/backend-PyTorch%20%2B%20optional%20Triton-success.svg)](#requirements)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green.svg)](LICENSE)

</div>

UniPhy is a PyTorch research framework for probabilistic forecasting of
spatio temporal physical fields. It combines continuous time dissipative latent
dynamics, stochastic ensemble generation, adaptive spatial mixing, residual free
latent readout, and efficient parallel scan execution.

The project is designed for transparent scientific use. Core numerical
contracts are covered by a dedicated verification suite, and the ablation
protocol is organized as a controlled single factor study.

## Features

| Capability | Summary |
|:--|:--|
| Continuous time dynamics | Hidden states evolve under learned dissipative dynamics and accept irregular time intervals directly. |
| Exact dissipative transition | The temporal propagator applies the closed form physical time update for each observed interval. |
| Residual free readout | Forecast states are decoded directly from dissipative latent dynamics, avoiding persistent residual correction during free rollout. |
| Probabilistic ensembles | Stochastic latent forcing produces calibrated ensemble members from a single learned diffusion model. |
| Adaptive spatial mixing | Local, regional, and large scale branches are combined through learned scale weights. |
| Parallel scan | PScan dispatches to Triton on supported CUDA shapes and otherwise uses a Torch Tree implementation. |
| Verification suite | Fifteen numerical tests cover discretization, stochastic scaling, scan equivalence, rollout invariants, and regression stability. |
| Ablation protocol | Controlled UniPhy variants and fixed interval SwinTrans and ConvLSTM baselines support reproducible model comparisons. |

## Installation

```bash
pip install torch numpy pyyaml rich
```

For CUDA deployments, install the PyTorch build that matches the target CUDA
runtime. Triton is optional. UniPhy automatically falls back to the Torch Tree
parallel scan when Triton is unavailable or when a scan shape is not supported.

## Quick Start

```python
import torch
from Model.UniPhy.ModelUniPhy import UniPhyModel

model = UniPhyModel(
    in_channels=30,
    out_channels=30,
    embed_dim=512,
    expand=4,
    depth=8,
    patch_size=(7, 15),
    img_height=721,
    img_width=1440,
    dt_ref=6.0,
    init_noise_scale=0.05,
)

x = torch.randn(2, 4, 30, 721, 1440)
dt = torch.tensor([[6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 12.0, 6.0]])

y_det = model(x, dt)
y_sto = model(x, dt, z=True)

rollout = model.forward_rollout(
    x_context=x[:, :2],
    dt_context=dt[:, :2],
    dt_list=[torch.tensor([6.0, 6.0])] * 16,
    z_context=True,
    z_rollout=True,
)
```

## Project Layout

| Path | Purpose |
|:--|:--|
| `Model/UniPhy/` | Model architecture, temporal operators, spatial mixers, IO layers, and PScan. |
| `Exp/ERA5/` | ERA5 data loading, training, alignment, and forecast evaluation. |
| `Exp/Ablation/` | Controlled ablation protocol, variant builders, training, evaluation, and aggregation. |
| `Check/` | Numerical verification suite and regression checks. |

## Data

UniPhy expects daily ERA5 NumPy shards with shape `[T, C, H, W]`.

```text
<data_dir>/<year>/<yearMMDD>.npy
```

The default ERA5 configuration uses 30 normalized channels, including surface
variables and pressure level variables.

## Training

```bash
torchrun --nproc_per_node=8 -m Exp.ERA5.train \
    --data-input-dir /data/ERA5 \
    --train-year-range 2000,2016
```

Rollout alignment can be run after the supervised stage.

```bash
torchrun --nproc_per_node=8 -m Exp.ERA5.align \
    --data-input-dir /data/ERA5 \
    --pretrained-ckpt <stage1_ckpt>
```

## Evaluation

```bash
python -m Exp.ERA5.eval_forecast \
    --checkpoint <ckpt> \
    --data-input-dir <eval_dir> \
    --climatology-dir <train_dir> \
    --climatology-year-range 2000,2016 \
    --lead-times 6,24,72,120,240
```

Use a fixed climatology for all reported ACC comparisons.

## Verification

```bash
python -m Check.tests.run_all
```

The verification suite checks analytic transition operators, PScan equivalence
against serial references, stochastic scaling, rollout invariance, basis
invertibility, temporal interpolation, CRPS consistency, and numerical
regression stability.

## Ablations

The ablation suite evaluates the full model, controlled single factor variants,
and fixed interval operational baselines. It supports paired seeds,
reproducible run manifests, RMSE, ACC, CRPS, and publication ready CSV, LaTeX,
and JSON summaries.

The three year controlled protocol identifies the residual free dissipative
baseline as the strongest model across standard, regular 6 h, regular 12 h,
irregular short, and irregular medium evaluation grids.

SwinTrans and ConvLSTM are provided as fixed interval operational baselines.
They are evaluated on regular 6 h rollouts only, including direct UniPhy
prediction to 12 h and 24 h and recursive fixed step prediction for all models.

See [Exp/Ablation](Exp/Ablation/README.md) for the protocol.

## Acknowledgements

The development of UniPhy was supported by the following organizations, in no
particular order.

* **中国科学院计算机网络信息中心 国家超级计算（中国科学院）中心**
  [Computer Network Information Center, Chinese Academy of Sciences, National Supercomputing Center, Chinese Academy of Sciences](https://cnic.cas.cn/jgsz/kyywbm/cjjszxyxyyyfws/)

* **北京积算科技有限公司**
  [Beijing iCompify Technology Co., Ltd.](https://www.icompify.com/)

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title  = {{UniPhy}: Continuous Time Probabilistic Forecasting for Physical Fields},
  year   = {2026},
  url    = {https://github.com/yrqUni/UniPhy}
}
```

## License

UniPhy is distributed under the Creative Commons Attribution NonCommercial 4.0
International License. Academic and other non commercial use is permitted with
attribution to Ruiqing Yan. Commercial and industrial use requires a separate
written commercial license. See [LICENSE](LICENSE).
