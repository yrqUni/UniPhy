<div align="center">

# UniPhy

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.1-ee4c2c.svg)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/backend-PyTorch%20%2B%20optional%20Triton-success.svg)](#installation)
[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green.svg)](LICENSE)

</div>

UniPhy is a PyTorch framework for medium range forecasting of physical fields. The active model uses a deterministic recurrent latent core, real valued convolutional encoding and decoding, multi scale spatial mixing, and a Tree parallel PScan backend with optional Triton acceleration.

## Highlights

| Capability | Summary |
|:--|:--|
| Deterministic UniPhy core | The active UniPhy model uses a single recurrent latent structure with explicit `dt_ref` rollout composition. |
| Direct and recursive agreement | Integer multiple leads are evaluated through the same internal rollout contract. |
| Tree parallel scan | Torch provides a parallel tree scan fallback when Triton is unavailable. |
| Fixed interval baselines | SwinTrans and ConvLSTM are provided as matched fixed interval comparison models. |
| Numerical verification | `Check` contains self contained mathematical tests only. |
| Controlled ablation | The ablation workflow reports matched results under a three year protocol. |

## Installation

```bash
pip install torch numpy pyyaml rich
```

For CUDA environments, install the PyTorch build that matches the target CUDA runtime. Triton is optional.

## Verification

```bash
python -m Check.tests.run_all
```

The verification suite checks PScan tree scan consistency, forward and backward agreement, deterministic UniPhy rollout equivalence, zero time identity, dt validation, gradient flow, and CRPS consistency.

## Project Layout

| Path | Purpose |
|:--|:--|
| `Model/UniPhy/` | active UniPhy model and parallel scan backend |
| `Model/SwinTrans/` | fixed interval Swin Transformer baseline |
| `Model/ConvLSTM/` | fixed interval ConvLSTM baseline |
| `Exp/ERA5/` | ERA5 training and evaluation entry points |
| `Exp/Ablation/` | controlled comparison, evaluation, and summary scripts |
| `Check/` | self contained numerical verification suite |

## Training And Evaluation

```bash
torchrun --nproc_per_node=8 -m Exp.ERA5.train \
    --data-input-dir /nfs/ERA5_data/data_norm \
    --train-year-range 2000,2001
```

```bash
python -m Exp.ERA5.eval_forecast \
    --checkpoint <ckpt> \
    --data-input-dir /nfs/ERA5_data/data_norm \
    --climatology-dir /nfs/ERA5_data/data_norm \
    --climatology-year-range 2000,2001 \
    --lead-times 6,24,72,120,240
```

## Ablation Protocol

See [Exp/Ablation](Exp/Ablation/README.md) for the formal protocol, fixed interval comparison, and time sensitivity evaluation.

## Acknowledgements

The development of UniPhy was supported by the following organizations, in no particular order.

- **中国科学院计算机网络信息中心 – 国家超级计算（中国科学院）中心**  
  [Computer Network Information Center, Chinese Academy of Sciences – National Supercomputing Center (Chinese Academy of Sciences)](https://cnic.cas.cn/jgsz/kyywbm/cjjszxyxyyyfws/)

- **北京积算科技有限公司**  
  [Beijing iCompify Technology Co., Ltd.](https://www.icompify.com/)

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title = {{UniPhy}: Medium Range Forecasting for Physical Fields},
  year = {2026},
  url = {https://github.com/yrqUni/UniPhy}
}
```

## License

UniPhy is licensed under Creative Commons Attribution NonCommercial 4.0 International. Academic and other non commercial use is permitted with attribution to Ruiqing Yan. Commercial or industrial use requires a separate written license. The restriction applies to this repository and any derived version, branch, or release.
