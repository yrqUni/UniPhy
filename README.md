UniPhy

Continuous-time probabilistic forecasting model for physical fields.

## Requirements

- Python 3.11+
- PyTorch with CUDA support

## Training

Stage I:

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train --data-input-dir <data_dir>
```

Stage II:

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align --data-input-dir <data_dir> --pretrained-ckpt <stage1_ckpt>
```

Stage II strict-loads the Stage I checkpoint. The model configuration must match exactly.

## Evaluation

```bash
python -m Exp.ERA5.eval_forecast --checkpoint <ckpt> --data-input-dir <data_dir>
```

## Verification

```bash
python -m Check.tests.run_all --tests T S --log-dir <log_dir> --json-out <json_path>
```

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title  = {{UniPhy}: Continuous-Time Probabilistic Weather Forecasting},
  year   = {2026},
  url    = {https://github.com/yrqUni/UniPhy}
}
```
