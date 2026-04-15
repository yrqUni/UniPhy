# UniPhy

UniPhy is a continuous-time probabilistic weather forecasting model for gridded ERA5 data.

## Architecture
- metric-free patch encoder with learned positional embeddings
- stacked latent blocks with biorthogonal spectral propagation, global flux memory, and multi-scale spatial mixing
- ensemble decoder with an inlined skip connection for field reconstruction
- autoregressive rollout with variable-step `dt` inputs

## Training
```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align
```

## Verification
```bash
python -m Model.UniPhy.Check
python -m Model.UniPhy.CheckPScan
python -m tests.cat1_imports
python -m tests.cat3_loss
python -m tests.cat5_training
python -m tests.cat9_before_after
```

## Data
Place normalized ERA5 tensors under `./data/ERA5` and adjust `Exp/ERA5/train.yaml` or `Exp/ERA5/align.yaml` if needed.
