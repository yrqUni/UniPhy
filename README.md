# UniPhy

UniPhy is a continuous spatiotemporal model for probabilistic forecasting of physical fields, with demonstrated application to global weather prediction on ERA5 data.

Most existing data-driven forecasters commit to a fixed integration step at training time, which limits their flexibility at inference and prevents principled uncertainty quantification across scales. UniPhy addresses both limitations through a complex-valued state-space formulation whose transition operators are derived from the exact matrix exponential, making the dynamics well-defined at any timestep with a single set of parameters.

## Method

Temporal evolution is carried out in a learned biorthogonal spectral basis, where each latent dimension follows an independent complex-valued linear ODE. Global information is aggregated and injected through a flux memory module whose recurrence is computed in $O(\log T)$ depth via a Triton parallel-scan kernel. Stochastic forcing is incorporated directly into the state update, with noise magnitude that scales consistently with the integration step, recovering deterministic dynamics in the zero-noise limit. Spatial structure is captured by a multi-scale depthwise mixing operator applied at every block. At inference, independent ensemble members are obtained by conditioning on distinct noise realizations, with no architectural modification.

## Training

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align
```

## Verification

See [`Check/`](./Check/) for the full operator-level (T01–T17) and system-level (S01–S05) test suite.

```bash
python Check/tests/run_all.py --tests T S --log-dir /tmp/uniphy_check_logs
```

Latest run: **22 / 22 PASS** · PyTorch 2.9.1+cu128 · CUDA 12.8

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title  = {{UniPhy}: Continuous-Time Probabilistic Weather Forecasting},
  year   = {2026},
  url    = {https://github.com/yrqUni/UniPhy}
}
```
