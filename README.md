# UniPhy

UniPhy is a continuous spatiotemporal model built for probabilistic forecasting of physical systems, with a primary application to global weather prediction on ERA5 data.

The core design centers on a complex-valued state-space formulation with exact continuous-time discretization, enabling the model to operate at **arbitrary timesteps** with a single set of weights. Temporal dynamics are governed by learnable spectral bases (biorthogonal by construction), global flux memory propagated via parallel scan, and physics-informed stochastic forcing whose magnitude scales consistently with `dt`.

## Architecture

- **Encoder** — metric-free patch embedding with learned positional encodings
- **Latent blocks** — spectral temporal propagation, multi-scale spatial mixing, and global flux tracking operating jointly in the complex domain
- **Decoder** — pixel-shuffle reconstruction with an adaptive skip connection from the encoder

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
