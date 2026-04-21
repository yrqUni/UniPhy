# UniPhy

Continuous-Time Probabilistic Forecasting of Physical Fields

UniPhy is a spatiotemporal foundation model for probabilistic forecasting of physical fields, demonstrated on global weather prediction with ERA5 reanalysis data.

Most data-driven forecasters commit to a fixed integration step at training time, which limits flexibility at inference and prevents principled uncertainty quantification across scales. UniPhy resolves both limitations through a complex-valued state-space formulation whose transition operators are derived from the exact matrix exponential, making the dynamics well-defined at any timestep with a single set of learned parameters.

## Key Ideas

**Continuous-time dynamics.** Temporal evolution is carried out in a learned bounded spectral basis, where each latent dimension follows an independent complex-valued linear ODE. Because the state update is parameterized through the matrix exponential, the model can be evaluated at arbitrary sub-step or super-step intervals without retraining.

**Efficient global context.** A flux memory module aggregates global information and injects it into the local state via a parallel-scan recurrence computed in O(log T) depth.

**Native uncertainty quantification.** Stochastic forcing is incorporated directly into the state update, with noise magnitude that scales consistently with the integration step. Independent ensemble members are obtained by conditioning on distinct noise realizations, requiring no post-hoc calibration.

**Multi-scale spatial structure.** A depthwise multi-scale mixing operator couples global and local spatial information at every block.

## Architecture Overview

```
Input x [B, T, C, H, W]

Encoder
  patch projection, conv stem, positional embedding, complex latent z

UniPhy Blocks x depth
  spatial mixer
  basis encode, flux tracker (parallel scan), spatial gating
  temporal propagator (parallel scan)
  basis decode, complex FFN residual

Decoder
  skip gate, conv residual blocks, upsampling, output x_hat [B, T, C, H, W]
```

## Normalized Time Formulation

UniPhy normalizes all physical timesteps by a reference duration `dt_ref`. The learned eigenvalues have units of per reference step, not per hour. The discretization is:

$$
h_t = \exp\!\left(\lambda\,\frac{dt}{dt_{\mathrm{ref}}}\right) h_{t-1}
    + \frac{dt}{dt_{\mathrm{ref}}}\;
      \phi_1\!\left(\lambda\,\frac{dt}{dt_{\mathrm{ref}}}\right) u_t,
\qquad
\phi_1(z) = \frac{e^z - 1}{z}.
$$

The physically meaningful quantity is the ratio $\tau = dt / dt_{\mathrm{ref}}$, not raw `dt` alone.

A model trained on 6-hour ERA5 data (`dt_ref = 6`) can produce 1-hour forecasts by simply setting the ratio to 1/6, with no retraining needed. To train on a dataset with a different native cadence, change `dt_ref` to match. Sub-step curricula during training are naturally consistent because they reduce the ratio inside the exponential and forcing term.

### Example

A learned mode with $\lambda = -0.6 + 0.2i$ and `dt_ref = 6`:

| Forecast step | Normalized ratio | Decay factor |
|---|---|---|
| 6 h (training cadence) | 1 | $\exp(-0.6 + 0.2i)$ |
| 1 h (finer inference) | 1/6 | $\exp\!\left((-0.6 + 0.2i)/6\right)$ |

Nothing about the learned eigenvalue changes. Only the normalized step does.

## Structurally Consistent Basis Transform

The basis transform constructs a bounded perturbation of a stored DFT basis and derives both the forward and inverse matrices from the same perturbation via a truncated Neumann series. This enforces inverse consistency structurally rather than through a separate learned inverse, improving numerical stability and reducing parameter count.

## Two-Stage Training

| | Stage I | Stage II |
|---|---|---|
| Purpose | Warm up encoder/decoder and dynamics under short-horizon supervision | Align longer-horizon dynamics under autoregressive rollout |
| Training mode | Single-step, short windows | Multi-step rollout with sub-step curriculum |

Weight transfer between stages uses name-and-shape matching. For full dynamics transfer, both stages should share the same embedding dimension.

## Getting Started

```bash
# Stage I: single-step pretraining
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train

# Stage II: multi-step alignment
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align
```

## Verification

A self-contained test suite is provided in `Check/`:

```bash
python Check/tests/run_all.py --tests T S --log-dir /tmp/uniphy_check_logs --json-out /tmp/uniphy_check_logs/results.json
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
