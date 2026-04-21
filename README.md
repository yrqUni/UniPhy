# UniPhy

UniPhy is a continuous spatiotemporal model for probabilistic forecasting of physical fields, with demonstrated application to global weather prediction on ERA5 data.

Most existing data-driven forecasters commit to a fixed integration step at training time, which limits their flexibility at inference and prevents principled uncertainty quantification across scales. UniPhy addresses both limitations through a complex-valued state-space formulation whose transition operators are derived from the exact matrix exponential, making the dynamics well-defined at any timestep with a single set of parameters.

## Method

Temporal evolution is carried out in a learned bounded spectral basis, where each latent dimension follows an independent complex-valued linear ODE in normalized time. Global information is aggregated and injected through a flux memory module whose recurrence is computed in $O(\log T)$ depth via a Triton parallel-scan kernel. Stochastic forcing is incorporated directly into the state update, with noise magnitude that scales consistently with the normalized integration step. Spatial structure is captured by a multi-scale depthwise mixing operator applied at every block. At inference, independent ensemble members are obtained by conditioning on distinct externally supplied noise realizations.

## Architecture

```text
input x[B,T,C,H,W]
  -> UniPhyEncoder
       replicate-pad to patch multiple
       -> patch projection conv
       -> conv stem + SiLU
       -> learned positional embedding
       -> complex latent z[B,T,D,H',W']

  -> UniPhyBlock x depth
       spatial mixer
       -> basis encode W
       -> flux spatial pool
       -> global flux tracker (PScan)
       -> spatial gate modulator
       -> temporal propagator (PScan)
       -> basis decode W_inv
       -> complex FFN residual

  -> decoder skip gate against encoder latent
  -> UniPhyEnsembleDecoder
       latent real/imag projection
       -> conv residual blocks
       -> pixel-shuffle-style upsampling
       -> smoothing conv
  -> output x_hat[B,T,C,H,W]
```

Within each block, the temporal state update is evaluated in the encoded basis, while the spatial mixer, pooling, and gating paths keep global and local information coupled.

## Dimensional analysis of dt_ref

This is the single most important usage note for UniPhy.

UniPhy does **not** treat `lambda` as a per-hour continuous-time eigenvalue. Internally, timesteps are normalized by `dt_ref`, and the learned `lambda` therefore has units of **per reference step**, not per hour.

With the default ERA5 configuration,

- `dt_ref = 6.0`
- one internal unit of time corresponds to **6 physical hours**

The implemented discretization is

$$
h_t = \exp\!\left(\lambda \frac{dt}{dt_{\mathrm{ref}}}\right) h_{t-1}
    + \frac{dt}{dt_{\mathrm{ref}}}
      \phi_1\!\left(\lambda \frac{dt}{dt_{\mathrm{ref}}}\right) u_t,
\qquad
\phi_1(z) = \frac{e^z - 1}{z}.
$$

That means the physically meaningful quantity is the ratio

$$
\tau = \frac{dt}{dt_{\mathrm{ref}}},
$$

not raw `dt` alone.

### Worked example: default 6-hour ERA5

Suppose a learned latent mode has

$$
\lambda = -0.6 + 0.2i
$$

and you evaluate one forecast step at the default ERA5 cadence, `dt = 6` hours.

Then

$$
\tau = \frac{6}{6} = 1,
$$

so the decay factor is

$$
\exp(\lambda \tau) = \exp(-0.6 + 0.2i).
$$

Now evaluate a finer substep at `dt = 1` hour **without retraining**.

Then

$$
\tau = \frac{1}{6},
$$

so the same latent mode advances by

$$
\exp\!\left(\lambda \frac{1}{6}\right).
$$

Nothing about `lambda` itself changes; only the normalized step does. This is exactly why the model can evaluate at non-training timesteps.

### Worked example: switching datasets from 6-hour to 1-hour cadence

Suppose you move from 6-hour ERA5 inputs to a truly 1-hour dataset.

What should change?

- **Change** `dt_ref` from `6.0` to `1.0` in the training/evaluation configuration.
- **Do not** reinterpret an old 6-hour-trained `lambda` as if it were now per hour.

Why?

Under the default 6-hour setup, the internal normalized step for a 6-hour jump is

$$
\tau = \frac{6}{6} = 1.
$$

Under a 1-hour dataset with `dt_ref = 1.0`, a 1-hour jump should also correspond to

$$
\tau = \frac{1}{1} = 1.
$$

So the model family preserves the same *mathematical role* for a “one reference-step jump,” but the physical duration of that reference step has changed.

Concretely:

- If you train on 6-hour data and later run on 1-hour data **without** changing `dt_ref`, then a 1-hour step is interpreted as
  $$\tau = \frac{1}{6},$$
  which is a fractional substep of the 6-hour dynamics.
- If you train on 1-hour data with `dt_ref = 1.0`, then a 1-hour step is interpreted as
  $$\tau = 1,$$
  which is a full reference-step transition for that new dataset.

So when switching cadence, what changes is the configuration value `dt_ref` and the training data; what does **not** change is the implementation rule that everything flows through `dt / dt_ref`.

### Practical consequences

1. `lambda` is **per-`dt_ref` normalized**, not per hour.
2. To train on a dataset with a different base cadence, set `dt_ref` to that cadence in hours.
3. The `sub_steps` curriculum in Stage II is mathematically consistent because it divides the physical step into smaller normalized steps, i.e. it reduces `dt / dt_ref` inside the exponential and forcing term.
4. If you compare timescales across runs with different `dt_ref`, convert back to physical units explicitly instead of comparing raw `lambda` values directly.

## Basis transform contract

After the audit fix, `ComplexSVDTransform` no longer stores an independent learned inverse path. Instead, it constructs a bounded perturbation of the stored DFT basis and derives the inverse from the same perturbation using a truncated Neumann series.

Let

$$
\alpha = \sigma(\alpha_{\mathrm{logit}}),
\qquad
\beta = 1 - \alpha,
\qquad
s = \mathrm{dim}^{-1/2},
\qquad
a = 1 + 10^{-3}\alpha.
$$

The bounded perturbation is formed from raw trainable weights via `tanh` and a spectral cap $\rho_{\max}$:

$$
\Delta = \beta\,\Phi(W_{\mathrm{re,raw}}, W_{\mathrm{im,raw}})\,s,
\qquad \|\Delta\|_2 \le \rho_{\max}.
$$

Then

$$
L = I + \Delta,
\qquad
L^{-1}_{(N)} = \sum_{k=0}^{N} (-\Delta)^k,
$$

and the matrices returned by the implementation are

$$
W = a\,(D_{\mathrm{FT}} L),
\qquad
W^{-1} = \frac{L^{-1}_{(N)} D_{\mathrm{FT}}^{-1}}{a}.
$$

This means:

- `W_inv` is no longer an independently learned object.
- inverse consistency is enforced structurally by deriving both matrices from the same perturbation.
- the truncation accuracy depends on the spectral cap and the chosen number of Neumann terms.

## Magic numbers reference

| parameter | initial value | location | physical meaning |
|---|---:|---|---|
| `dt_ref` | `6.0` | `Exp/ERA5/train.yaml`, `Exp/ERA5/align.yaml` | Reference timestep in hours; raw `dt` is normalized by this value |
| `alpha_logit` | `2.0` | `Model/UniPhy/UniPhyOps.py` | Initial bias toward the stored DFT basis |
| `alpha_scale` factor | `1e-3` | `Model/UniPhy/UniPhyOps.py` | Small amplitude rescaling of the basis matrices |
| Neumann truncation order | implementation-dependent (`neumann_terms`) | `Model/UniPhy/UniPhyOps.py` | Controls inverse approximation accuracy |
| spatial mixer `output_scale` | `0.1` | `Model/UniPhy/UniPhyOps.py` | Initial residual strength of the multi-scale spatial mixer |
| flux `gate_min` | `0.01` | `Model/UniPhy/UniPhyOps.py` | Lower bound on the global/local gate |
| flux `gate_max` | `0.99` | `Model/UniPhy/UniPhyOps.py` | Upper bound on the global/local gate |
| flux decay real init | `randn * 0.1 - 1.0` | `Model/UniPhy/UniPhyOps.py` | Initializes global-flux damping in a dissipative regime |
| flux decay imag init | `randn * 0.1` | `Model/UniPhy/UniPhyOps.py` | Initializes global-flux oscillatory frequency |
| temporal `lam_re` raw init | `randn * 0.01` | `Model/UniPhy/UniPhyOps.py` | Raw damping parameter before `-softplus(...)` |
| temporal `lam_im` init | `randn * 0.1` | `Model/UniPhy/UniPhyOps.py` | Initial oscillatory frequency per latent mode |
| `init_noise_scale` / `base_noise` | `1e-4` | `Exp/ERA5/train.yaml`, `Model/UniPhy/UniPhyOps.py` | Base stochastic forcing amplitude before timestep scaling |
| uncertainty multiplier | `2.0` | `Model/UniPhy/UniPhyOps.py` | Multiplier applied to the learned uncertainty envelope |
| FFN `centering_scale` | `0.5` | `Model/UniPhy/UniPhyFFN.py` | Strength of spatial-mean subtraction in the FFN branch |
| FFN `output_scale` | `1.0` | `Model/UniPhy/UniPhyFFN.py` | Initial amplitude of the FFN residual branch |
| layer-norm epsilon | `1e-6` | `Model/UniPhy/UniPhyFFN.py` | Numerical stabilization constant for normalization |
| encoder positional embedding std | `0.02` | `Model/UniPhy/UniPhyIO.py` | Initialization scale for patch-grid positional embeddings |
| PScan minimum block size | `16` | `Model/UniPhy/PScan.py` | Lower bound on Triton scan block width before power-of-two expansion |
| Stage I basis regularization weight | `0.01` | `Exp/ERA5/train.yaml`, `Exp/ERA5/train.py` | Soft inverse-consistency penalty used during Stage I training |

## Two-stage training (Stage I → Stage II)

UniPhy currently uses a two-stage training layout.

### Stage I

- `embed_dim = 128`
- single-step training on short windows
- objective: warm up the encoder/decoder surface and the smaller dynamic stack under short-horizon supervision

### Stage II

- `embed_dim = 512`
- multi-step rollout training with the `sub_steps` curriculum
- objective: align longer-horizon dynamics under autoregressive rollout

### Handoff semantics

`load_matching_pretrained_weights()` only transfers tensors whose names and shapes match exactly. Because Stage I and Stage II use different embedding widths, most block-internal tensors do **not** transfer. In practice, this means Stage II is a partial warm start of compatible outer layers and scalar parameters, not a full continuation of Stage I dynamics.

If a user wants true end-to-end weight transfer of the dynamics, Stage II must use the same `embed_dim` as Stage I.

## Training

```bash
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.train
torchrun --nproc_per_node=<num_gpus> -m Exp.ERA5.align
```

## Verification

See [`Check/`](./Check/) for the rev-2 verification suite and the measured audit results on `math-fixes`.

```bash
python Check/tests/T17_numerical_regression.py --regenerate
python Check/tests/run_all.py --tests T S --log-dir /tmp/uniphy_check_logs --json-out /tmp/uniphy_check_logs/results.json
```

## Known issues / open questions

- The duplicated `Model/UniPhy/Check/dt_check/` tree still exists and should either be archived explicitly or removed in a future cleanup.
- Stage I → Stage II transfer is intentionally partial because of the `embed_dim` mismatch, but that training design should be confirmed by the researcher as a long-term choice.
- `align_step()` still uses repeated CPU↔GPU copies as a memory tradeoff. This audit did not benchmark alternatives.
- Historical no-complex ablation artifacts referenced by the mission text were not present locally, so that ablation question remains unresolved.

## Citation

```bibtex
@misc{uniphy2026,
  author = {Ruiqing Yan},
  title  = {{UniPhy}: Continuous-Time Probabilistic Weather Forecasting},
  year   = {2026},
  url    = {https://github.com/yrqUni/UniPhy}
}
```
