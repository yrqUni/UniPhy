# UniPhy Temporal Verification Suite

## Overview

This suite verifies the temporal dynamics used by UniPhy. It covers the discretized state-space operators, basis transforms, timestep semantics, recurrent rollout behavior, gradient propagation, and the parallel-scan implementation that underpins the sequential and batched update paths.

## Mathematical Background

UniPhy evolves a continuous-time complex state-space model

\[
\frac{dh}{dt} = \lambda h + u(t), \qquad \lambda = \lambda_{\mathrm{re}} + i\lambda_{\mathrm{im}}, \qquad \lambda_{\mathrm{re}} < 0.
\]

For a step size `dt`, the exact discretization used by the operator tests is

\[
h_t = e^{\lambda \, dt} h_{t-1} + \phi_1(\lambda \, dt) \, dt \, u_t,
\qquad
\phi_1(z) = \frac{e^z - 1}{z}.
\]

For small `|z|`, UniPhy evaluates `\phi_1` with the Taylor expansion

\[
\phi_1(z) \approx 1 + \frac{z}{2} + \frac{z^2}{6},
\]

which is the stabilized branch exercised by `T01` and `T03`.

The learnable complex basis matrices are built by `ComplexSVDTransform.get_matrix()` with

\[
\alpha = \sigma(\alpha_{\mathrm{logit}}),
\qquad
\beta = 1 - \alpha,
\qquad
s = \mathrm{dim}^{-1/2}.
\]

Let

\[
\Delta = \beta \,(W_{\mathrm{re}} + i W_{\mathrm{im}}) \, s,
\qquad
L = I + \Delta,
\]

and define the truncated inverse series used in the source code as

\[
L^{-1}_{(4)} = I - \Delta + \Delta^2 - \Delta^3 + \Delta^4.
\]

With

\[
a = 1 + 10^{-3}\alpha,
\qquad
C = \beta \,(W^{-1}_{\mathrm{re}} + i W^{-1}_{\mathrm{im}}) \, s,
\]

the matrices returned by the implementation are

\[
W = a \,(D_{\mathrm{FT}} L),
\qquad
W^{-1} = \frac{(L^{-1}_{(4)} + C) D_{\mathrm{FT}}^{-1}}{a}.
\]

Here `D_FT` and `D_FT^{-1}` are the stored discrete Fourier transform basis matrices from `UniPhyOps.py`.

## Test Catalogue

### Operator-level tests

| id  | name | what is verified | tolerance |
|-----|------|------------------|-----------|
| T01 | phi1_stability | Stabilized `\phi_1` matches the float64 reference near zero | `1e-5` relative |
| T02 | ssm_discretisation | The discrete propagator matches the analytic SSM solution | `1e-5` |
| T03 | dt_zero_limit | As `dt \to 0`, decay approaches identity and forcing approaches zero | asymptotic |
| T04 | variable_dt_pscan | Parallel scan matches sequential recurrence for variable timesteps | `1e-4` |
| T05 | hprev_injection_equivalence | Initial hidden-state injection matches direct recurrence construction | `1e-5` |
| T06 | flux_prev_compensation | Flux carry-over compensation matches explicit recurrence | `1e-4` |
| T07 | cumprod_decay_purity | The decay operator is separated cleanly from the forcing term | `1e-6` |
| T08 | rollout_dt_alignment | Context-state timestep indexing matches the stepped update path | `1e-4` |
| T09 | forward_vs_step_single | `forward()` and `forward_step()` agree for a single step | `1e-4` |
| T10 | forward_vs_rollout_multistep | Autoregressive rollout matches chained single-step updates | `1e-4` |
| T11 | basis_encode_decode_identity | Basis encode/decode is numerically close to identity | `1e-4` |
| T12 | basis_biorthogonality | The learned basis pair remains numerically biorthogonal | recorded |
| T13 | sde_scale_physics | Diffusion scale varies monotonically with timestep and damping | assertions |
| T14 | gradient_flow | All temporal parameters receive non-zero gradients | none |
| T15 | dt_zero_mask | Zero-timestep outputs preserve the input exactly | `1e-6` |
| T16 | negative_dt_rejection | Negative timesteps are rejected with `ValueError` | all cases |
| T17 | numerical_regression | Forward outputs remain stable against the materialized golden tensors | `1e-5` |

### System-level tests

| id  | name | what is verified |
|-----|------|------------------|
| S01 | parallel_serial_consistency | Parallel and serial execution agree for `forward`, `forward_rollout`, flux scan, temporal propagator scan, and spatial mixing |
| S02 | timestep_semantics | Zero-step identity, timestep scaling, rollout indexing, stride and offset rules, scalar-versus-tensor dt handling, small-eigenvalue limits, negative-dt rejection, and dt normalization shapes |
| S03 | parameter_consistency | `dt_ref`, noise scales, state dimensions, parameter naming, basis alpha shape, mixer and tracker types, encoder state, skip wiring, and fixed numerical constants |
| S04 | architecture_verification | Removal of deprecated variants, constructor signature cleanup, gradient requirements, prohibition of `linalg.solve` and `linalg.inv`, and absence of legacy helper methods |
| S05 | pscan_correctness | Parallel-scan forward and backward agreement, shape coverage, long-sequence behavior, and 4D-input semantics |

## Results

| field | value |
|-------|-------|
| commit | ac5205bb696b8457f08248498ff73ee4bcd4cc99 |
| branch | main |
| date | 2026-04-20 |
| node | 172.16.0.21 |
| device | GPU 0 |
| python | 3.12.12 |
| torch | 2.9.1+cu128 |
| cuda | 12.8 |
| run_all | 22 / 22 PASS, 0 FAIL, 0 SKIP |

### Operator-level results

| id  | name | status | max error |
|-----|------|--------|-----------|
| T01 | phi1_stability | PASS | 1.347725e-07 |
| T02 | ssm_discretisation | PASS | 5.331202e-07 |
| T03 | dt_zero_limit | PASS | 5.746428e-11 |
| T04 | variable_dt_pscan | PASS | 4.915125e-07 |
| T05 | hprev_injection_equivalence | PASS | 4.768372e-07 |
| T06 | flux_prev_compensation | PASS | 2.665601e-07 |
| T07 | cumprod_decay_purity | PASS | 0.000000e+00 |
| T08 | rollout_dt_alignment | PASS | 1.408168e-05 |
| T09 | forward_vs_step_single | PASS | 0.000000e+00 |
| T10 | forward_vs_rollout_multistep | PASS | 0.000000e+00 |
| T11 | basis_encode_decode_identity | PASS | 6.491098e-07 |
| T12 | basis_biorthogonality | PASS | 3.407730e-11 |
| T13 | sde_scale_physics | PASS | 0.000000e+00 |
| T14 | gradient_flow | PASS | 0.000000e+00 |
| T15 | dt_zero_mask | PASS | 0.000000e+00 |
| T16 | negative_dt_rejection | PASS | 0.000000e+00 |
| T17 | numerical_regression | PASS | 0.000000e+00 |

### System-level results

| id  | name | status | max error |
|-----|------|--------|-----------|
| S01 | parallel_serial_consistency | PASS | - |
| S02 | timestep_semantics | PASS | - |
| S03 | parameter_consistency | PASS | - |
| S04 | architecture_verification | PASS | - |
| S05 | pscan_correctness | PASS | - |

## Reproduction

Clone the repository and use the root-level `Check/` package:

```bash
git clone git@github.com:yrqUni/UniPhy.git
cd UniPhy
```

Run the full operator and system suite:

```bash
python Check/tests/run_all.py --tests T S --log-dir /tmp/uniphy_check_logs
```

The numerical regression check materializes `Check/golden/golden.pt` on first run and compares against it on subsequent runs.

Run a single system group through its `__main__` entry:

```bash
python Check/tests/S05_pscan_correctness.py
```

Run a single operator test directly:

```bash
python Check/tests/T11_basis_encode_decode_identity.py
```

## Environment

The reported results were collected on node `172.16.0.21` with `CUDA_VISIBLE_DEVICES=0`, Python `3.12.12`, PyTorch `2.9.1+cu128`, and CUDA `12.8` in the `tch` conda environment.
