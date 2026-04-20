# UniPhy Temporal Verification Suite

## Overview

Verification tests for the temporal dynamics of UniPhy. The suite covers numerical correctness of the state-space operators, equivalence between parallel and sequential execution, basis transform invertibility, and gradient flow through the learnable temporal parameters.

## Mathematical Background

UniPhy discretises a continuous-time state-space model:

```text
dh/dt = λh + u(t),   λ = lam_re + i·lam_im,   lam_re < 0
```

The exact discretisation at step size `dt` is:

```text
h[t] = exp(λ·dt) · h[t−1] + φ₁(λ·dt) · dt · u[t]
```

where `φ₁(z) = (exp(z) − 1) / z` is stabilised for `|z| < 1e-7` with the Taylor approximation `φ₁(z) ≈ 1 + z/2 + z²/6`.

The latent state is encoded and decoded through a learnable transform `W` with inverse `W_inv`. In `ComplexSVDTransform.get_matrix()`, with identity `I`, DFT matrix `F`, inverse DFT matrix `F⁻¹`, `α = sigmoid(alpha_logit)`, `β = 1 − α`, scale `s = dim^(-1/2)`, and `α_scale = 1 + 10^-3 α`, the matrices are computed as:

```text
P        = β · (w_re + i·w_im) · s
L        = I + P
L_inv    = I − P + P² − P³ + P⁴
C        = β · (w_inv_re + i·w_inv_im)
W        = α_scale · (F · L)
W_inv    = ((L_inv + C · s) · F⁻¹) / α_scale
```

Biorthogonality is monitored through the residual `‖W · W_inv − I‖` during verification.

## Test Suite

| id  | name                         | layer | what is verified                            | tolerance |
|-----|------------------------------|-------|---------------------------------------------|-----------|
| T01 | phi1_stability               | 1     | φ₁ Taylor accuracy vs float64               | 1e-5 rel  |
| T02 | ssm_discretisation           | 1     | O(dt²) convergence of the SSM               | 1e-5      |
| T03 | dt_zero_limit                | 1     | decay → 1 and forcing → 0 as dt → 0         | O(dt²)    |
| T04 | variable_dt_pscan            | 1     | parallel scan matches sequential recurrence | 1e-4      |
| T05 | hprev_injection_equivalence  | 2     | initial-state injection via h_prev          | 1e-5      |
| T06 | flux_prev_compensation       | 2     | flux carry-over correction                  | 1e-4      |
| T07 | cumprod_decay_purity         | 2     | decay operator has no forcing term          | 1e-6      |
| T08 | rollout_dt_alignment         | 2     | dt indexing in autoregressive rollout       | 1e-4      |
| T09 | forward_vs_step_single       | 2     | forward() matches forward_step()            | 1e-4      |
| T10 | forward_vs_rollout_multistep | 2     | rollout matches chained steps for N = 1–24  | 1e-4      |
| T11 | basis_encode_decode_identity | 2     | decode_with(encode_with(x, W), W_inv) ≈ x   | 1e-4      |
| T12 | basis_biorthogonality        | 2     | `‖W · W_inv − I‖` at initialisation         | record    |
| T13 | sde_scale_physics            | 2     | SDE noise scale is monotone in dt           | asserts   |
| T14 | gradient_flow                | 3     | every temporal parameter receives gradient  | none      |
| T15 | dt_zero_mask                 | 3     | dt = 0 preserves the input                  | 1e-6      |
| T16 | negative_dt_rejection        | 3     | dt < 0 raises ValueError                    | 4 / 4     |
| T17 | numerical_regression         | 3     | forward output matches the recorded golden  | 1e-5      |

Layer 1 — mathematical correctness of individual operators  
Layer 2 — code implements the stated mathematics  
Layer 3 — system-level contracts

## Results

| field  | value |
|--------|-------|
| commit | 524f14e9cae6ec4fb4f6a14043b4ebb5b4560ec9 |
| branch | main |
| date   | 2026-04-20 |
| node   | 172.16.0.21 |
| device | GPU 0 |
| python | 3.12.12 |
| torch  | 2.9.1+cu128 |
| cuda   | 12.8 |

**run_all: 17 / 17 PASS   0 FAIL   0 SKIP**

| id  | name                         | status | max error |
|-----|------------------------------|--------|-----------|
| T01 | phi1_stability               | PASS   | 1.347725e-07 |
| T02 | ssm_discretisation           | PASS   | 5.331202e-07 |
| T03 | dt_zero_limit                | PASS   | 5.746428e-11 |
| T04 | variable_dt_pscan            | PASS   | 4.915125e-07 |
| T05 | hprev_injection_equivalence  | PASS   | 4.768372e-07 |
| T06 | flux_prev_compensation       | PASS   | 2.665601e-07 |
| T07 | cumprod_decay_purity         | PASS   | 0.000000e+00 |
| T08 | rollout_dt_alignment         | PASS   | 1.408168e-05 |
| T09 | forward_vs_step_single       | PASS   | 0.000000e+00 |
| T10 | forward_vs_rollout_multistep | PASS   | 0.000000e+00 |
| T11 | basis_encode_decode_identity | PASS   | 6.491098e-07 |
| T12 | basis_biorthogonality        | PASS   | 3.407730e-11 |
| T13 | sde_scale_physics            | PASS   | 0.000000e+00 |
| T14 | gradient_flow                | PASS   | 0.000000e+00 |
| T15 | dt_zero_mask                 | PASS   | 0.000000e+00 |
| T16 | negative_dt_rejection        | PASS   | 0.000000e+00 |
| T17 | numerical_regression         | PASS   | 0.000000e+00 |

**check.py: 26 / 26 PASS   0 FAIL**

| check name | result | detail |
|------------|--------|--------|
| no_linalg_solve | PASS | linalg.solve found=False |
| no_linalg_inv | PASS | linalg.inv found=False |
| no_get_biorthogonal_pair | PASS | — |
| no_encode_method | PASS | — |
| no_decode_method | PASS | — |
| no_combine_output | PASS | — |
| no_img_height_in_block_init | PASS | — |
| no_img_width_in_block_init | PASS | — |
| no_kernel_size_in_block_init | PASS | — |
| get_matrix_returns_pair | PASS | residual_0=3.28e-15 residual_100=3.26e-11 cond=1.0 |
| basis_grad_w_re | PASS | 8.526e-08 |
| basis_grad_w_im | PASS | 5.684e-08 |
| basis_grad_w_inv_re | PASS | 3.263e-01 |
| basis_grad_w_inv_im | PASS | 5.569e-01 |
| model_train_w_re_gradient | PASS | w_re.grad.max=2.204e+05 |
| eps_1e7_present | PASS | eps=1e-7 in _safe_forcing/_compute_sde_scale |
| alpha_logit_init_2 | PASS | alpha_logit initial value |
| gate_min_001 | PASS | gate_min in GlobalFluxTracker |
| gate_max_099 | PASS | gate_max in GlobalFluxTracker |
| phi1_stability | PASS | max_rel_err=7.81e-08 |
| sde_scale_physics | PASS | err_c=0.00e+00 |
| forward_vs_step_latent | PASS | max_err=0.00e+00 |
| forward_vs_step_h_state | PASS | max_err=0.00e+00 |
| dt_zero_identity | PASS | max_err=0.00e+00 |
| negative_dt_raises | PASS | — |
| encode_decode_identity | PASS | err_a=2.46e-07 err_b=6.49e-07 err_c=3.29e-15 |

## Reproduction

```bash
git clone git@github.com:yrqUni/UniPhy.git
cd UniPhy
git checkout main
export PYTHONPATH="$PWD:$PWD/Model/UniPhy/Check"
```

Run the aggregate verification:

```bash
python Model/UniPhy/Check/dt_check/check.py
```

Run the individual tests:

```bash
python Model/UniPhy/Check/dt_check/run_all.py \
  --tests T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 \
          T11 T12 T13 T14 T15 T16 T17 \
  --log-dir /tmp/dt_check_logs
```

## Environment

The results above were collected on node `172.16.0.21` with PyTorch `2.9.1+cu128`, Python `3.12.12`, CUDA `12.8`, and `CUDA_VISIBLE_DEVICES=0`.
