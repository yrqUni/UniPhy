# UniPhy Temporal Variation Verification

## Overview

This branch contains the verification test suite for the temporal variation design of UniPhy.

## Mathematical Framework

The temporal variation design implements a continuous-time SSM:

```text
dh/dt = λh + u(t),  λ = lam_re + i·lam_im,  lam_re < 0
```

Discretised as:

```text
h[t] = exp(λ·dt)·h[t-1] + φ₁(λ·dt)·dt·u[t]
```

where φ₁(z) = (exp(z)−1)/z, numerically stabilised near z=0 via Taylor expansion for |z| < 1e-7.

The ComplexSVDTransform encodes latent states into an eigenspace defined by a learnable matrix W blended with a fixed DFT matrix, controlled by a scalar α = sigmoid(alpha_logit). W and W_inv are parametrised through coupled perturbation matrices around the DFT pair, and biorthogonality ||W·W_inv − I||_F is maintained by a regularisation term with basis_reg_weight = 0.01.

## Test Catalogue

| id | name | layer | what is tested | tolerance |
| --- | --- | --- | --- | --- |
| T01 | phi1_stability | 1 | _safe_forcing Taylor accuracy | 1e-5 rel |
| T02 | ssm_discretisation | 1 | O(dt²) convergence | 1e-5 |
| T03 | dt_zero_limit | 1 | decay→1, forcing→0 at dt=0 | O(dt²) |
| T04 | variable_dt_pscan | 1 | PScan == sequential recurrence | 1e-4 |
| T05 | hprev_injection_equivalence | 2 | h_prev injection trick | 1e-5 |
| T06 | flux_prev_compensation | 2 | flux_prev cumprod correction | 1e-4 |
| T07 | cumprod_decay_purity | 2 | a_flux == pure decay | 1e-6 |
| T08 | rollout_dt_alignment | 2 | dt indexing in forward_rollout | 1e-4 |
| T09 | forward_vs_step_single | 2 | forward() == forward_step() | 1e-4 |
| T10 | forward_vs_rollout_multistep | 2 | rollout == step chain N=1..24 | 1e-4 |
| T11 | basis_encode_decode_identity | 2 | decode(encode(x)) ≈ x | 1e-4 |
| T12 | basis_biorthogonality | 2 | \|\|W·W_inv − I\|\| monitoring | record |
| T13 | sde_scale_physics | 2 | SDE scale monotonicity | asserts |
| T14 | gradient_flow | 3 | all temporal params get grad | 0 failures |
| T15 | dt_zero_mask | 3 | dt=0 → output == input | 1e-6 |
| T16 | negative_dt_rejection | 3 | dt<0 raises ValueError | 4/4 |
| T17 | numerical_regression | 3 | golden value stability | 1e-5 |

Layer 1 = Mathematical correctness of operators  
Layer 2 = Code-to-math correspondence  
Layer 3 = System-level properties

## Test Results

Verified on commit `0bf240e4dd14862490d109e9746f43e3d0f97d80`. **Summary: 26/26 PASS  0/26 FAIL**

| check | result | detail |
| --- | --- | --- |
| no_linalg_solve | PASS | linalg.solve found=False |
| no_linalg_inv | PASS | linalg.inv found=False |
| no_get_biorthogonal_pair | PASS | removed |
| no_encode_method | PASS | removed |
| no_decode_method | PASS | removed |
| no_combine_output | PASS | removed |
| no_img_height_in_block_init | PASS | removed |
| no_img_width_in_block_init | PASS | removed |
| no_kernel_size_in_block_init | PASS | removed |
| get_matrix_returns_pair | PASS | residual_0=3.28e-15 residual_100=3.37e-11 cond=1.0 |
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
| negative_dt_raises | PASS | ValueError raised |
| encode_decode_identity | PASS | err_a=2.46e-07 err_b=6.49e-07 err_c=3.29e-15 |

## Reproduction

Clone and run check script:

```bash
git clone git@github.com:yrqUni/UniPhy.git
git checkout dt-check
conda activate tch
python -m dt_check.check
```

Run individual tests:

```bash
python -m dt_check.run_all --tests T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 T13 T14 T15 T16 T17 --log-dir /tmp/logs
```

## Environment

| item | value |
| --- | --- |
| git commit | 1c1cad8e44a0ce4ff29216feeaea2aeee83b3564 |
| branch tested | main |
| date | 2026-04-20 |
| torch | 2.9.1+cu128 |
| CUDA | 12.8 |
