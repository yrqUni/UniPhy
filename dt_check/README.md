# UniPhy Temporal Variation Verification

## Overview
This branch verifies the temporal variation design of UniPhy against the main branch implementation. The suite covers 17 tests across three layers: mathematical correctness of the operators, code-to-math correspondence in the implementation, and system-level temporal behavior.

## Mathematical Framework
The tests target the continuous-time state-space update, its discrete transition operators, the phi1 forcing term used near zero, the GlobalFluxTracker scan recurrence, the ComplexSVDTransform basis pair, and the stochastic scale term used in the temporal propagator.

## Test Catalogue
| id | name | layer | what is tested | tolerance |
| --- | --- | --- | --- | --- |
| T1 | phi1_stability | 1 | Taylor switch accuracy in `_safe_forcing` near zero | rel err < 1e-5 |
| T2 | ssm_discretisation | 1 | discrete operators against analytic solution | max err < 1e-5 |
| T3 | dt_zero_limit | 1 | decay and forcing limit as dt approaches zero | residual < 100 dt^2 |
| T4 | variable_dt_pscan | 1 | pscan against sequential recurrence for non-uniform dt | max err < 1e-4 |
| T5 | hprev_injection_equivalence | 2 | `h_prev` injection equivalence to sequential recurrence | max err < 1e-5 |
| T6 | flux_prev_compensation | 2 | `flux_prev` compensation via cumulative decay | max err < 1e-4 |
| T7 | cumprod_decay_purity | 2 | flux scan decay operator purity | max err < 1e-6 |
| T8 | rollout_dt_alignment | 2 | rollout context dt slicing alignment | max err < 1e-4 |
| T9 | forward_vs_step_single | 2 | single-step block forward equivalence | max err < 1e-4 |
| T10 | forward_vs_rollout_multistep | 2 | multistep rollout equivalence and error growth | max err < 1e-4 |
| T11 | basis_encode_decode_identity | 2 | basis encode/decode identity in three regimes | init < 1e-5, random < 1e-4, DFT < 1e-5 |
| T12 | basis_biorthogonality | 2 | biorthogonality residual before and after SGD | record only |
| T13 | sde_scale_physics | 2 | stochastic scale physical monotonicity and zero limit | asserts + err < 1e-5 |
| T14 | gradient_flow | 3 | gradient flow to all temporal variation parameters | 0 zero-grad params |
| T15 | dt_zero_mask | 3 | dt zero masking in forward and forward_step | max err < 1e-6 |
| T16 | negative_dt_rejection | 3 | rejection of negative dt values | 4/4 cases |
| T17 | numerical_regression | 3 | fixed-seed numerical regression against golden outputs | max err < 1e-5 |

## Results
Tested against: main branch commit 772fb75788eea7b026c6d79c09e53fd52c231143
Date: 2026-04-20  Environment: torch=2.9.1+cu128 CUDA=12.8

Overall: 16/17 PASS  1/17 FAIL  0/17 SKIP

| id | name | layer | status | max_error | note |
| --- | --- | --- | --- | --- | --- |
| T1 | phi1_stability | 1 | PASS | 1.347725e-07 | boundary check passed |
| T2 | ssm_discretisation | 1 | PASS | 5.331202e-07 | O(dt^2) slopes = 2.00, 2.00, 2.00 |
| T3 | dt_zero_limit | 1 | PASS | 5.746428e-11 | zero-step residual well below bound |
| T4 | variable_dt_pscan | 1 | PASS | 4.915125e-07 | uniform, random, and sub-step sequences matched |
| T5 | hprev_injection_equivalence | 2 | PASS | 4.768372e-07 | `h_prev` injection matches sequential recurrence |
| T6 | flux_prev_compensation | 2 | PASS | 2.665601e-07 | cumulative decay compensation matched |
| T7 | cumprod_decay_purity | 2 | PASS | 0.000000e+00 | pure decay operator confirmed |
| T8 | rollout_dt_alignment | 2 | PASS | 1.343064e-05 | rollout dt alignment matched stepwise execution |
| T9 | forward_vs_step_single | 2 | PASS | 0.000000e+00 | forward and forward_step matched exactly |
| T10 | forward_vs_rollout_multistep | 2 | PASS | 0.000000e+00 | rollout and repeated forward_step matched exactly |
| T11 | basis_encode_decode_identity | 2 | PASS | 3.129102e-06 | init, random, and DFT regimes passed |
| T12 | basis_biorthogonality | 2 | PASS | 7.996146e-16 | stable at init and after 100 SGD steps |
| T13 | sde_scale_physics | 2 | PASS | 0.000000e+00 | monotonicity and zero-limit checks passed |
| T14 | gradient_flow | 3 | FAIL | 4.000000e+00 | zero-grad params: `w_re`, `w_im`, `w_inv_re`, `w_inv_im` |
| T15 | dt_zero_mask | 3 | PASS | 0.000000e+00 | dt=0 masking held in sequence and step modes |
| T16 | negative_dt_rejection | 3 | PASS | 0.000000e+00 | all 4 rejection cases passed |
| T17 | numerical_regression | 3 | PASS | 0.000000e+00 | golden outputs saved on first run |

### Issues Found
**T14 — gradient_flow**  [CRITICAL]
Observed: max_error = 4.000000e+00  (tolerance = 0 zero-grad params)
Meaning: This test measures whether every temporal variation parameter participates in the training signal. The failure shows that the learned basis parameters `w_re`, `w_im`, `w_inv_re`, and `w_inv_im` receive zero gradient under the tested training path, so part of the temporal variation parameterization is not being learned.
Suspected cause: The current loss path through `ComplexSVDTransform` in `Model/UniPhy/UniPhyOps.py` and its use in `Model/UniPhy/ModelUniPhy.py` does not produce non-zero sensitivity for those basis weights in the tested configuration.

### T12 — Biorthogonality Record
residual at init: 8.00e-16
residual after 100 SGD steps: 8.00e-16
condition number: 1.3
Recommendation: no action needed.

## Conclusions
Overall verdict: CRITICAL FAILURE

Per-layer summary:
Layer 1 (Mathematical correctness): verified; all operator-level tests passed with comfortable margins.
Layer 2 (Code-to-math): verified; all correspondence tests passed and T12 remained numerically stable.
Layer 3 (System properties): critical failure; T14 shows basis parameters do not receive gradients even though masking, rejection, and regression behavior passed.

## Reproduction
```bash
git clone git@github.com:yrqUni/UniPhy.git
git checkout dt-check
conda activate tch
cd UniPhy
python -m dt_check.run_all --log-dir /tmp/dt_check_logs
```

## Environment
| item | value |
| --- | --- |
| git commit | 772fb75788eea7b026c6d79c09e53fd52c231143 |
| branch | main (tested) |
| date | 2026-04-20 |
| torch | 2.9.1+cu128 |
| CUDA | 12.8 |
