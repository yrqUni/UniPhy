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
| T11 | basis_encode_decode_identity | 2 | basis encode/decode identity in three regimes | mixed |
| T12 | basis_biorthogonality | 2 | biorthogonality residual before and after SGD | record only |
| T13 | sde_scale_physics | 2 | stochastic scale physical monotonicity and zero limit | err < 1e-5 |
| T14 | gradient_flow | 3 | gradient flow to all temporal variation parameters | no zero grads |
| T15 | dt_zero_mask | 3 | dt zero masking in forward and forward_step | max err < 1e-6 |
| T16 | negative_dt_rejection | 3 | rejection of negative dt values | all 4 cases |
| T17 | numerical_regression | 3 | fixed-seed numerical regression against golden outputs | max err < 1e-5 |

## Results
| id | name | status | max_error | note |
| --- | --- | --- | --- | --- |
| T1 | phi1_stability | pending | - | pending |
| T2 | ssm_discretisation | pending | - | pending |
| T3 | dt_zero_limit | pending | - | pending |
| T4 | variable_dt_pscan | pending | - | pending |
| T5 | hprev_injection_equivalence | pending | - | pending |
| T6 | flux_prev_compensation | pending | - | pending |
| T7 | cumprod_decay_purity | pending | - | pending |
| T8 | rollout_dt_alignment | pending | - | pending |
| T9 | forward_vs_step_single | pending | - | pending |
| T10 | forward_vs_rollout_multistep | pending | - | pending |
| T11 | basis_encode_decode_identity | pending | - | pending |
| T12 | basis_biorthogonality | pending | - | pending |
| T13 | sde_scale_physics | pending | - | pending |
| T14 | gradient_flow | pending | - | pending |
| T15 | dt_zero_mask | pending | - | pending |
| T16 | negative_dt_rejection | pending | - | pending |
| T17 | numerical_regression | pending | - | pending |

## Reproduction
```bash
git clone git@github.com:yrqUni/UniPhy.git
git checkout dt-check
conda activate tch
cd UniPhy
python -m dt_check.run_all --log-dir /tmp/dt_check_logs
```

## Environment
Pending step5 collection.
