# UniPhy Verification Suite

This is the verification report for UniPhy's mathematical operators and system-level contracts. The suite separates numerical operator checks, numerical system checks, and static regression guards so that readers can see which tests measure floating-point behavior, which tests enforce source-level contracts, and which checks catch specific failure modes.

## Mathematical Background

UniPhy evolves the continuous-time complex state-space model

$$
\frac{dh}{dt} = \lambda h + u(t), \qquad \lambda = \lambda_{\mathrm{re}} + i\lambda_{\mathrm{im}}, \qquad \lambda_{\mathrm{re}} < 0.
$$

The discretization implemented by the source code is

$$
h_t = \exp(\lambda \tau)\, h_{t-1} + \tau\,\phi_1(\lambda \tau)\,u_t,
\qquad \tau = \frac{dt}{dt_{\mathrm{ref}}},
\qquad \phi_1(z) = \frac{e^z - 1}{z}.
$$

`lambda` is per `dt_ref` normalized, not per hour. In the default ERA5 configuration `dt_ref = 6.0` hours, so internal time is measured in 6-hour reference units. The dimensional analysis and practical implications are expanded in the project README.

For small |z|, the forcing helper uses the stabilized Taylor branch

$$
\phi_1(z) \approx 1 + \frac{z}{2} + \frac{z^2}{6}, \qquad \text{for } |z| \le 10^{-7}.
$$

The learnable basis transform is implemented as a bounded perturbation of the stored DFT basis. Let

$$
\alpha = \sigma(\alpha_{\mathrm{logit}}),
\qquad \beta = 1 - \alpha,
\qquad a = 1 + 10^{-3}\alpha,
\qquad s = \mathrm{dim}^{-1/2}.
$$

The bounded perturbation uses raw trainable matrices passed through a bounded map (here `tanh`) and a spectral cap $\rho_{\max}$:

$$
\Delta = \beta\,\Phi(W_{\mathrm{re,raw}}, W_{\mathrm{im,raw}})\, s,
\qquad \|\Delta\|_2 \le \rho_{\max}.
$$

Then

$$
L = I + \Delta,
\qquad
L^{-1}_{(N)} = \sum_{k=0}^{N} (-\Delta)^k,
$$

and the returned matrices are

$$
W = a\,(D_{\mathrm{FT}} L),
\qquad
W^{-1} = \frac{L^{-1}_{(N)} D_{\mathrm{FT}}^{-1}}{a}.
$$

The truncation error obeys the Neumann-series remainder bound

$$
\|W W^{-1} - I\|_F = O(\rho_{\max}^{N+1}),
$$

so `neumann_terms` trades extra matrix multiplies for tighter inverse accuracy.

The stochastic term is a reparameterized OU-style increment. The caller supplies an external noise tensor, the implementation normalizes it by RMS, and then scales it by the timestep-dependent square root of the discretized variance:

$$
\sigma(\tau) = \mathrm{base\_noise}\,\sqrt{\tau\,\phi_1(2\lambda_{\mathrm{re}}\tau)}.
$$

`pscan()` implements the standard affine linear recurrence combine rule for

$$
S_t = A_t S_{t-1} + X_t,
$$

with associative combine

$$
(B,Y) \star (A,X) = (BA, BX + Y).
$$

Internally, the scan pads singleton state dimension D=1 and trailing width 1 to the minimum matrix shape required by the Triton kernel, then strips the padding on output.

## Test Catalogue

### Numerical operator tests

| ID | Name | What is Verified | Tolerance |
|---|---|---|---|
| T01 | `phi1_stability` | Stabilized phi1 matches the float64 reference near zero | 1e-5 relative |
| T02 | `ssm_discretisation` | The normalized discrete propagator matches the analytic SSM solution | 1e-5 |
| T03 | `dt_zero_limit` | As dt approaches 0, decay approaches identity and forcing approaches zero | asymptotic |
| T04 | `variable_dt_pscan` | Parallel scan matches sequential recurrence for variable timesteps | 1e-4 |
| T05 | `hprev_injection_equivalence` | Initial hidden-state injection matches direct recurrence construction | 1e-5 |
| T06 | `flux_prev_compensation` | Flux carry-over compensation matches explicit recurrence | 1e-4 |
| T07 | `cumprod_decay_purity` | Flux decay operator matches its closed-form exponential branch | exact algebraic check |
| T08 | `rollout_dt_alignment` | Context-state timestep indexing matches the stepped update path | 1e-4 |
| T09 | `forward_vs_step_single` | forward() and forward_step() agree for a single deterministic step | exact implementation equivalence |
| T10 | `forward_vs_rollout_multistep` | Autoregressive rollout matches chained single-step updates | exact implementation equivalence |
| T11 | `basis_encode_decode_identity` | Basis encode/decode stays numerically close to identity | 1e-4 |
| T12 | `basis_biorthogonality` | The basis pair rejects a deliberately broken inverse and preserves a correct one | assertion-based |
| T13 | `sde_scale_physics` | Diffusion scale varies monotonically with timestep and damping | assertions |
| T14 | `gradient_flow` | All temporal parameters receive gradients | none |
| T15 | `dt_zero_mask` | Zero-timestep outputs preserve the input exactly while non-zero steps do not | exact mask + nontriviality check |
| T16 | `negative_dt_rejection` | Negative timesteps are rejected with ValueError | all cases |
| T17 | `numerical_regression` | Current outputs match the golden tensors | 1e-5 |
| T18 | `basis_inverse_under_randomized_params` | Randomized basis perturbations still produce a valid inverse | 1e-2 |

### Static regression guards

| ID | Name | What is Verified |
|---|---|---|
| T21 | `crps_gradient_decomposition` | align.py contains the correct CRPS pairwise term and no dead code paths |
| T22 | `pscan_padding_contract` | pscan() retains the singleton-padding docstring contract |
| T23 | `t12_is_not_trivial` | T12 contains real assertion-based checks |
| T24 | `t17_missing_golden_policy` | T17 does not self-authorize missing goldens |
| T25 | `recheck_runner_features` | run_all.py supports json-out, and T17 supports regenerate with SHA256 reporting |

Static guards protect against future regression in the checked source contracts. They do not validate floating-point operator correctness on their own; that role remains with the numerical operator and system tests.

### Numerical system tests

| ID | Name | What is Verified |
|---|---|---|
| S01 | `parallel_serial_consistency` | Parallel and serial execution agree for forward, forward_rollout, flux scan, temporal propagator scan, and spatial mixing |
| S02 | `timestep_semantics` | Zero-step identity, timestep scaling, rollout indexing, stride/offset rules, scalar-vs-tensor dt handling, small-eigenvalue limits, negative-dt rejection, and dt normalization shapes |
| S03 | `parameter_consistency` | dt_ref, noise scales, state dimensions, parameter naming, basis alpha shape, mixer/tracker types, encoder state, skip wiring, and fixed numerical constants |
| S04 | `architecture_verification` | Removal of deprecated variants, constructor signature cleanup, and absence of forbidden linear-algebra helpers |
| S05 | `pscan_correctness` | Parallel-scan forward/backward agreement, shape coverage, long-sequence behavior, and 4D-input semantics |

## Results

| Field | Value |
|---|---|
| Device | NVIDIA A800-SXM4-80GB |
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu128 |
| Triton | 3.5.1 |
| CUDA | 12.8 |
| Overall | 28 / 28 PASS, 0 FAIL, 0 SKIP |

### Numerical operator results

| ID | Name | Status | Max Error |
|---|---|---|---:|
| T01 | `phi1_stability` | PASS | 1.347725e-07 |
| T02 | `ssm_discretisation` | PASS | 5.331202e-07 |
| T03 | `dt_zero_limit` | PASS | 5.746428e-11 |
| T04 | `variable_dt_pscan` | PASS | 4.915125e-07 |
| T05 | `hprev_injection_equivalence` | PASS | 4.768372e-07 |
| T06 | `flux_prev_compensation` | PASS | 2.665601e-07 |
| T07 | `cumprod_decay_purity` | PASS | 0.000000e+00 |
| T08 | `rollout_dt_alignment` | PASS | 1.408168e-05 |
| T09 | `forward_vs_step_single` | PASS | 0.000000e+00 |
| T10 | `forward_vs_rollout_multistep` | PASS | 0.000000e+00 |
| T11 | `basis_encode_decode_identity` | PASS | 7.378708e-07 |
| T12 | `basis_biorthogonality` | PASS | 3.779047e-14 |
| T13 | `sde_scale_physics` | PASS | 0.000000e+00 |
| T14 | `gradient_flow` | PASS | 0.000000e+00 |
| T15 | `dt_zero_mask` | PASS | 0.000000e+00 |
| T16 | `negative_dt_rejection` | PASS | 0.000000e+00 |
| T17 | `numerical_regression` | PASS | 0.000000e+00 |
| T18 | `basis_inverse_under_randomized_params` | PASS | 1.579590e-14 |

### Static regression guard results

| ID | Name | Status |
|---|---|---|
| T21 | `crps_gradient_decomposition` | PASS |
| T22 | `pscan_padding_contract` | PASS |
| T23 | `t12_is_not_trivial` | PASS |
| T24 | `t17_missing_golden_policy` | PASS |
| T25 | `recheck_runner_features` | PASS |

### Numerical system results

| ID | Name | Status |
|---|---|---|
| S01 | `parallel_serial_consistency` | PASS |
| S02 | `timestep_semantics` | PASS |
| S03 | `parameter_consistency` | PASS |
| S04 | `architecture_verification` | PASS |
| S05 | `pscan_correctness` | PASS |

Zero-valued max errors are expected for tests that are exact-by-construction (T07, T09, T10, T15, T17) or where the reported scalar is a count/guard outcome rather than a floating-point drift metric (T13, T14, T16, T21 through T25).

## Reproduction

```bash
git clone git@github.com:yrqUni/UniPhy.git
cd UniPhy
python Check/tests/T17_numerical_regression.py --regenerate
python Check/tests/run_all.py --tests T S \
    --log-dir /tmp/uniphy_check_logs \
    --json-out /tmp/uniphy_check_logs/results.json
```

Single-test examples:

```bash
python Check/tests/T18_basis_inverse_under_randomized_params.py
python Check/tests/S05_pscan_correctness.py
```
