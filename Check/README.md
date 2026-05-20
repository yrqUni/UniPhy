# Numerical Verification

`Check` contains the numerical verification suite for UniPhy. The tests focus on mathematical contracts that should remain stable across implementation changes.

## Usage

```bash
python -m Check.tests.run_all --log-dir Check/logs --json-out Check/logs/results.json
```

Run a single test module when focused verification is useful.

```bash
python -m Check.tests.T02_pscan_forward_backward_correctness
```

T13 compares CPU outputs with the committed numerical reference at `Check/golden/golden.pt`. Update this reference only when model semantics or deterministic backend numerics intentionally change.

```bash
python -m Check.tests.T13_numerical_regression_baseline --regenerate
```

## Coverage

| ID | Area | Contract |
|:--|:--|:--|
| T01 | Transition operators | ETD coefficients match analytic solutions and stable small step limits. |
| T02 | Parallel scan | Auto dispatch, Torch Tree, and Triton when available match serial forward and backward references. |
| T03 | Flux tracker | Parallel scan recurrence matches sequential stepping. |
| T04 | Block dynamics | Sequence forward and step wise execution agree. |
| T05 | Rollout | Chunking, stride, and offset selection preserve rollout semantics. |
| T06 | Basis | Complex basis encode and decode round trip correctly. |
| T07 | Stochastic term | OU variance, state independence, and train rollout noise consistency hold. |
| T08 | Time intervals | Zero time step is identity and invalid time values are rejected. |
| T09 | CRPS | Training, alignment, and evaluation CRPS computations agree. |
| T10 | Temporal sampling | Linear interpolation and time intervals are consistent on nonuniform grids. |
| T11 | Flux gate | Gate outputs remain in the configured range. |
| T12 | IO and FFN | Shapes and dtypes are preserved. |
| T13 | Regression | CPU forward and rollout outputs match the golden baseline within tolerance. |
| T14 | Integrator order | ETD2 exhibits second order convergence under step refinement. |
| T15 | Physical units | SDE kernels depend on physical time consistently. |
## Backend Coverage

T02 always checks the serial reference, auto dispatch, and Torch Tree fallback
on CPU. On CUDA systems with Triton support, the same test also checks the
Triton branch.
