# Numerical Verification

`Check` provides a lightweight numerical verification suite for UniPhy. The suite is self contained and requires no ERA5 files.

## Usage

```bash
python -m Check.tests.run_all --log-dir Check/logs --json-out Check/logs/results.json
```

Run a focused test by module name when needed.

```bash
python -m Check.tests.T01_pscan_forward
```

## Coverage

| ID | Area | Contract |
|:--|:--|:--|
| T01 | Parallel scan | Torch Tree and auto dispatch match serial scan references. |
| T02 | Parallel scan gradients | Torch Tree gradients match sequential autograd. |
| T03 | UniPhy recurrence | Sequence forward matches stepwise execution. |
| T04 | Rollout equivalence | Direct long lead rollout equals fixed step recursive rollout when the lead is an integer multiple of `dt_ref`. |
| T05 | Time intervals | Zero time interval is an identity map. |
| T06 | Time validation | Negative and nonfinite time intervals are rejected. |
| T07 | Differentiability | Model output shape is preserved and trainable parameters receive gradients. |
| T08 | Objective | Weighted CRPS matches the pairwise reference formula. |
