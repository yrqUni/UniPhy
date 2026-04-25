# UniPhy numerical verification

The `Check/` directory contains the numerical verification suite for UniPhy. The suite is restricted to independent numerical contracts and is organized as a consecutive `T01`–`T13` sequence.

## Run

```bash
python -m Check.tests.run_all --log-dir <log_dir> --json-out <json_path>
```

To run a subset:

```bash
python -m Check.tests.run_all --tests T01 T09 T13 --log-dir <log_dir> --json-out <json_path>
```

## Coverage

- `T01` transition operator discretization
- `T02` scan forward/backward correctness
- `T03` flux scan vs. step equivalence
- `T04` block forward vs. forward-step consistency
- `T05` rollout invariance under chunking/stride/offset
- `T06` basis inverse roundtrip and biorthogonality
- `T07` stochastic scaling and explicit-noise normalization
- `T08` timestep validation and zero-identity semantics
- `T09` CRPS consistency across train, align, and evaluation paths
- `T10` ERA5 dataset interpolation and `dt` semantics
- `T11` flux gate range contract
- `T12` encoder/decoder and FFN shape/dtype contract
- `T13` numerical regression baseline

## Latest recorded run

Environment:

- Node: `node21`
- Device: `NVIDIA A800-SXM4-80GB`
- Python: `3.12.12`
- Torch: `2.9.1+cu128`
- Date: `2026-04-25T15:30:38.920895`

Command:

```bash
python -m Check.tests.run_all --log-dir /home/ruiqingyan/Agent/UniPhy/Check/logs --json-out /home/ruiqingyan/Agent/UniPhy/Check/logs/results.json
```

Summary:

- Total: `13`
- Pass: `13`
- Fail: `0`
- Skip: `0`

## Result snapshot

| Test | Status | Max error | Detail |
| --- | --- | ---: | --- |
| T01 | PASS | 5.83e-07 | transition operator and zero-step limit |
| T02 | PASS | 2.90e-06 | scan forward/backward correctness |
| T03 | PASS | 6.66e-08 | flux scan recurrence equivalence |
| T04 | PASS | 2.95e-13 | block forward/step consistency |
| T05 | PASS | 4.09e-13 | rollout chunk/stride/offset invariance |
| T06 | PASS | 2.46e-07 | basis roundtrip and inverse consistency |
| T07 | PASS | 5.96e-08 | stochastic scaling and normalization |
| T08 | PASS | 0.00e+00 | timestep validation and zero-identity |
| T09 | PASS | 1.19e-07 | CRPS consistency across code paths |
| T10 | PASS | 0.00e+00 | ERA5 interpolation and `dt` semantics |
| T11 | PASS | 0.00e+00 | gate-range contract |
| T12 | PASS | 7.45e-08 | IO and FFN shape/dtype contract |
| T13 | PASS | 0.00e+00 | numerical regression baseline |

Detailed artifacts are written to the log directory declared in the command above, including per-test result files and the JSON summary.
