# Ablation Protocol

The UniPhy ablation suite is a controlled single factor protocol. Each variant
changes one modeling choice while keeping data, optimization, model scale,
ensemble size, lead times, and climatology fixed.

## Variant Set

| Group | Variant | Isolated factor |
|:--|:--|:--|
| Reference | `baseline` | Full UniPhy model |
| Continuous time sampling | `A1_no_dt` | Physical time interval conditioning |
| Continuous time sampling | `A2_discrete_rnn` | Continuous time dissipative transition |
| Latent dynamics | `B1_complex_latent` | Real dissipative latent state |
| Latent memory | `B2_fixed_decay` | Learned continuous decay spectrum |
| Probabilistic dynamics | `C1_deterministic` | Latent stochastic forcing |
| Probabilistic dynamics | `C2_no_readout_residual` | Residual latent readout |
| Probabilistic dynamics | `C3_constant_readout` | Persistent residual latent readout |
| Spatial coupling | `D1_single_scale` | Multi scale spatial mixing |
| Spatial coupling | `D2_fixed_scale_weights` | Adaptive multi scale weighting |
| Training objective | `E1_l1_only` | CRPS ensemble objective |
| Temporal integration | `F1_etd1_integrator` | Exact dissipative transition |
| Operational baseline | `G1_swin_transformer` | Fixed interval Swin style single frame prediction |
| Operational baseline | `G2_convlstm` | Fixed interval ConvLSTM recurrence |

`protocol.py` stores the public variant specifications. `variants.py` maps
those specifications to model builders.

## Recommended Protocols

The preferred publication protocol uses paired repeated runs for the baseline
and every variant. The default plan uses seeds `{42, 43, 44}` and lead times
`{6 h, 24 h, 72 h, 120 h, 240 h}`. Report RMSE, ACC, and CRPS with mean,
standard deviation, confidence interval, and paired delta against the baseline
at matching seeds.

For a controlled three year study, use 2000 and 2001 for training and 2002 for
evaluation. Each variant should complete ten epochs over the two training
years. This budget is the default accelerated scientific protocol for UniPhy:
it preserves a meaningful optimization trajectory while keeping paired
single factor comparisons tractable. Very small `--max-steps` values are
reserved for connectivity checks and compact validation. They should not be
used as evidence for model ranking.

The time sensitivity study should evaluate every final variant on regular and
irregular lead time grids. Report the same trained checkpoint across all grids
so that changes in RMSE, ACC, and CRPS isolate the effect of inference time
intervals.

SwinTrans and ConvLSTM are fixed interval baselines. They should not be used in
irregular time interval comparisons. Their protocol is restricted to regular
6 h data. Report UniPhy direct prediction to each lead time together with
recursive 6 h rollout for UniPhy, SwinTrans, and ConvLSTM.

## Reference Result

The three year controlled protocol trains on 2000 and 2001 and evaluates on
2002. The final residual free baseline is the lowest mean RMSE model across all
five evaluation grids.

| Grid | Baseline mean RMSE | Best competing variant | Competing mean RMSE |
|:--|--:|:--|--:|
| Standard | 0.087036 | `C2_no_readout_residual` | 0.087257 |
| Regular 6 h | 0.082669 | `C2_no_readout_residual` | 0.082764 |
| Regular 12 h | 0.083601 | `C2_no_readout_residual` | 0.083640 |
| Irregular short | 0.080436 | `C2_no_readout_residual` | 0.080610 |
| Irregular medium | 0.086710 | `C2_no_readout_residual` | 0.086965 |

The residual readout variants show that an additional convolutional readout can
over correct the dissipative latent trajectory during free forecasts. The Euler
transition variant is unstable at long lead time, supporting the exact
dissipative transition used by the baseline.

## Plan

```bash
python -m Exp.Ablation.plan \
    --output-dir Exp/Ablation/results \
    --data-input-dir /data/ERA5 \
    --train-year-range 2000,2016 \
    --eval-year-range 2017,2018 \
    --climatology-year-range 2000,2016 \
    --epochs <N> \
    --gpus <gpus>
```

For the three year screening protocol:

```bash
python -m Exp.Ablation.plan \
    --output-dir Exp/Ablation/results/final_td \
    --data-input-dir /data/ERA5 \
    --train-year-range 2000,2001 \
    --eval-year-range 2002,2002 \
    --climatology-year-range 2000,2001 \
    --epochs 10 \
    --seeds 42 \
    --lead-times 6,24,72,120,240 \
    --gpus 8
```

## Train

```bash
torchrun --nproc_per_node=<gpus> -m Exp.Ablation.runner \
    --variant A1_no_dt \
    --seed 42 \
    --data-input-dir /data/ERA5 \
    --train-year-range 2000,2016 \
    --epochs <N> \
    --ckpt-dir Exp/Ablation/results/A1_no_dt/seed_42
```

Do not pass `--max-steps` for scientific ablation runs. Leaving it unset lets
the runner consume the full epoch defined by the selected training years.

## Evaluate

```bash
python -m Exp.Ablation.eval \
    --variant A1_no_dt \
    --seed 42 \
    --checkpoint Exp/Ablation/results/A1_no_dt/seed_42/ckpt_final.pt \
    --data-input-dir /data/ERA5 \
    --climatology-dir /data/ERA5 \
    --climatology-year-range 2000,2016 \
    --eval-year-range 2017,2018 \
    --lead-times 6,24,72,120,240 \
    --output-json Exp/Ablation/results/A1_no_dt/seed_42/eval.json
```

## Summarize

```bash
python -m Exp.Ablation.compare \
    --results-dir Exp/Ablation/results \
    --lead-times 24,120,240 \
    --metric rmse \
    --output-csv Exp/Ablation/results/summary.csv \
    --output-tex Exp/Ablation/results/summary.tex \
    --output-json Exp/Ablation/results/summary.json
```

`Exp/Ablation/results` is reserved for generated outputs from local
experiments and concise public summaries.

## Time Sensitivity

```bash
python -m Exp.Ablation.dt_sensitivity \
    --variant baseline \
    --checkpoint Exp/Ablation/results/baseline/seed_42/ckpt_final.pt \
    --data-input-dir /data/ERA5 \
    --climatology-dir /data/ERA5 \
    --climatology-year-range 2000,2001 \
    --eval-year-range 2002,2002 \
    --output-json Exp/Ablation/results/baseline/seed_42/dt_sensitivity.json
```

The default grids include regular 6 h and 12 h intervals plus irregular short
and medium range schedules. The same trained checkpoint is evaluated on every
grid so that the result isolates inference time interval behavior.

## Fixed Interval Baselines

```bash
python -m Exp.Ablation.fixed_interval_compare \
    --variant G1_swin_transformer \
    --checkpoint Exp/Ablation/results/G1_swin_transformer/seed_42/ckpt_final.pt \
    --data-input-dir /data/ERA5 \
    --climatology-dir /data/ERA5 \
    --climatology-year-range 2000,2001 \
    --eval-year-range 2002,2002 \
    --lead-times 6,12,18,24 \
    --step-hours 6 \
    --mode recursive \
    --output-json Exp/Ablation/results/G1_swin_transformer/seed_42/fixed_6h_recursive.json
```

Use `--mode direct` for UniPhy direct lead time evaluation and `--mode
recursive` for 6 h autoregressive rollout.

## Files

| File | Purpose |
|:--|:--|
| `protocol.py` | Variant specifications, result validation, and aggregation helpers. |
| `plan.py` | Reproducible execution plan generation. |
| `components.py` | Module replacements used by variants. |
| `variants.py` | Variant registry and model builders. |
| `runner.py` | Distributed training entry point. |
| `eval.py` | RMSE, ACC, and CRPS evaluation. |
| `compare.py` | Multi seed aggregation and table export. |
| `fixed_interval_compare.py` | Fixed 6 h direct and recursive comparison for UniPhy, SwinTrans, and ConvLSTM. |
