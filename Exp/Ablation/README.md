# Ablation Protocol

The UniPhy ablation suite uses a single factor comparison design. Each variant changes one modeling choice while the data split, optimization settings, model scale, lead times, and climatology remain matched.

## Public Panel

| Group | Variant | Purpose |
|:--|:--|:--|
| Reference | `baseline` | Active UniPhy model |
| Time conditioning | `A1_no_dt` | Remove explicit interval conditioning |
| Recurrence | `A2_discrete_rnn` | Replace the temporal transition with a plain recurrent update |
| Latent dynamics | `B1_complex_latent` | Historical complex latent pathway |
| Memory | `B2_fixed_decay` | Fix the temporal decay spectrum |
| Determinism | `C1_deterministic` | Remove stochastic sampling |
| Readout | `C2_no_readout_residual` | Remove residual readout correction |
| Readout timing | `C3_constant_readout` | Keep residual readout constant in lead time |
| Spatial mixing | `D1_single_scale` | Keep only the local spatial branch |
| Spatial mixing | `D2_fixed_scale_weights` | Fix scale weights to uniform values |
| Objective | `E1_l1_only` | Remove CRPS from training |
| Integration | `F1_etd1_integrator` | Replace the exact dissipative transition |
| Baseline | `G1_swin_transformer` | Fixed interval Swin Transformer |
| Baseline | `G2_convlstm` | Fixed interval ConvLSTM |

`protocol.py` stores the public variant definitions. `variants.py` maps each definition to a model builder.

## Recommended Protocol

Use 2000 and 2001 for training and 2002 for evaluation. Final claims should use ten epochs. Screening runs may use less, but screening results are not final evidence.

Report RMSE, ACC, and CRPS for lead times `6, 24, 72, 120, 240` hours. For fixed interval comparison, report UniPhy direct and recursive evaluation together with SwinTrans and ConvLSTM on regular 6 h grids. For time sensitivity, evaluate UniPhy on regular 6 h, regular 12 h, and irregular grids with the same checkpoint.

## Result Summary

The current evidence shows that the active deterministic UniPhy panel is competitive on fixed interval evaluation and remains numerically consistent between direct and recursive integer multiple leads. The Tree parallel PScan fallback matches the serial reference on the public verification suite.

## Commands

```bash
python -m Exp.Ablation.plan --output-dir Exp/Ablation/results --data-input-dir /nfs/ERA5_data/data_norm --train-year-range 2000,2001 --eval-year-range 2002,2002 --climatology-year-range 2000,2001 --epochs 10 --seeds 42 --lead-times 6,24,72,120,240 --gpus 8
```

```bash
torchrun --nproc_per_node=<gpus> -m Exp.Ablation.runner --variant A1_no_dt --seed 42 --data-input-dir /nfs/ERA5_data/data_norm --train-year-range 2000,2001 --epochs 10 --ckpt-dir Exp/Ablation/results/A1_no_dt/seed_42
```

```bash
python -m Exp.Ablation.fixed_interval_compare --variant G1_swin_transformer --checkpoint Exp/Ablation/results/G1_swin_transformer/seed_42/ckpt_final.pt --data-input-dir /nfs/ERA5_data/data_norm --climatology-dir /nfs/ERA5_data/data_norm --climatology-year-range 2000,2001 --eval-year-range 2002,2002 --lead-times 6,12,18,24 --step-hours 6 --mode recursive --output-json Exp/Ablation/results/G1_swin_transformer/seed_42/fixed_6h_recursive.json
```
