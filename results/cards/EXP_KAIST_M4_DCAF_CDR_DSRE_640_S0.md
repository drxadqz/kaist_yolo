# EXP_KAIST_M4_DCAF_CDR_DSRE_640_S0

- Status: `validated`
- Setting: `single-model ablation`
- Model: `DCAF + CDR + DSRE`
- Dataset: `KAIST`
- Protocol: `Reasonable`
- Input size: `640`
- Seed: `0`

## Metrics

- `MR-all = 9.0550`
- `MR-day = 10.1644`
- `MR-night = 6.3577`
- `mAP@0.5 = 77.4729`
- `mAP@0.5:0.95 = 34.6111`

## Evidence

- `results/formal_mainline/m4_metrics_summary.json`
- `results/benchmark_summary.csv`

## Boundary

M4 does not use the complete M5 ignore-aware recipe and is slightly worse than
M2 on MR-all. Training budgets and hyperparameters across M1–M5 were not fully
normalized, so this row is not a strict causal estimate of DSRE's contribution.
