# EXP_KAIST_B0_RGB_640_S0

- Status: `validated`
- Setting: `single-model baseline`
- Model: `YOLOv5s RGB-only`
- Dataset: `KAIST`
- Protocol: `Reasonable`
- Input size: `640`
- Seed: `0`
- Core runtime flag: `--mono-rgb`

## Metrics

- `MR-all = 39.1741`
- `MR-day = 31.6749`
- `MR-night = 53.6021`
- `mAP@0.5 = 51.0720`
- `mAP@0.5:0.95 = 19.6662`

## Evidence

- `results/formal_mainline/summary_repro_eval.source.csv`
- `results/benchmark_summary.csv`

## Notes

- Same-protocol repository re-evaluation used to show that visible-only input
  degrades strongly at night.
- Older project notes contained a different `35.40%` MR snapshot. It is not used
  in public claims because it conflicts with the machine-readable re-evaluation.
