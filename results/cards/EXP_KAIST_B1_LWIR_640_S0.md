# EXP_KAIST_B1_LWIR_640_S0

- Status: `validated`
- Setting: `single-model baseline`
- Model: `YOLOv5s LWIR-only`
- Dataset: `KAIST`
- Protocol: `Reasonable`
- Input size: `640`
- Seed: `0`
- Core runtime flag: `--mono-thermal`

## Metrics

- `MR-all = 21.3832`
- `MR-day = 26.5253`
- `MR-night = 10.3017`
- `mAP@0.5 = 67.2170`
- `mAP@0.5:0.95 = 27.7010`

## Evidence

- `results/formal_mainline/summary_repro_eval.source.csv`
- `results/benchmark_summary.csv`

## Notes

- Same-protocol repository re-evaluation used to show that thermal-only input
  is much stronger at night but remains weaker in daytime scenes.
- Older project notes contained a different `21.74%` MR snapshot. It is not used
  in public claims.
