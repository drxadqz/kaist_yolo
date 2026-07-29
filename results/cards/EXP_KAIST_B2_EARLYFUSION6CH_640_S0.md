# EXP_KAIST_B2_EARLYFUSION6CH_640_S0

- Status: `validated`
- Setting: `single-model baseline`
- Model: `YOLOv5s naive 6-channel early fusion`
- Dataset: `KAIST`
- Protocol: `Reasonable`
- Input size: `640`
- Seed: `0`
- Core runtime flag: `--early-fusion-6ch`

## Metrics

- `MR-all = 13.5347`
- `MR-day = 15.3622`
- `MR-night = 8.8198`
- `mAP@0.5 = 75.9233`
- `mAP@0.5:0.95 = 34.2715`

## Evidence

- `results/formal_mainline/summary_repro_eval.source.csv`
- `results/benchmark_summary.csv`

## Notes

- This is the key repository reference showing that DARP-Fusion/DARP-Net's MR improvement is not
  explained by naive paired-input concatenation alone.
- Older project notes contained a different `13.90%` MR snapshot. It is not used
  in public claims.
