# EXP_KAIST_M5_IA_DASR_640_S0

- Status: `validated`
- Setting: `formal single-model mainline`
- Paper name: `IA-DASR-main`
- Model: `DCAF + CDR + DSRE + ignore-aware objectness`
- Dataset: `KAIST`
- Protocol: `Reasonable`
- Input size: `640`
- Seed: `0`
- Checkpoint: `checkpoints/ia_dasr_stage2_e10_best.pt` (not distributed yet)
- Config alias: `configs/models/ia_dasr_formal.yaml`
- Source model config: `src/models/transformer/yolov5s_Transfusion_KAIST_IA_DASR_3050.yaml`

## Metrics

- `MR-all = 7.1372`
- `MR-day = 7.9167`
- `MR-night = 4.1591`
- `mAP@0.5 = 71.7411`
- `mAP@0.5:0.95 = 35.3757`

## Runtime Recipe

- Stage-1 initialization checkpoint: `checkpoints/ia_dasr_stage1_best.pt`
- `--kaist-day-roi-filter`
- `--kaist-ignore-aware-obj`
- `--drr-aux-weight 0.03`

See:

- `configs/experiments/formal_mainline.yaml`
- `results/formal_mainline/opt.yaml`

## Evidence

- `results/formal_mainline/summary_repro_eval.source.csv`
- `results/benchmark_summary.csv`
- `docs/BENCHMARKS.md`

## Notes

- This is the main single-model result card that public project claims should
  anchor to.
- Do not merge this card with routed or expert-combined systems in one table without a clear `Setting` column.
- Historical documents rounded this line to `7.13 / 7.91 / 4.17`; the public
  release uses the canonical full-precision export.
- The validated run was a stage-2 continuation; the exact stage-1 initialization
  checkpoint has not yet been published.
