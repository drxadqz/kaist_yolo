# IA-DASR Round 2I+ model card

## Summary

Round 2I+ is the best archived **single-checkpoint, protocol-optimized** IA-DASR
variant. It adds a factorized protocol-semantic head and bounded score
calibration to the dual-stream detector.

| Field | Value |
|---|---|
| Dataset/protocol | KAIST Reasonable |
| Input size | 640 |
| Checkpoint setting | one inference checkpoint |
| Inference checkpoint SHA-256 | `626D279F6C0D547F3559B3483009D9C34F860D1D9A1D970F9950F38BAC03B7C6` |
| Architecture config | `configs/models/ia_dasr_round2i_plus.yaml` |
| Claim class | protocol-optimized, not formal mainline |

## Locked result

Official reload/fused evaluation, preserved in the
[machine-readable evidence export](../results/protocol_aware_best/official_reload_metrics.source.json):

| Metric | Value |
|---|---:|
| MR-all | 6.909% |
| MR-day | 7.370% |
| MR-night | 4.844% |
| MR-near / medium / far | 0.79 / 14.31 / 50.89% |
| MR-none / partial / heavy | 25.83 / 25.93 / 49.96% |
| Recall-all | 98.42% |

Training-time epoch-2 row:

| Metric | Value |
|---|---:|
| MR-all / day / night | 6.8899 / 7.3680 / 4.8358% |
| mAP@.5 | 71.7859% |
| mAP@.5:.95 | 34.7485% |
| Recall-all | 98.4182% |

Use the official reload value in public summaries.

## Required inference behavior

- KAIST daytime ROI filter;
- night score calibration;
- PCSF semantic score factor enabled;
- semantic alpha 0.05;
- retained factor clipped to [0.95, 1.06];
- semantic center 0.25;
- beta 4.0.

The score factor is deliberately weak and cannot replace the detector output.

## Cross-dataset boundary

KAIST-specific switches must be disabled on other datasets. Zero-shot
diagnostics are weak on FLIR and moderate on LLVIP; they do not establish broad
domain generalization. Target-trained CVC-14/LLVIP runs are separate experiments.

## Weight availability

The checkpoint is not committed and no release URL is claimed. When published,
verify it against the SHA-256 above. Raw optimizer checkpoints are not release
artifacts.

## Prohibited descriptions

- generic IA-DASR mainline;
- state of the art;
- general multispectral foundation model;
- LLM or VLM detector;
- cross-dataset generalization without target adaptation.
