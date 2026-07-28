# Benchmarks and evidence boundaries

## Canonical source

[`results/benchmark_summary.csv`](../results/benchmark_summary.csv) is the only
hand-off table intended for README generation, automated reporting, and résumé
drafting. It records:

- dataset and protocol;
- model setting;
- input size and seed;
- MR, mAP, recall, and available latency;
- evidence level;
- source artifact;
- a mandatory claim boundary.

The original full-precision re-evaluation export is preserved at
[`results/formal_mainline/summary_repro_eval.source.csv`](../results/formal_mainline/summary_repro_eval.source.csv).
The protocol-aware reload, routed-system, and calibrated post-processing rows
point directly to sanitized machine-readable sources:

- [`official_reload_metrics.source.json`](../results/protocol_aware_best/official_reload_metrics.source.json);
- [`iaer_router_metrics.source.json`](../results/system/iaer_router_metrics.source.json);
- [`ecr_bestall_metrics.source.csv`](../results/postprocess/ecr_bestall_metrics.source.csv)
  and [`ecr_balanced_metrics.source.csv`](../results/postprocess/ecr_balanced_metrics.source.csv).

## Metric interpretation

KAIST Reasonable primarily reports log-average miss rate across a low-FPPI
range. MR is a percentage and lower is better. mAP summarizes a different part
of the detector ranking curve. A method may improve low-FPPI MR without
improving mAP@.5.

That happens here:

| Model | MR-all ↓ | mAP@.5 ↑ | Total ms/image |
|---|---:|---:|---:|
| Early Fusion 6-channel | 13.535 | **75.923** | **19.04** |
| IA-DASR formal | **7.137** | 71.741 | 59.52 |

The defensible conclusion is:

> IA-DASR substantially improves the KAIST low-FPPI miss-rate operating region
> over simple early fusion, at a higher runtime cost; it does not dominate the
> baseline on every detection metric.

## Formal same-protocol matrix

| ID | Model | MR-all | MR-day | MR-night | mAP@.5 | mAP@.5:.95 | Recall-all |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | YOLOv5s RGB-only | 39.174 | 31.675 | 53.602 | 51.072 | 19.666 | 96.564 |
| B1 | YOLOv5s LWIR-only | 21.383 | 26.525 | 10.302 | 67.217 | 27.701 | 98.076 |
| B2 | Early Fusion 6-channel | 13.535 | 15.362 | 8.820 | 75.923 | 34.271 | 98.969 |
| M1 | DCAF | 9.416 | 10.694 | 6.456 | 78.391 | **35.555** | 98.282 |
| M2 | DCAF + CDR | 9.017 | 10.418 | 5.209 | 77.096 | 34.893 | 98.969 |
| M4 | DCAF + CDR + DSRE | 9.055 | 10.164 | 6.358 | 77.473 | 34.611 | 98.625 |
| M5 | IA-DASR formal | **7.137** | **7.917** | **4.159** | 71.741 | 35.376 | 98.351 |

Recorded improvements from B2 to M5:

- MR-all: `13.535 → 7.137`, 6.398 percentage points, 47.27% relative.
- MR-day: `15.362 → 7.917`, 7.445 percentage points, 48.47% relative.
- MR-night: `8.820 → 4.159`, 4.661 percentage points, 52.84% relative.

M3 has no same-level machine-readable evaluation summary and is omitted. The
M1–M5 runs are useful engineering evidence, but their training budgets and
hyperparameters were not fully normalized. M4 is slightly worse than M2 on
MR-all, so the table does not support a claim that DSRE alone caused the final
M5 gain. Do not interpret every difference as a strict causal module ablation.

## Best protocol-aware single checkpoint

The Round 2I+ archive records two nearby values:

- training-time epoch-2 row: `6.8899 / 7.3680 / 4.8358%`;
- official reload/fused re-evaluation: `6.909 / 7.370 / 4.844%`.

Use the official reload value in public tables. The path-sanitized
[source export](../results/protocol_aware_best/official_reload_metrics.source.json)
records the archived evidence-card SHA-256, checkpoint SHA-256, inference
switches, and both metric snapshots. It also records:

- `mAP@.5 = 71.7859%` in the training-time row;
- `Recall-all = 98.4182%`;
- near / medium / far MR = `0.79 / 14.31 / 50.89%`;
- none / partial / heavy MR = `25.83 / 25.93 / 49.96%`.

This result requires KAIST-specific ROI filtering, night calibration, and a
bounded protocol-semantic score factor. It is a single checkpoint, but not a
protocol-neutral mainline.

## Routed and post-processing systems

| System | MR-all | MR-day | MR-night | Interpretation |
|---|---:|---:|---:|---|
| IA-DASR + IAER | 6.900 | 7.620 | 4.190 | illumination-aware expert routing |
| + ECR best-all | **6.841** | **7.429** | 4.276 | calibrated prediction reranking |
| + ECR balanced | 6.883 | 7.462 | **4.115** | calibration selected for balance |

The IAER row is transcribed without local label paths in its
[sanitized evaluator summary](../results/system/iaer_router_metrics.source.json).
The exact ECR exports are retained for the
[best-all](../results/postprocess/ecr_bestall_metrics.source.csv) and
[balanced](../results/postprocess/ecr_balanced_metrics.source.csv) settings.

The best-all ECR value is the lowest recorded MR-all, but it is a
prediction-only system result and slightly worsens night MR relative to the
formal single model. It must never be presented as a new single-detector
checkpoint.

## Cross-dataset evidence

The best archived KAIST checkpoint was evaluated with KAIST-only switches
disabled:

| Diagnostic | Images | mAP@.5 | mAP@.5:.95 | Status |
|---|---:|---:|---:|---|
| FLIR aligned person, zero-shot | 1,013 | 16.79 | 5.40 | weak transfer |
| LLVIP, zero-shot | 2,943 | 40.20 | 13.90 | sanity check only |

Target-dataset fine-tuning records:

| Dataset/run | mAP@.5 | mAP@.5:.95 | Boundary |
|---|---:|---:|---|
| CVC-14 fusion-only fine-tune | 90.83 | 48.99 | target-trained, separate protocol |
| CVC-14 portable branch | 90.98 | 49.24 | target-trained, separate protocol |
| LLVIP stage 1 | 75.28 | 31.23 | target-trained |
| LLVIP stage 2 | 78.30 | 32.57 | target-trained |

These runs demonstrate portability of the engineering stack after target
fine-tuning. They are not evidence that the KAIST checkpoint generalizes broadly
without adaptation.

## Legacy tracking diagnostic

The old repository contained a tracker output and a sparse GT subset. Re-running
the legacy evaluator gives:

```text
Precision 91.36%, Recall 72.55%, MOTA 65.69%, MOTP 75.69%
```

This is excluded from headline results because:

- GT has detections on only 93 frames while tracker output spans 1,828 frames;
- false positives on empty-GT frames are ignored;
- every one of 408 GT boxes has a unique ID, so identity metrics are invalid.

It is retained only as a historical diagnostic in Git history.

## What may be claimed

Safe:

> Built a dual-stream RGB-T detector achieving 7.14% MR on KAIST Reasonable and
> reducing MR-all by 47.3% relative to a same-protocol 6-channel early-fusion
> baseline.

Safe with boundary:

> Archived a KAIST protocol-optimized single checkpoint at 6.909% MR-all and
> 98.42% Recall-all using bounded semantic calibration.

Unsafe:

- “State of the art on KAIST.”
- “The LLM improves detection accuracy.”
- “6.84% single-model MR.”
- “Generalizes across multispectral datasets.”
- “All metrics outperform early fusion.”

Known same-table external methods include values below the formal `7.137%`
mainline, so this release does not claim SOTA.
