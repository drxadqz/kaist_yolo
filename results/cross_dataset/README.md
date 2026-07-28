# Cross-dataset fine-tuning evidence

These CSV files are selected training-result tables from target-dataset
fine-tuning runs. They are included to make the portability claims in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) auditable.

| File | Best epoch | mAP@.5 | mAP@.5:.95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `cvc14_fusiononly_results.csv` | 9 | 90.8312% | 48.9861% | 84.7642% | 85.6259% |
| `cvc14_portable_results.csv` | 5 | 90.9832% | 49.2385% | 82.3793% | 86.4178% |
| `llvip_stage1_results.csv` | 2 | 75.2796% | 31.2277% | 87.5534% | 69.3896% |
| `llvip_stage2_results.csv` | 2 | 78.2968% | 32.5709% | 88.3818% | 72.5587% |

These are target-trained internal validation runs:

- they are not zero-shot results;
- they use dataset-specific training and evaluation;
- they cannot be compared numerically with KAIST log-average MR;
- the original run metadata did not preserve a public seed field, so the
  canonical table records `not_recorded` instead of inventing one;
- they do not establish broad domain generalization.

The LLVIP stage-2 run is a continuation from the best stage-1 checkpoint. The
CVC-14 portable and fusion-only rows are distinct training modes, not repeated
seeds of one experiment.
