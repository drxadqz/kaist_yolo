# DARP-Net naming and experiment traceability

The public model name and the historical experiment identifiers intentionally
coexist. The former is used in the thesis, repository landing page, figures,
and résumé. The latter remains in code, config filenames, archived run metadata,
and checkpoint filenames so results can be traced without breaking paths.

| Historical identifier | Public name | Meaning |
|---|---|---|
| `IA-DASR Round 2I+` | **DARP-Net** | Locked protocol-aware single checkpoint; official reload MR-all/day/night `6.909/7.370/4.844%` |
| `IA-DASR Stage2` / formal mainline | **DARP-Fusion** | Protocol-neutral deformable alignment and reliability-guided fusion predecessor; MR-all `7.137%` |
| `FusionOnly` | **DARP-FusionOnly** | Fusion-only cross-dataset control |
| `Portable` | **DARP-Portable** | Dataset-portable setting with KAIST-only calibration disabled |

## Module names

| Abbreviation | Full name | Role |
|---|---|---|
| DCAF | Deformable Cross-modal Alignment Fusion | Multi-head, bidirectional local deformable cross-attention at P3/P4/P5 |
| CDR | Consensus-Detail Residual Refinement | Separates cross-modal consensus from modality-specific detail |
| DSRE | Detail-Scene Reliability Estimator | Models foreground support, conflict, uncertainty, and modality reliability |
| PFH | Protocol Factorization Head | Predicts humanness, countability, groupness, uncertainty, and protocol bias |
| BPSC | Bounded Protocol Score Calibration | Applies a weak bounded residual score factor at inference |
| PUR | Protocol-risk Upper-bound Regularization | Constrains duplicate-halo and isolated-night risks during training |

## Claim boundary

DARP-Net is a Transformer-based RGB-T detector, not a large language model.
Its Transformer connection is the cross-modal attention mechanism in the vision
fusion stack. The optional experiment-report copilot reads the canonical result
table after evaluation and can produce a structured report; it does not take
part in detector training or inference and does not explain any metric gain.

The locked DARP-Net number includes KAIST daytime ROI filtering, night score
calibration, and bounded protocol-score settings. Portable CVC-14 and LLVIP
experiments disable KAIST-only settings and are reported as separate,
target-trained studies.
