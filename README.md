<div align="center">

# DARP-Net

### Deformable Alignment, Reliability Modeling and Protocol-aware Detection for RGB-T Pedestrians

[![CI](https://github.com/drxadqz/kaist_yolo/actions/workflows/ci.yml/badge.svg)](https://github.com/drxadqz/kaist_yolo/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](pyproject.toml)
[![Task](https://img.shields.io/badge/Task-RGB--T%20Detection-E85D75.svg)](docs/ARCHITECTURE.md)
[![DARP-Net result](https://img.shields.io/badge/DARP--Net%20MR--all-6.909%25-2A9D8F.svg)](results/protocol_aware_best/official_reload_metrics.source.json)
[![Recall](https://img.shields.io/badge/Recall--all-98.42%25-457B9D.svg)](results/protocol_aware_best/official_reload_metrics.source.json)

**[中文说明](README.zh-CN.md) · [Naming](docs/NAMING.md) · [Method](docs/ARCHITECTURE.md) ·
[Benchmarks](docs/BENCHMARKS.md) · [Reproduction](docs/REPRODUCIBILITY.md) ·
[Model cards](model_cards/)**

</div>

DARP-Net is a dual-stream RGB-thermal detector built to remain useful when the
two sensors are imperfectly aligned, one modality becomes unreliable, or the
benchmark contains ignored and non-countable regions. It combines three-scale
bidirectional deformable cross-modal attention, consensus/detail refinement,
scene-reliability modulation, and ignore-aware supervision in a YOLOv5-style
detector.

The repository is organized around **auditable claims**. Every headline number
has a setting, protocol, evidence level, and source in
[`results/benchmark_summary.csv`](results/benchmark_summary.csv). Single-model,
protocol-optimized, routed-system, and post-processing results are deliberately
kept separate.

> **Naming and scope.** The locked thesis-facing model is **DARP-Net**. Its
> historical experiment identifier is `IA-DASR Round 2I+`; the earlier
> protocol-neutral stage-2 line is displayed as **DARP-Fusion**. These aliases
> remain in configs and result paths so the experiment trail stays reproducible.
> The detector is not an LLM or language-conditioned vision model. The optional
> [experiment-report copilot](docs/LLM_COPILOT.md) is a separate downstream tool
> and never changes training, fusion, predictions, or metrics.

## Highlights

- **Locked DARP-Net checkpoint:** `6.909 / 7.370 / 4.844%`
  MR-all/day/night with `98.42%` Recall-all on the KAIST Reasonable official
  reload/fused re-evaluation.
- **49.0% lower MR-all than 6-channel early fusion:** `13.535% → 6.909%` on
  the same dataset/evaluator. DARP-Net includes the explicitly documented
  KAIST ROI, night calibration, and bounded protocol-score settings.
- **Transformer-based multimodal reasoning:** multi-head, bidirectional
  deformable local cross-attention aligns RGB and thermal features at P3/P4/P5.
- **Reliability-aware fusion:** foreground agreement, modality conflict,
  local detail, ambiguity, and scene context modulate fusion conservatively.
- **Reproducibility-first release:** machine-readable metrics, result cards,
  model cards, source configs, checkpoint hashes, tests, CI, and explicit claim
  boundaries.
- **Cross-dataset boundary checks:** target-trained CVC-14 and LLVIP results are
  reported separately from KAIST rather than presented as zero-shot gains.

## Results

MR is log-average miss rate in percent; **lower is better**. The table uses the
repository's KAIST Reasonable evaluator, seed `0`, and input size `640`. The
first seven rows form the protocol-neutral ablation matrix; the final row is the
locked DARP-Net checkpoint with its KAIST-specific inference settings stated
explicitly.

| Model | Setting | MR-all ↓ | MR-day ↓ | MR-night ↓ | mAP@.5 ↑ |
|---|---|---:|---:|---:|---:|
| YOLOv5s RGB-only | single model | 39.174 | 31.675 | 53.602 | 51.072 |
| YOLOv5s LWIR-only | single model | 21.383 | 26.525 | 10.302 | 67.217 |
| Early Fusion, 6-channel | single model | 13.535 | 15.362 | 8.820 | 75.923 |
| DCAF | single model | 9.416 | 10.694 | 6.456 | **78.391** |
| DCAF + CDR | single model | 9.017 | 10.418 | 5.209 | 77.096 |
| DCAF + CDR + DSRE | single model | 9.055 | 10.164 | 6.358 | 77.473 |
| **DARP-Fusion** (`IA-DASR Stage2`) | protocol-neutral single model | **7.137** | **7.917** | **4.159** | 71.741 |
| **DARP-Net** (`Round 2I+`) | protocol-aware single checkpoint | **6.909** | **7.370** | 4.844 | 71.786 |

The following system-level results use routing or post-processing and therefore
must not be presented as DARP-Net single-checkpoint results:

| Model/system | Setting | MR-all ↓ | MR-day ↓ | MR-night ↓ | Boundary |
|---|---|---:|---:|---:|---|
| DARP-Net + IAER | routed system | 6.900 | 7.620 | **4.190** | expert routing; not one detector |
| DARP-Net + IAER + ECR | post-processing system | 6.841 | 7.429 | 4.276 | prediction-only calibrated reranking |

See [Benchmarks](docs/BENCHMARKS.md) for the protocol, exact decimals,
cross-dataset diagnostics, and why mAP and low-FPPI MR can move differently.

![DARP-Net benchmark comparison](assets/benchmark_mr.png)

## Architecture

![DARP-Net dual-stream architecture](assets/architecture.png)

For paired visible and thermal inputs, each backbone produces features at three
detection scales. At every scale:

1. **DCAF** samples locally offset features in both cross-modal directions to
   reduce sensitivity to RGB-thermal misalignment.
2. **CDR** separates consensus evidence from modality-specific detail instead
   of blindly averaging features.
3. **DSRE** estimates foreground support, conflict, uncertainty, and scene
   reliability, then applies bounded residual modulation.
4. **Ignore-aware objectness** removes ignored KAIST regions from negative
   supervision.
5. **PFH + BPSC + PUR** factorize protocol semantics, apply a weak bounded score
   residual at inference, and regularize duplicate/night risks during training.
   These terms are KAIST-specific and are disabled in portable variants.

More detail, equations, and ownership boundaries are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Qualitative comparison

The montage compares RGB + ground truth, a reproduced DeformCAT-style baseline,
DARP-Fusion, and DARP-Net on selected KAIST scenes. It is qualitative evidence, not a
substitute for the complete test-set metrics above.

![Qualitative RGB-T detections](assets/qualitative_montage.png)

## Quick start

### 1. Install

```bash
git clone https://github.com/drxadqz/kaist_yolo.git
cd kaist_yolo
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch build for your platform before the remaining
dependencies when GPU training is required.

### 2. Prepare KAIST

KAIST images and training labels are not redistributed. Obtain them from the
[official KAIST project](https://github.com/SoonminHwang/rgbt-ped-detection),
then build the strict local view. The public evaluator annotation JSON and the
selected qualitative montage retained in this repository are treated as KAIST
dataset materials under CC BY-NC-SA 4.0; see
[`docs/DATASET.md`](docs/DATASET.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

```bash
python src/setup_local_kaist.py \
  --image-source-root /path/to/kaist_yolo_images \
  --train-annotation-root /path/to/sanitized_train_annotations \
  --test-annotation-root /path/to/official_test_annotations \
  --target-root datasets/KAIST_local_strict
```

The expected directory contract is documented in
[`docs/DATASET.md`](docs/DATASET.md). The release protocol uses 2,252 ordered
test pairs, 3,390 positive person instances, and 864 ignored regions.

### 3. Validate DARP-Net and its protocol-neutral predecessor

This CPU smoke path requires no dataset or checkpoint:

```bash
python src/models/yolo_test.py \
  --cfg configs/models/ia_dasr_round2i_plus.yaml \
  --device cpu

python src/models/yolo_test.py \
  --cfg configs/models/ia_dasr_formal.yaml \
  --device cpu
```

The filenames keep the historical experiment identifiers. See
[`docs/NAMING.md`](docs/NAMING.md) for the reproducible name mapping.

### 4. Rebuild the DARP-Fusion starting point

The validated DARP-Fusion run was a 10-epoch continuation from a stage-1
checkpoint—not a 10-epoch from-scratch run. Place a compatible stage-1 file at
`checkpoints/ia_dasr_stage1_best.pt`:

```bash
python src/train.py \
  --weights checkpoints/ia_dasr_stage1_best.pt \
  --cfg configs/models/ia_dasr_formal.yaml \
  --data configs/datasets/kaist.example.yaml \
  --hyp configs/experiments/hyp.formal_stage2.yaml \
  --epochs 10 \
  --batch-size 1 \
  --img-size 640 640 \
  --workers 0 \
  --kaist-day-roi-filter \
  --kaist-ignore-aware-obj \
  --drr-aux-weight 0.03
```

The exact historical stage-1 checkpoint is not yet distributed, so the command
reconstructs the recorded predecessor recipe but cannot currently guarantee the
locked DARP-Net checkpoint from a clean clone. DARP-Net's sanitized run options,
hyperparameters, source hashes, and inference settings are archived under
[`results/protocol_aware_best/`](results/protocol_aware_best/).

### 5. Evaluate

Model weights are intentionally not committed as Git blobs. Place the verified
DARP-Net checkpoint at `checkpoints/ia_dasr_round2i_plus_best.pt`; the filename
keeps the historical run identifier. See
[`checkpoints/README.md`](checkpoints/README.md) for the release policy.

```bash
python src/test.py \
  --weights checkpoints/ia_dasr_round2i_plus_best.pt \
  --data configs/datasets/kaist.example.yaml \
  --img-size 640 \
  --batch-size 2 \
  --device 0 \
  --kaist-day-roi-filter \
  --kaist-night-score-calib \
  --pcsf-apply-in-inference \
  --pcsf-semantic-alpha 0.05 \
  --pcsf-factor-min 0.95 \
  --pcsf-factor-max 1.06 \
  --pcsf-semantic-center 0.25 \
  --pcsf-score-beta 4.0
```

Full commands, hashes, and protocol-aware variants are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Auxiliary evidence reporting (optional)

The offline path reads only the canonical CSV and produces an evidence packet or
prompt—no API key and no network call:

```bash
python scripts/experiment_copilot.py \
  --input-csv results/benchmark_summary.csv \
  --format context-json \
  --output outputs/experiment_context.json
```

An optional structured-output path can use the OpenAI Responses API:

```bash
python -m pip install -r requirements-llm.txt
# Set OPENAI_API_KEY in the process environment.
python scripts/experiment_copilot.py \
  --call-openai \
  --model gpt-5.6-luna \
  --output outputs/experiment_report.json
```

This utility is not part of the detector. It is guarded against converting
routed or post-processed numbers into single-checkpoint claims. See
[`docs/LLM_COPILOT.md`](docs/LLM_COPILOT.md).

## Repository map

```text
.
├── assets/                    # selected architecture, benchmark, qualitative assets
├── checkpoints/               # policy and hashes; no weights in ordinary Git
├── configs/                   # portable dataset, model, and experiment recipes
├── docs/                      # method, protocol, reproduction, résumé guidance
├── model_cards/               # formal and protocol-aware model boundaries
├── results/
│   ├── benchmark_summary.csv  # canonical machine-readable claim source
│   ├── cards/                 # experiment cards
│   ├── formal_mainline/       # source re-evaluation table
│   ├── protocol_aware_best/   # reload summary, metadata, plots; no weights
│   ├── system/                # sanitized routed-system summary
│   └── postprocess/           # calibrated reranking summaries
├── scripts/                   # validation, asset generation, experiment copilot
├── src/                       # dual-stream detector, training, evaluation
└── tests/                     # offline release-integrity and copilot tests
```

The original 01–06 YOLO/MOT prototype is preserved in Git history and summarized
in [`legacy/README.md`](legacy/README.md). Its unverified fusion/tracking outputs
are not used as headline results.

## Relevance to multimodal and foundation-model roles

This project demonstrates capabilities that transfer to VLM and multimodal
foundation-model work without relabeling a detector as an LLM:

- Transformer-based cross-modal attention and representation alignment;
- learned modality reliability and conditional routing;
- structured supervision under noisy/ignored labels;
- protocol-aware evaluation and failure taxonomy;
- reproducible PyTorch experimentation and claim provenance;
- evidence-grounded experiment reporting, with optional structured LLM output.

For direct LLM post-training evidence, see
[SP-WFM](https://github.com/drxadqz/SP-WFM), which covers Qwen2.5, QLoRA,
SFT/GRPO, and learned expert routing. The
[SensorLedger3D reading log](https://github.com/drxadqz/sensorledger3d-reading-log)
tracks VFM/VLM/LLM/VLA and world-model research.

## Limitations

- The primary benchmark is KAIST; the locked DARP-Net result includes
  KAIST-specific calibration and should not be generalized to arbitrary
  multispectral datasets.
- FLIR zero-shot transfer is weak; LLVIP zero-shot is only a diagnostic.
- Checkpoints are not yet distributed from this repository. Configs, hashes, and
  selected evidence are included so a future Release asset can be verified.
- The selected qualitative montage contains Chinese labels inherited from the
  thesis asset; all quantitative evidence and documentation are available in
  English.
- Removing historical weight files from the current tree does not shrink old Git
  history. History rewriting is intentionally out of scope for this release.

## License and attribution

The code is released under [AGPL-3.0](LICENSE). It builds on the
[DeformCAT](https://github.com/jiongger/DeformCAT) and
[YOLOv5](https://github.com/ultralytics/yolov5) code families. Upstream
ownership and this repository's original contributions are detailed in
[`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).

If this repository helps your work, please cite [`CITATION.cff`](CITATION.cff).
