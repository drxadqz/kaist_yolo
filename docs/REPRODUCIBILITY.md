# Reproducibility guide

## What is included

- detector source and model graphs;
- portable dataset/config templates;
- formal experiment recipe;
- full-precision re-evaluation summary;
- result and model cards;
- selected training metadata and plots for Round 2I+;
- SHA-256 for the archived inference checkpoint;
- release-integrity and LLM-copilot tests.

Datasets, full prediction dumps, and model weights are not committed.

## Environment

The original research code descends from an older YOLOv5/DeformCAT stack. The
curated requirements set a modern lower bound, but CUDA builds must be selected
for the target machine:

```bash
python --version
python -m venv .venv
# Activate .venv.
python -m pip install --upgrade pip
# Install torch and torchvision from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

Record these values with any new benchmark:

```bash
python -VV
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m pip freeze
git rev-parse HEAD
```

## Release validation

The offline validation path does not import PyTorch:

```bash
python scripts/validate_release.py
python -m unittest discover -s tests -v
python scripts/experiment_copilot.py --format context-json
```

CPU graph smoke tests:

```bash
python src/models/yolo_test.py \
  --cfg configs/models/ia_dasr_formal.yaml \
  --device cpu

python src/models/yolo_test.py \
  --cfg configs/models/ia_dasr_round2i_plus.yaml \
  --device cpu
```

The curated release was smoke-tested with PyTorch `2.11.0+cpu`. Both graph
variants completed construction and a paired `1 × 3 × 128 × 128` forward pass:
the formal model reported 32.45M parameters and the factorized Round 2I+ model
reported 32.46M. This validates graph/source compatibility, not checkpoint
accuracy or full-dataset reproduction.

## Formal training

The validated formal run was a stage-2 continuation. Its sanitized archived
options are in
[`results/formal_mainline/opt.yaml`](../results/formal_mainline/opt.yaml).
The initialization source was the best checkpoint from an earlier
`kaist_dasr_iao_strict_v2` run. That exact stage-1 checkpoint and hash are not
currently distributed, so a clean clone cannot yet reproduce the headline model
from scratch. This is a release gap, not something the command hides.

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
  --project runs/train \
  --name ia_dasr_formal_s0 \
  --kaist-day-roi-filter \
  --kaist-ignore-aware-obj \
  --drr-aux-weight 0.03 \
  --drr-aux-warmup 2.0 \
  --drr-aux-fg-gain 1.0 \
  --drr-aux-conflict-gain 0.35 \
  --drr-aux-rgb-bg-gain 0.15 \
  --drr-aux-day-gain 0.25 \
  --drr-aux-box-pad 0.08 \
  --drr-aux-day-thr 0.28 \
  --drr-aux-day-tau 0.08
```

The historical formal run used seed 0, but older training entrypoints do not
expose a single `--seed` flag consistently. Record the actual seed
initialization and deterministic backend state when re-running. A from-scratch
experiment can pass an empty `--weights` value, but it is a new experiment and
must not be expected to reproduce the 10-epoch continuation result.

## Formal evaluation

```bash
python src/test.py \
  --weights checkpoints/ia_dasr_stage2_e10_best.pt \
  --data configs/datasets/kaist.example.yaml \
  --img-size 640 \
  --batch-size 2 \
  --device 0 \
  --project runs/test \
  --name ia_dasr_formal_eval \
  --exist-ok \
  --kaist-day-roi-filter
```

Expected repository re-evaluation:

```text
MR-all  = 0.0713722247
MR-day  = 0.0791671205
MR-night= 0.0415905875
mAP@.5  = 0.7174111607
mAP@.5:.95 = 0.3537565045
Recall-all = 0.9835051546
```

Floating-point, library, hardware, and NMS differences can cause small
variation. Do not overwrite the canonical row unless the checkpoint hash,
dataset manifest, source commit, environment, and exact command are stored.

## Round 2I+ evaluation

Place the verified file at `checkpoints/ia_dasr_round2i_plus_best.pt`, then run:

```bash
python src/test.py \
  --weights checkpoints/ia_dasr_round2i_plus_best.pt \
  --data configs/datasets/kaist.example.yaml \
  --img-size 640 \
  --batch-size 2 \
  --device 0 \
  --project runs/test \
  --name ia_dasr_round2i_plus_eval \
  --exist-ok \
  --kaist-day-roi-filter \
  --kaist-night-score-calib \
  --pcsf-apply-in-inference \
  --pcsf-semantic-alpha 0.05 \
  --pcsf-factor-min 0.95 \
  --pcsf-factor-max 1.06 \
  --pcsf-semantic-center 0.25 \
  --pcsf-score-beta 4.0
```

Expected official reload/fused result:

```text
MR-all / day / night = 0.06909 / 0.07370 / 0.04844
```

The main inference checkpoint SHA-256 is:

```text
626D279F6C0D547F3559B3483009D9C34F860D1D9A1D970F9950F38BAC03B7C6
```

## Result update protocol

For every new result:

1. create a unique experiment ID;
2. record dataset manifest and protocol;
3. save config, command, seed, commit, environment, and checkpoint SHA-256;
4. classify the setting as `single_model`, `protocol_optimized`, `system`, or
   `postprocess`;
5. add a result card;
6. append a canonical CSV row;
7. run `python scripts/validate_release.py`;
8. rebuild the README chart with `python scripts/build_readme_assets.py`.

Never select one row for MR and another row for mAP without explicitly saying so.
