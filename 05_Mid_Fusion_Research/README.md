# Mid-Fusion Baseline (KAIST)

This folder provides a practical and lightweight **mid-level fusion baseline** for KAIST visible/IR pedestrian detection research.

It does not replace your current YOLO detector training pipeline. Instead, it adds a dual-stream feature module that can be trained quickly and then plugged into your current late-fusion outputs for score refinement.

## Why this baseline

- Keeps your existing visible/IR YOLO baselines unchanged
- Adds learnable cross-modal fusion at feature level
- Easy to ablate against:
  - visible-only
  - lwir-only
  - late fusion
  - late fusion + mid-fusion refiner (this module)

## File overview

- `mid_fusion_refiner.py`: dual-stream CNN with mid-level feature fusion
- `dataset_refiner.py`: KAIST paired sample builder (positive from GT, negative from random windows)
- `train_refiner.py`: training entry for the refiner
- `infer_mid_fusion.py`: inference adapter to refine scores from `03_Modal_Fusion/late_fusion.py`

## Expected dataset structure

Uses the same KAIST structure you already use:

- `E:/KAIST_Dataset/images/<set>/<video>/visible/*.jpg`
- `E:/KAIST_Dataset/images/<set>/<video>/lwir/*.jpg`
- `E:/KAIST_Dataset/annotations/<set>/<video>/visible/*.txt`

## Quick start

1) Train refiner:

```bash
python E:/kaist_yolo/05_Mid_Fusion_Research/train_refiner.py
```

2) Run one-image test with refined fusion:

```bash
python E:/kaist_yolo/05_Mid_Fusion_Research/infer_mid_fusion.py
```

## Notes for paper writing

- This module provides **feature-level cross-modal fusion** over paired ROI patches.
- Report it as a "mid-level fusion refinement head on top of dual YOLO detections".
- For fair comparison, keep detector weights and NMS settings unchanged.

