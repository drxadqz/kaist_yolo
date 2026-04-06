# 02_YOLO_Train

Detector training stage (single-modality baselines).

## Main scripts

- `train_visible.py`: visible branch training/resume
- `train_lwir.py`: lwir branch training/resume
- `kaist_visible.yaml`: visible dataset config
- `kaist_lwir.yaml`: lwir dataset config

## Baseline goal

Train two independent YOLO detectors:

- Visible baseline (RGB)
- Infrared baseline (LWIR)

These two checkpoints are reused by late-fusion and tracking stages.

## Run

```bash
python E:/kaist_yolo/02_YOLO_Train/train_visible.py
python E:/kaist_yolo/02_YOLO_Train/train_lwir.py
```

## Outputs

Typical output roots:

- `02_YOLO_Train/runs/visible_baseline/.../weights/best.pt`
- `02_YOLO_Train/runs/ir_baseline/.../weights/best.pt`

Keep these paths fixed for fair ablation in the paper.

