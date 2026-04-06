# kaist_yolo

KAIST multispectral pedestrian detection and tracking project.

This repository is organized as two parallel experiment lines:

- **Baseline line** (single-modal + late fusion + tracking): `01` -> `04`
- **Mid-fusion line** (isolated for paper ablation): `05` -> `06`

## Directory map

- `01_Data_Prepare/`: data cleaning, organizing, label conversion scripts
- `02_YOLO_Train/`: visible and lwir YOLO training scripts/configs
- `03_Modal_Fusion/`: late-fusion inference and detection evaluation
- `04_MOT_Tracking/`: baseline tracking video generation + MOT evaluation
- `05_Mid_Fusion_Research/`: mid-level fusion refiner model training/inference
- `06_MOT_Tracking_MidFusion/`: isolated mid-fusion tracking and MOT evaluation

## Recommended reproduction order

1) Data prep and label checks (`01_Data_Prepare`)
2) Train visible/lwir detectors (`02_YOLO_Train`)
3) Run late-fusion detection evaluation (`03_Modal_Fusion`)
4) Baseline tracking video + MOT metrics (`04_MOT_Tracking`)
5) Train mid-fusion refiner (`05_Mid_Fusion_Research`)
6) Mid-fusion tracking video + MOT metrics (`06_MOT_Tracking_MidFusion`)

## Isolation principle for paper experiments

- Baseline outputs are kept in `04_MOT_Tracking`.
- Mid-fusion outputs are kept in `06_MOT_Tracking_MidFusion/outputs_midfusion`.
- Do not overwrite cross-line artifacts to keep ablation tables reproducible.
