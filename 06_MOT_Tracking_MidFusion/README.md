# 06_MOT_Tracking_MidFusion

This folder is the **isolated mid-fusion tracking experiment line** for paper-ready comparison.

It is intentionally separated from `04_MOT_Tracking` so you can run baseline and mid-fusion pipelines independently without overwriting outputs.

## Files

- `config_midfusion.py`: all paths and run settings
- `track_vis_midfusion.py`: generate tracking video + MOT txt (`gt.txt`, `tracker.txt`)
- `track_eval_midfusion.py`: compute MOT metrics for this line only

## Run order

1) Make sure refiner checkpoint exists:

```bash
python E:/kaist_yolo/05_Mid_Fusion_Research/train_refiner.py
```

2) Generate MidFusion sequence outputs:

```bash
python E:/kaist_yolo/06_MOT_Tracking_MidFusion/track_vis_midfusion.py
```

3) Evaluate MidFusion MOT metrics:

```bash
python E:/kaist_yolo/06_MOT_Tracking_MidFusion/track_eval_midfusion.py
```

## Output isolation

All outputs are stored under:

- `E:/kaist_yolo/06_MOT_Tracking_MidFusion/outputs_midfusion/<RUN_NAME>/`

This prevents conflicts with baseline outputs in `04_MOT_Tracking`.

