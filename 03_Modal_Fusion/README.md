# 03_Modal_Fusion

Late-fusion detection stage and visualization.

## Main files

- `late_fusion.py`: core late-fusion inference
- `fusion_eval.py`: detection-level metric comparison (visible / lwir / late fusion)
- `fusion_vis.py`: qualitative image visualization
- `README_fusion_vis.md`: extra visualization usage details

## Paper role

This stage provides your baseline fusion method before introducing mid-level fusion.

## Typical run order

```bash
python E:/kaist_yolo/03_Modal_Fusion/fusion_eval.py
python E:/kaist_yolo/03_Modal_Fusion/fusion_vis.py
```

## Notes

- Keep detector checkpoints and confidence thresholds consistent across all compared methods.
- The outputs here are detection-focused, not MOT temporal metrics.

