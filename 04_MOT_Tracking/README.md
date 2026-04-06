# 04_MOT_Tracking

Baseline tracking experiment line (late fusion + tracker).

## Main files

- `track_vis.py`: sequence tracking visualization and MOT txt generation
- `track_eval.py`: MOT metric evaluation
- `sort_tracker.py`: SORT implementation
- `bytetrack_tracker.py`: ByteTrack implementation

## Baseline run

1) Generate video + MOT files:

```bash
python E:/kaist_yolo/04_MOT_Tracking/track_vis.py
```

2) Evaluate:

```bash
python E:/kaist_yolo/04_MOT_Tracking/track_eval.py
```

## Key outputs

- `kaist_continuous_sequence_tracking.mp4`
- `continuous_seq_frames/`
- `gt.txt`
- `tracker.txt`

## Important metric note

If GT has no temporal identity IDs, identity-consistency metrics (IDF1/IDSW/MT/ML/Frag) are reported as N/A.

