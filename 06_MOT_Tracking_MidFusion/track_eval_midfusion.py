import argparse
import os
from typing import Dict, List, Tuple

import motmetrics as mm
import numpy as np

from config_midfusion import IOU_THRESH_EVAL, OUT_GT_FILE, OUT_TRACKER_FILE


def _xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_distance_matrix(
    gt_boxes: List[Tuple[float, float, float, float]],
    trk_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float,
) -> np.ndarray:
    mat = np.full((len(gt_boxes), len(trk_boxes)), np.nan, dtype=np.float64)
    for i, g in enumerate(gt_boxes):
        for j, t in enumerate(trk_boxes):
            iou = _iou_xywh(g, t)
            if iou >= iou_threshold:
                mat[i, j] = 1.0 - iou
    return mat


def parse_mot_file(file_path: str) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            if w <= 0 or h <= 0:
                continue
            if frame_id not in data:
                data[frame_id] = {}
            data[frame_id][obj_id] = (x, y, w, h)
    return data


def evaluate_mot(gt_file: str, tracker_file: str, iou_threshold: float, only_gt_frames: bool = True):
    gt_data = parse_mot_file(gt_file)
    tracker_data = parse_mot_file(tracker_file)

    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted(gt_data.keys()) if only_gt_frames else sorted(set(gt_data.keys()) | set(tracker_data.keys()))
    for frame in frames:
        gt_ids = list(gt_data.get(frame, {}).keys())
        trk_ids = list(tracker_data.get(frame, {}).keys())
        gt_boxes = list(gt_data.get(frame, {}).values())
        trk_boxes = list(tracker_data.get(frame, {}).values())
        if len(gt_boxes) == 0 or len(trk_boxes) == 0:
            dist_matrix = np.empty((len(gt_boxes), len(trk_boxes)))
        else:
            dist_matrix = _iou_distance_matrix(gt_boxes, trk_boxes, iou_threshold)
        acc.update(gt_ids, trk_ids, dist_matrix, frameid=frame)

    mh = mm.metrics.create()
    metrics = [
        "num_frames", "num_objects", "num_predictions", "recall", "precision",
        "mota", "motp", "idf1", "idp", "idr", "num_switches", "num_false_positives",
        "num_misses", "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations",
    ]
    row = mh.compute(acc, metrics=metrics, name="tracker").loc["tracker"]

    results = {}
    percent_keys = {"recall", "precision", "mota", "idf1", "idp", "idr"}
    for k in metrics:
        v = float(row[k]) if np.isfinite(row[k]) else 0.0
        if k in percent_keys:
            v *= 100.0
        if k == "motp":
            v = (1.0 - v) * 100.0
        results[k] = v

    gt_ids = [obj_id for frame_objs in gt_data.values() for obj_id in frame_objs.keys()]
    repeated_gt_ids = sum(1 for cnt in np.unique(gt_ids, return_counts=True)[1] if cnt > 1) if gt_ids else 0
    has_temporal_gt_ids = repeated_gt_ids > 0
    return results, has_temporal_gt_ids, len(gt_data), len(tracker_data)


def print_report(results: Dict[str, float], has_temporal_gt_ids: bool):
    print("\n" + "=" * 72)
    print("MidFusion Tracking Evaluation (MOTChallenge-style)")
    print("=" * 72)
    ordered = [
        ("num_frames", "Frames"), ("num_objects", "GT Objects"), ("num_predictions", "Tracker Outputs"),
        ("recall", "Recall(%)"), ("precision", "Precision(%)"), ("mota", "MOTA(%)"), ("motp", "MOTP(%)"),
        ("idf1", "IDF1(%)"), ("idp", "IDP(%)"), ("idr", "IDR(%)"), ("num_switches", "IDSW"),
        ("num_false_positives", "FP"), ("num_misses", "FN"), ("mostly_tracked", "MT"),
        ("partially_tracked", "PT"), ("mostly_lost", "ML"), ("num_fragmentations", "Frag"),
    ]
    id_related = {"idf1", "idp", "idr", "num_switches", "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations"}
    count_keys = {"num_frames", "num_objects", "num_predictions", "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations"}
    for key, title in ordered:
        if (not has_temporal_gt_ids) and (key in id_related):
            print(f"{title:20}: N/A (requires temporal GT IDs)")
            continue
        value = results[key]
        if key in count_keys:
            print(f"{title:20}: {int(round(value))}")
        else:
            print(f"{title:20}: {value:.2f}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, default=OUT_GT_FILE)
    parser.add_argument("--tracker", type=str, default=OUT_TRACKER_FILE)
    parser.add_argument("--iou", type=float, default=IOU_THRESH_EVAL)
    parser.add_argument("--all-frames", action="store_true")
    args = parser.parse_args()

    results, has_temporal_gt_ids, n_gt_frames, n_trk_frames = evaluate_mot(
        args.gt, args.tracker, iou_threshold=args.iou, only_gt_frames=not args.all_frames
    )
    print(f"GT frames          : {n_gt_frames}")
    print(f"Tracker frames     : {n_trk_frames}")
    print(f"Eval frame policy  : {'union(gt,tracker)' if args.all_frames else 'gt-only'}")
    if not has_temporal_gt_ids:
        print("[Warning] GT IDs are unique per frame; ID metrics are not valid.")
    print_report(results, has_temporal_gt_ids)


if __name__ == "__main__":
    main()

