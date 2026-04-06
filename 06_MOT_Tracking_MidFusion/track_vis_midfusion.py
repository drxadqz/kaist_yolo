import os
import sys
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config_midfusion import (
    ANNOTATION_ROOT,
    FPS,
    OUT_GT_FILE,
    OUT_TRACKER_FILE,
    REFINER_CKPT,
    REFINE_ALPHA,
    REFINE_SCORE_THRESH,
    RUN_DIR,
    SAVE_IMAGE_DIR,
    SAVE_VIDEO_PATH,
    SEQ_IR_DIR,
    SEQ_VISIBLE_DIR,
    TARGET_CLASSES,
    TRACKER_KWARGS,
    TRACKER_TYPE,
)


# project imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "03_Modal_Fusion"))
sys.path.insert(0, os.path.join(REPO_ROOT, "04_MOT_Tracking"))
sys.path.insert(0, os.path.join(REPO_ROOT, "05_Mid_Fusion_Research"))

from late_fusion import late_fusion_single_image  # noqa: E402
from mid_fusion_refiner import MidFusionRefiner  # noqa: E402
from sort_tracker import SORT  # noqa: E402
from bytetrack_tracker import BYTETracker  # noqa: E402


BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
TITLE_BAR_HEIGHT = 60


def make_video_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def draw_tracking_result(frame, tracks):
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        text = f"ID:{track_id} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_y = max(y1 - 10, text_h + 10)
        cv2.rectangle(canvas, (x1, text_y - text_h - 6), (x1 + text_w + 4, text_y), BOX_COLOR, -1)
        cv2.putText(canvas, text, (x1 + 2, text_y - 4), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)
    return canvas


def load_kaist_annotations(annot_dir: str, img_names: List[str], target_classes=None):
    if target_classes is None:
        target_classes = {"person"}
    gt_data: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    labeled_frames: Set[int] = set()
    next_gt_id = 1
    if not os.path.isdir(annot_dir):
        return gt_data, labeled_frames
    for frame_idx, img_name in enumerate(img_names):
        frame_id = frame_idx + 1
        stem, _ = os.path.splitext(img_name)
        ann_path = os.path.join(annot_dir, stem + ".txt")
        if not os.path.exists(ann_path):
            continue
        objs = []
        with open(ann_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if (not line) or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_name = parts[0].lower()
            if cls_name not in target_classes:
                continue
            try:
                x1, y1, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            objs.append((next_gt_id, x1, y1, w, h))
            next_gt_id += 1
        if objs:
            gt_data[frame_id] = objs
            labeled_frames.add(frame_id)
    return gt_data, labeled_frames


def clamp_xyxy(box: np.ndarray, w: int, h: int):
    x1, y1, x2, y2 = box[:4]
    x1 = int(max(0, min(round(float(x1)), w - 1)))
    y1 = int(max(0, min(round(float(y1)), h - 1)))
    x2 = int(max(x1 + 1, min(round(float(x2)), w)))
    y2 = int(max(y1 + 1, min(round(float(y2)), h)))
    return x1, y1, x2, y2


def crop_patch(img, xyxy, patch_size):
    x1, y1, x2, y2 = xyxy
    patch = img[y1:y2, x1:x2]
    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0)


class MidFusionScoreRefiner:
    def __init__(self, ckpt_path: str, alpha: float = 0.6, score_thresh: float = 0.25):
        self.alpha = alpha
        self.score_thresh = score_thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Refiner checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        base_ch = int(ckpt.get("base_ch", 32))
        self.patch_size = int(ckpt.get("patch_size", 96))
        self.model = MidFusionRefiner(base_ch=base_ch).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    def refine(self, late_boxes: np.ndarray, vis_img: np.ndarray, ir_img: np.ndarray) -> np.ndarray:
        late_boxes = np.asarray(late_boxes, dtype=np.float32)
        if late_boxes.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        h, w = vis_img.shape[:2]
        refined = []
        with torch.no_grad():
            for det in late_boxes:
                x1, y1, x2, y2, score, cls_id = det
                xyxy = clamp_xyxy(det, w, h)
                vis_patch = crop_patch(vis_img, xyxy, self.patch_size).to(self.device)
                ir_patch = crop_patch(ir_img, xyxy, self.patch_size).to(self.device)
                ref_score = torch.sigmoid(self.model(vis_patch, ir_patch)).item()
                new_score = self.alpha * float(score) + (1.0 - self.alpha) * ref_score
                if new_score >= self.score_thresh:
                    refined.append([x1, y1, x2, y2, new_score, cls_id])
        if len(refined) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(refined, dtype=np.float32)


def build_tracker():
    if TRACKER_TYPE.lower() == "sort":
        return SORT(**TRACKER_KWARGS)
    if TRACKER_TYPE.lower() == "bytetrack":
        return BYTETracker(**TRACKER_KWARGS)
    raise ValueError(f"Unsupported TRACKER_TYPE: {TRACKER_TYPE}")


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

    tracker = build_tracker()
    refiner = MidFusionScoreRefiner(REFINER_CKPT, alpha=REFINE_ALPHA, score_thresh=REFINE_SCORE_THRESH)

    seq_parent_dir = os.path.dirname(SEQ_VISIBLE_DIR)
    set_v_dir = os.path.basename(seq_parent_dir)
    set_dir = os.path.basename(os.path.dirname(seq_parent_dir))

    img_names = sorted([f for f in os.listdir(SEQ_VISIBLE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if len(img_names) == 0:
        print("No sequence images found.")
        return
    print(f"Found frames: {len(img_names)}")

    gt_annot_dir = os.path.join(ANNOTATION_ROOT, set_dir, set_v_dir, "visible")
    gt_data, labeled_frames = load_kaist_annotations(gt_annot_dir, img_names, target_classes=TARGET_CLASSES)
    print(f"GT labeled frames: {len(labeled_frames)} / {len(img_names)}")

    first_vis = cv2.imread(os.path.join(SEQ_VISIBLE_DIR, img_names[0]))
    first_ir = cv2.imread(os.path.join(SEQ_IR_DIR, img_names[0]))
    if first_vis is None or first_ir is None:
        print("Failed to read first frame.")
        return
    h, w = first_vis.shape[:2]
    total_width = w * 2
    total_height = h + TITLE_BAR_HEIGHT
    video_writer = make_video_writer(SAVE_VIDEO_PATH, FPS, total_width, total_height)

    gt_f = open(OUT_GT_FILE, "w", encoding="utf-8")
    trk_f = open(OUT_TRACKER_FILE, "w", encoding="utf-8")

    failed_frames = []
    for frame_idx, img_name in enumerate(tqdm(img_names, desc="MidFusion tracking", total=len(img_names))):
        frame_id = frame_idx + 1
        vis_path = os.path.join(SEQ_VISIBLE_DIR, img_name)
        ir_path = os.path.join(SEQ_IR_DIR, img_name)

        if frame_id in gt_data:
            for gt_id, x1, y1, w_gt, h_gt in gt_data[frame_id]:
                gt_f.write(f"{frame_id},{int(gt_id)},{x1:.1f},{y1:.1f},{w_gt:.1f},{h_gt:.1f},1.0,-1,-1,-1\n")

        try:
            late_boxes, _, _, vis_frame = late_fusion_single_image(vis_path, ir_path)
            ir_frame = cv2.imread(ir_path)
            if ir_frame is None:
                raise ValueError("IR frame read failed")
            if ir_frame.ndim == 2:
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            ir_frame = cv2.resize(ir_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            refined_boxes = refiner.refine(late_boxes, vis_frame, ir_frame)
        except Exception as e:
            failed_frames.append(f"{frame_id}:{img_name}::{e}")
            continue

        dets = refined_boxes[:, :5] if len(refined_boxes) > 0 else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(dets)

        for t in tracks:
            x1, y1, x2, y2, tid, conf = t
            trk_f.write(f"{frame_id},{int(tid)},{x1:.1f},{y1:.1f},{(x2-x1):.1f},{(y2-y1):.1f},{conf:.3f},-1,-1,-1\n")

        vis_with_track = draw_tracking_result(vis_frame, tracks)
        ir_with_track = draw_tracking_result(ir_frame, tracks)
        title_bar = np.zeros((TITLE_BAR_HEIGHT, total_width, 3), dtype=np.uint8)
        cv2.putText(title_bar, "Visible", (w // 2 - 60, TITLE_BAR_HEIGHT // 2 + 10), FONT, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(title_bar, "Infrared", (w + w // 2 - 60, TITLE_BAR_HEIGHT // 2 + 10), FONT, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        final_frame = cv2.vconcat([title_bar, cv2.hconcat([vis_with_track, ir_with_track])])
        video_writer.write(final_frame)
        cv2.imwrite(os.path.join(SAVE_IMAGE_DIR, img_name), final_frame)

    gt_f.close()
    trk_f.close()
    video_writer.release()
    cv2.destroyAllWindows()

    print("=" * 70)
    print("MidFusion sequence tracking done")
    print(f"Video: {os.path.abspath(SAVE_VIDEO_PATH)}")
    print(f"Frames: {os.path.abspath(SAVE_IMAGE_DIR)}")
    print(f"GT: {os.path.abspath(OUT_GT_FILE)}")
    print(f"Tracker: {os.path.abspath(OUT_TRACKER_FILE)}")
    print(f"Failed: {len(failed_frames)}")
    if failed_frames:
        print("First failed:", failed_frames[0])
    print("=" * 70)


if __name__ == "__main__":
    main()

