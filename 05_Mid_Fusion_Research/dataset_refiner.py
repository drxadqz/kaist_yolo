import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _clamp_box_xywh(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box
    x1 = max(0, min(int(round(x)), w - 1))
    y1 = max(0, min(int(round(y)), h - 1))
    x2 = max(x1 + 1, min(int(round(x + bw)), w))
    y2 = max(y1 + 1, min(int(round(y + bh)), h))
    return x1, y1, x2, y2


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _read_kaist_ann(ann_path: str, target_classes=None) -> List[Tuple[float, float, float, float]]:
    if target_classes is None:
        target_classes = {"person"}
    if not os.path.exists(ann_path):
        return []
    boxes = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
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
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, w, h))
    return boxes


class KAISTRefinerDataset(Dataset):
    """
    Build ROI patch pairs:
      positive: GT boxes
      negative: random boxes with IoU < neg_iou_thresh w.r.t. all GT boxes
    """
    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        patch_size: int = 96,
        negatives_per_image: int = 2,
        neg_iou_thresh: float = 0.1,
        target_classes=None,
        max_samples: int = 20000,
        seed: int = 42,
    ):
        super().__init__()
        if target_classes is None:
            target_classes = {"person"}
        self.patch_size = patch_size
        self.negatives_per_image = negatives_per_image
        self.neg_iou_thresh = neg_iou_thresh
        self.target_classes = target_classes
        self.rng = random.Random(seed)

        self.samples = []
        self._build_index(image_root, annotation_root, max_samples=max_samples)

    def _build_index(self, image_root: str, annotation_root: str, max_samples: int) -> None:
        # Walk annotation folders and map to paired images by same relative path/name
        for set_name in sorted(os.listdir(annotation_root)):
            set_dir = os.path.join(annotation_root, set_name)
            if not os.path.isdir(set_dir):
                continue
            for vid_name in sorted(os.listdir(set_dir)):
                ann_vis_dir = os.path.join(set_dir, vid_name, "visible")
                if not os.path.isdir(ann_vis_dir):
                    continue

                vis_img_dir = os.path.join(image_root, set_name, vid_name, "visible")
                ir_img_dir = os.path.join(image_root, set_name, vid_name, "lwir")
                if (not os.path.isdir(vis_img_dir)) or (not os.path.isdir(ir_img_dir)):
                    continue

                for ann_file in sorted(os.listdir(ann_vis_dir)):
                    if not ann_file.lower().endswith(".txt"):
                        continue
                    stem = os.path.splitext(ann_file)[0]
                    vis_path = os.path.join(vis_img_dir, stem + ".jpg")
                    ir_path = os.path.join(ir_img_dir, stem + ".jpg")
                    if (not os.path.exists(vis_path)) or (not os.path.exists(ir_path)):
                        # fallback png
                        vis_path = os.path.join(vis_img_dir, stem + ".png")
                        ir_path = os.path.join(ir_img_dir, stem + ".png")
                        if (not os.path.exists(vis_path)) or (not os.path.exists(ir_path)):
                            continue

                    gt_xywh = _read_kaist_ann(os.path.join(ann_vis_dir, ann_file), self.target_classes)
                    if len(gt_xywh) == 0:
                        continue

                    # positives
                    for box in gt_xywh:
                        self.samples.append((vis_path, ir_path, box, 1.0))
                        if len(self.samples) >= max_samples:
                            return

                    # negatives from random windows
                    vis = cv2.imread(vis_path)
                    if vis is None:
                        continue
                    h, w = vis.shape[:2]
                    gt_xyxy = [_clamp_box_xywh(b, w, h) for b in gt_xywh]
                    for _ in range(self.negatives_per_image):
                        bw = self.rng.randint(24, max(25, min(128, w // 3)))
                        bh = self.rng.randint(48, max(49, min(192, h // 2)))
                        x1 = self.rng.randint(0, max(0, w - bw))
                        y1 = self.rng.randint(0, max(0, h - bh))
                        neg_box = (x1, y1, x1 + bw, y1 + bh)
                        max_iou = max((_iou_xyxy(neg_box, g) for g in gt_xyxy), default=0.0)
                        if max_iou < self.neg_iou_thresh:
                            # store as xywh
                            self.samples.append((vis_path, ir_path, (x1, y1, bw, bh), 0.0))
                            if len(self.samples) >= max_samples:
                                return

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_and_resize(self, img: np.ndarray, box_xywh: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _clamp_box_xywh(box_xywh, w, h)
        patch = img[y1:y2, x1:x2]
        patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        return patch

    def __getitem__(self, idx: int):
        vis_path, ir_path, box_xywh, label = self.samples[idx]
        vis = cv2.imread(vis_path)
        ir = cv2.imread(ir_path)
        if vis is None or ir is None:
            # fallback to next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

        vis_patch = self._crop_and_resize(vis, box_xywh)
        ir_patch = self._crop_and_resize(ir, box_xywh)
        return (
            torch.from_numpy(vis_patch),
            torch.from_numpy(ir_patch),
            torch.tensor([label], dtype=torch.float32),
        )

