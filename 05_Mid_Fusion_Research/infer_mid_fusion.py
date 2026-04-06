import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch

from mid_fusion_refiner import MidFusionRefiner


# ===================== Config =====================
REFINER_CKPT = r"E:\kaist_yolo\05_Mid_Fusion_Research\weights\mid_fusion_refiner.pt"
VISIBLE_TEST_IMG = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val\set09_V000_I01259.jpg"
IR_TEST_IMG = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val\set09_V000_I01259.jpg"
ALPHA = 0.6  # final_score = alpha * late_fusion_score + (1-alpha) * refiner_score

# late_fusion import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "03_Modal_Fusion"))
from late_fusion import late_fusion_single_image  # noqa: E402
# ==================================================


def clamp_xyxy(box: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box[:4]
    x1 = int(max(0, min(round(float(x1)), w - 1)))
    y1 = int(max(0, min(round(float(y1)), h - 1)))
    x2 = int(max(x1 + 1, min(round(float(x2)), w)))
    y2 = int(max(y1 + 1, min(round(float(y2)), h)))
    return x1, y1, x2, y2


def crop_patch(img: np.ndarray, xyxy: Tuple[int, int, int, int], patch_size: int) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy
    patch = img[y1:y2, x1:x2]
    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0)


def load_refiner(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Refiner checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    base_ch = int(ckpt.get("base_ch", 32))
    patch_size = int(ckpt.get("patch_size", 96))
    model = MidFusionRefiner(base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, patch_size


def refine_late_fusion_scores(visible_img_path: str, ir_img_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_refiner(REFINER_CKPT, device)

    final_boxes, _, _, _ = late_fusion_single_image(visible_img_path, ir_img_path)
    final_boxes = np.asarray(final_boxes, dtype=np.float32)
    if final_boxes.size == 0:
        print("No detections from late fusion.")
        return final_boxes

    vis = cv2.imread(visible_img_path)
    ir = cv2.imread(ir_img_path)
    if vis is None or ir is None:
        raise RuntimeError("Failed to read visible/ir image.")
    h, w = vis.shape[:2]
    ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)

    refined = []
    with torch.no_grad():
        for det in final_boxes:
            x1, y1, x2, y2, score, cls_id = det
            xyxy = clamp_xyxy(det, w, h)
            vis_patch = crop_patch(vis, xyxy, patch_size).to(device)
            ir_patch = crop_patch(ir, xyxy, patch_size).to(device)
            logit = model(vis_patch, ir_patch)
            ref_score = torch.sigmoid(logit).item()
            new_score = ALPHA * float(score) + (1.0 - ALPHA) * ref_score
            refined.append([x1, y1, x2, y2, new_score, cls_id])

    refined = np.asarray(refined, dtype=np.float32)
    print(f"Input late-fusion detections: {len(final_boxes)}")
    print(f"Refined detections: {len(refined)}")
    print("Top-5 refined boxes (x1,y1,x2,y2,score,cls):")
    print(refined[:5])
    return refined


if __name__ == "__main__":
    refine_late_fusion_scores(VISIBLE_TEST_IMG, IR_TEST_IMG)

