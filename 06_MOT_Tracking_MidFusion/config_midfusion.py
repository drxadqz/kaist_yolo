import os

# ===================== Sequence Inputs =====================
SEQ_VISIBLE_DIR = r"E:\KAIST_Dataset\images\set11\V000\visible"
SEQ_IR_DIR = r"E:\KAIST_Dataset\images\set11\V000\lwir"
ANNOTATION_ROOT = r"E:\KAIST_Dataset\annotations"
TARGET_CLASSES = {"person"}

# ===================== Fusion / Refiner =====================
REFINER_CKPT = r"E:\kaist_yolo\05_Mid_Fusion_Research\weights\mid_fusion_refiner.pt"
REFINE_ALPHA = 0.6  # final_score = alpha * late_score + (1 - alpha) * refiner_score
REFINE_SCORE_THRESH = 0.25

# ===================== Tracking =====================
TRACKER_TYPE = "sort"  # "sort" or "bytetrack"
TRACKER_KWARGS = {
    "max_age": 5,
    "min_hits": 2,
    "iou_threshold": 0.3,
}

# ===================== Output Isolation =====================
BASE_OUTPUT_DIR = r"E:\kaist_yolo\06_MOT_Tracking_MidFusion\outputs_midfusion"
RUN_NAME = "set11_v000_midfusion_sort"
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
SAVE_VIDEO_PATH = os.path.join(RUN_DIR, "kaist_midfusion_tracking.mp4")
SAVE_IMAGE_DIR = os.path.join(RUN_DIR, "continuous_seq_frames")
OUT_GT_FILE = os.path.join(RUN_DIR, "gt.txt")
OUT_TRACKER_FILE = os.path.join(RUN_DIR, "tracker.txt")

# ===================== Misc =====================
FPS = 10
IOU_THRESH_EVAL = 0.5

