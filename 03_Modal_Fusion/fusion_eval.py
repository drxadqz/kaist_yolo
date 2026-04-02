from ultralytics import YOLO
from late_fusion import late_fusion_single_image
import os
import numpy as np
from tqdm import tqdm
import cv2

# ===================== 配置项 =====================
VAL_VISIBLE_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val"
VAL_IR_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val"
VAL_LABEL_DIR = r"E:\KAIST_Dataset\kaist_yolo\labels\visible\val"
IR_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\ir_baseline\kaist_lwir_yolo11n\weights\best.pt"
VISIBLE_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\visible_baseline\kaist_visible_yolo11n\weights\best.pt"
IMG_SIZE = 640
IOU_THRESH = 0.5
CONF_THRESH = 0.25  # 固定正常推理阈值，不改动！
# ==================================================

# 加载模型
ir_model = YOLO(IR_MODEL_PATH)
visible_model = YOLO(VISIBLE_MODEL_PATH)


def load_ground_truth(label_path, img_width, img_height):
    gt_boxes = []
    if not os.path.exists(label_path):
        return np.array(gt_boxes)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = map(float, parts)
        # 统一类别为int（当前仅person，cls=0）
        cls = int(cls)
        x1 = (cx - w / 2) * img_width
        y1 = (cy - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cy + h / 2) * img_height
        gt_boxes.append([x1, y1, x2, y2, cls])
    return np.array(gt_boxes, dtype=np.float32)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(pred_boxes, gt_boxes, iou_thresh=0.5):
    # 过滤空值，统一类型
    pred_boxes = np.array(pred_boxes, dtype=np.float32) if len(pred_boxes) > 0 else np.empty((0, 6), dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.empty((0, 5), dtype=np.float32)

    if len(gt_boxes) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        precision = 1.0 if fp == 0 else 0.0
        recall = 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, tp, fp, fn

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)

    for i, pred in enumerate(pred_boxes):
        pred_xyxy = pred[:4]
        max_iou = 0
        max_idx = -1
        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            current_iou = iou(pred_xyxy, gt[:4])
            if current_iou > max_iou:
                max_iou = current_iou
                max_idx = j
        if max_iou >= iou_thresh and max_idx != -1:
            gt_matched[max_idx] = True
            pred_matched[i] = True

    tp = np.sum(pred_matched)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - np.sum(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn


def evaluate_all():
    img_names = [f for f in os.listdir(VAL_VISIBLE_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"共找到 {len(img_names)} 张测试图像")

    total_tp_vis, total_fp_vis, total_fn_vis = 0, 0, 0
    total_tp_ir, total_fp_ir, total_fn_ir = 0, 0, 0
    total_tp_fusion, total_fp_fusion, total_fn_fusion = 0, 0, 0
    failed_imgs = []  # 记录读取失败的图像

    for img_name in tqdm(img_names, desc="评估中"):
        try:
            visible_img_path = os.path.join(VAL_VISIBLE_DIR, img_name)
            ir_img_path = os.path.join(VAL_IR_DIR, img_name)
            label_path = os.path.join(VAL_LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')

            # 读取可见光图像（用于获取尺寸）
            img = cv2.imread(visible_img_path)
            if img is None:
                failed_imgs.append(f"可见光图像读取失败: {visible_img_path}")
                continue
            img_height, img_width = img.shape[:2]
            gt_boxes = load_ground_truth(label_path, img_width, img_height)

            # ===================== 单模态推理（固定conf=0.25）=====================
            # 可见光推理
            vis_results = visible_model(img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
            vis_boxes = vis_results.boxes.data.cpu().numpy() if vis_results.boxes is not None else np.array([])
            p_vis, r_vis, f1_vis, tp_vis, fp_vis, fn_vis = calculate_metrics(vis_boxes, gt_boxes)

            # 红外推理
            ir_img = cv2.imread(ir_img_path)
            if ir_img is None:
                failed_imgs.append(f"红外图像读取失败: {ir_img_path}")
                ir_boxes = np.array([])
            else:
                ir_results = ir_model(ir_img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
                ir_boxes = ir_results.boxes.data.cpu().numpy() if ir_results.boxes is not None else np.array([])
            p_ir, r_ir, f1_ir, tp_ir, fp_ir, fn_ir = calculate_metrics(ir_boxes, gt_boxes)

            # 融合推理
            fusion_boxes = late_fusion_single_image(visible_img_path, ir_img_path)
            # 兼容返回值
            if isinstance(fusion_boxes, (tuple, list)):
                fusion_boxes = fusion_boxes[0]
            p_fusion, r_fusion, f1_fusion, tp_fusion, fp_fusion, fn_fusion = calculate_metrics(fusion_boxes, gt_boxes)

            # 累加指标
            total_tp_vis += tp_vis
            total_fp_vis += fp_vis
            total_fn_vis += fn_vis
            total_tp_ir += tp_ir
            total_fp_ir += fp_ir
            total_fn_ir += fn_ir
            total_tp_fusion += tp_fusion
            total_fp_fusion += fp_fusion
            total_fn_fusion += fn_fusion

        except Exception as e:
            failed_imgs.append(f"处理失败 {img_name}: {str(e)}")
            continue

    # 打印失败日志
    if failed_imgs:
        print(f"\n⚠️ 共 {len(failed_imgs)} 张图像处理失败：")
        for err in failed_imgs[:5]:  # 仅打印前5条，避免刷屏
            print(err)

    # 计算全局指标
    def calc_global_metrics(total_tp, total_fp, total_fn):
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    p_vis, r_vis, f1_vis = calc_global_metrics(total_tp_vis, total_fp_vis, total_fn_vis)
    p_ir, r_ir, f1_ir = calc_global_metrics(total_tp_ir, total_fp_ir, total_fn_ir)
    p_fusion, r_fusion, f1_fusion = calc_global_metrics(total_tp_fusion, total_fp_fusion, total_fn_fusion)

    # 输出结果
    print("=" * 80)
    print("【统一评估标准（conf=0.25，IoU=0.5）：三者公平对比】")
    print("=" * 80)
    print(f"可见光 | P:{p_vis:.4f} R:{r_vis:.4f} F1:{f1_vis:.4f} | TP:{total_tp_vis} FP:{total_fp_vis} FN:{total_fn_vis}")
    print(f"红外   | P:{p_ir:.4f} R:{r_ir:.4f} F1:{f1_ir:.4f} | TP:{total_tp_ir} FP:{total_fp_ir} FN:{total_fn_ir}")
    print(f"融合   | P:{p_fusion:.4f} R:{r_fusion:.4f} F1:{f1_fusion:.4f} | TP:{total_tp_fusion} FP:{total_fp_fusion} FN:{total_fn_fusion}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_all()