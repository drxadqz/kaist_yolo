from ultralytics import YOLO
import cv2
import numpy as np

# ===================== 配置项 =====================
IR_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\ir_baseline\kaist_lwir_yolo11n\weights\best.pt"
VISIBLE_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\visible_baseline\kaist_visible_yolo11n\weights\best.pt"

IMG_SIZE = 640
IOU_THRESH = 0.5       # 两路检测框配对阈值
CONF_THRESH =0.25     # 单模态检测置信度阈值
NMS_THRESH = 0.45      # 最终NMS阈值
# ==================================================

# 加载模型
ir_model = YOLO(IR_MODEL_PATH)
visible_model = YOLO(VISIBLE_MODEL_PATH)


def iou(box1, box2):
    """计算两个检测框的 IoU，box 格式：(x1,y1,x2,y2)"""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    box2_area = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms_xyxy(boxes, iou_threshold=0.45):
    """
    对 xyxy 格式的 boxes 做 NMS
    boxes: [N, 6] -> x1,y1,x2,y2,conf,cls
    """
    if boxes is None:
        return np.empty((0, 6), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 仅按置信度做 NMS（当前任务只有 person 类，足够）
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[i] + areas[order[1:]] - inter_area
        ious = np.where(union_area > 0, inter_area / union_area, 0.0)

        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep]


def late_fusion_single_image(visible_img_path, ir_img_path):
    """
    对单张配对的可见光-红外图像执行晚期融合
    返回:
        final_boxes: [N,6]  x1,y1,x2,y2,conf,cls
        visible_boxes: [M,6]
        ir_boxes: [K,6]
        visible_img: 原始可见光图像
    """
    visible_img = cv2.imread(visible_img_path)
    ir_img = cv2.imread(ir_img_path)

    if visible_img is None or ir_img is None:
        print(f"[late_fusion] 图像读取失败: {visible_img_path} | {ir_img_path}")
        empty = np.empty((0, 6), dtype=np.float32)
        return empty, empty, empty, visible_img

    # 两路模型分别推理
    visible_results = visible_model(visible_img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
    ir_results = ir_model(ir_img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

    visible_boxes = visible_results.boxes.data.cpu().numpy().astype(np.float32) if len(visible_results.boxes) else np.empty((0, 6), dtype=np.float32)
    ir_boxes = ir_results.boxes.data.cpu().numpy().astype(np.float32) if len(ir_results.boxes) else np.empty((0, 6), dtype=np.float32)

    # 步骤1：两路框配对融合
    matched_boxes = []
    visible_used = np.zeros(len(visible_boxes), dtype=bool)
    ir_used = np.zeros(len(ir_boxes), dtype=bool)

    for i, vis_box in enumerate(visible_boxes):
        vis_xyxy = vis_box[:4]
        vis_conf = float(vis_box[4])

        for j, ir_box in enumerate(ir_boxes):
            if ir_used[j]:
                continue

            ir_xyxy = ir_box[:4]
            ir_conf = float(ir_box[4])

            current_iou = iou(vis_xyxy, ir_xyxy)
            if current_iou >= IOU_THRESH:
                # 置信度加权融合坐标
                weight_sum = vis_conf + ir_conf
                if weight_sum > 0:
                    fused_xyxy = (vis_xyxy * vis_conf + ir_xyxy * ir_conf) / weight_sum
                else:
                    fused_xyxy = vis_xyxy.copy()

                fused_conf = max(vis_conf, ir_conf)
                fused_cls = vis_box[5]  # 当前任务只做 person，直接沿用可见光类别即可

                fused_box = np.array(
                    [fused_xyxy[0], fused_xyxy[1], fused_xyxy[2], fused_xyxy[3], fused_conf, fused_cls],
                    dtype=np.float32
                )

                matched_boxes.append(fused_box)
                visible_used[i] = True
                ir_used[j] = True
                break

    # 步骤2：保留未配对框
    unpaired_boxes = []

    for i, vis_box in enumerate(visible_boxes):
        if not visible_used[i]:
            unpaired_boxes.append(vis_box.astype(np.float32))

    for j, ir_box in enumerate(ir_boxes):
        if not ir_used[j]:
            unpaired_boxes.append(ir_box.astype(np.float32))

    # 步骤3：合并并 NMS
    if len(matched_boxes) + len(unpaired_boxes) == 0:
        final_boxes = np.empty((0, 6), dtype=np.float32)
        return final_boxes, visible_boxes, ir_boxes, visible_img

    all_boxes = np.vstack(matched_boxes + unpaired_boxes).astype(np.float32)
    final_boxes = nms_xyxy(all_boxes, iou_threshold=NMS_THRESH)

    return final_boxes, visible_boxes, ir_boxes, visible_img


# 测试代码
if __name__ == "__main__":
    test_visible_img = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val\set09_V000_I01259.jpg"
    test_ir_img = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val\set09_V000_I01259.jpg"

    final_boxes, vis_boxes, ir_boxes, vis_img = late_fusion_single_image(test_visible_img, test_ir_img)

    print(f"可见光检测框数量：{len(vis_boxes)}")
    print(f"红外检测框数量：{len(ir_boxes)}")
    print(f"融合后检测框数量：{len(final_boxes)}")
    print("融合结果：", final_boxes)