import sys
import os
import time
import json
import csv
import random
import argparse
import platform
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from datetime import datetime

# ===================== 路径处理+导入逻辑 =====================
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
fusion_module_dir = os.path.join(parent_dir, "03_Modal_Fusion")
sys.path.insert(0, fusion_module_dir)

try:
    import late_fusion
    from late_fusion import (
        IR_MODEL_PATH,
        VISIBLE_MODEL_PATH,
        IMG_SIZE,
        CONF_THRESH,
        ir_model,
        visible_model,
        late_fusion_single_image,
    )

    print("✅ late_fusion模块导入成功！")
except ImportError as e:
    print(f"❌ late_fusion导入失败！错误信息：{e}")
    late_fusion = None
    IR_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\ir_baseline\kaist_lwir_yolo11n\weights\best.pt"
    VISIBLE_MODEL_PATH = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\visible_baseline\kaist_visible_yolo11n\weights\best.pt"
    IMG_SIZE = 640
    CONF_THRESH = 0.25
    ir_model = None
    visible_model = None


    def late_fusion_single_image(*args, **kwargs):
        raise ImportError("late_fusion模块不可用，无法执行融合推理")


    sys.exit(1)

# ========== 跟踪器一键切换：SORT/ByteTrack ==========
from sort_tracker import SORT

# 要对比ByteTrack时，注释上面一行，取消下面一行注释即可
# from bytetrack_tracker import BYTETracker as SORT
# ====================================================

# ===================== 100%适配你的路径+论文标准 =====================
# 图片根目录
IMAGES_ROOT = r"E:\KAIST_Dataset\images"
# 正确的标注根目录
ANNOTATION_ROOT = r"E:\KAIST 多光谱行人检测数据集"

# MOT评估核心参数
MATCH_IOU_THRESH = 0.5
# ===================== 【临时修改1】只跑 set06 验证 =====================
EVAL_SETS = ['set06']  # 验证完改回：['set06', 'set07', 'set08', 'set09', 'set10', 'set11']
# ========================================================================
# 是否启用跳帧采样（MOT评估必须关闭，使用连续完整帧）
USE_PAPER_STANDARD_SAMPLING = False

# 结果输出目录
VIS_RESULTS_DIR = os.path.join(parent_dir, "04_MOT_Tracking", "vis_results")
os.makedirs(VIS_RESULTS_DIR, exist_ok=True)

# 是否保存可视化视频（可选功能，验证时建议关闭，跑更快）
ENABLE_VIDEO_VIS = False

# ===================== 【临时修改2】开启调试日志 =====================
ENABLE_DEBUG_LOG = True  # 验证完可改为 False 减少输出


# ====================================================================

# ============================================================================

def iou(box1, box2):
    """标准IoU计算，全代码统一逻辑"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def ensure_dir(dir_path):
    """【修复】统一创建输出目录，兼容Windows路径。"""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def set_random_seed(seed=42):
    """设置随机种子，提升实验可复现性（论文复现实验推荐开启）。"""
    random.seed(seed)
    np.random.seed(seed)


def safe_cpu_numpy(data):
    """【修复】兼容 CUDA 张量/CPU 张量/NumPy 数组/列表，统一转为 NumPy。"""
    if data is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(data, np.ndarray):
        return data
    try:
        # 先转 CPU，再转 numpy，避免 cuda:0 直接 numpy 报错
        return data.cpu().numpy()
    except Exception:
        try:
            return np.asarray(data)
        except Exception:
            return np.empty((0,), dtype=np.float32)


def yolo_infer(model_obj, image_bgr):
    """【修复】统一封装 YOLO 推理，避免静态分析误判 None 调用。"""
    if model_obj is None:
        raise ImportError("YOLO 模型未正确加载，无法执行推理")
    return model_obj(image_bgr, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]


def draw_boxes(image, boxes, color, prefix, is_track=False):
    """【修复】统一绘制检测框/跟踪框，框上方标注模态前缀、置信度和ID。"""
    if image is None:
        return None

    canvas = image.copy()
    if boxes is None:
        return canvas

    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return canvas

    h, w = canvas.shape[:2]
    color = tuple(int(c) for c in color)

    for box in boxes:
        if len(box) < 5:
            continue

        x1, y1, x2, y2 = map(float, box[:4])
        if is_track:
            obj_id = int(box[4]) if len(box) > 4 else -1
            conf = float(box[5]) if len(box) > 5 else 0.0
            label = f"{prefix} ID:{obj_id} {conf:.2f}"
        else:
            conf = float(box[4]) if len(box) > 4 else 0.0
            label = f"{prefix} {conf:.2f}"

        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 文本背景框，避免在亮图上看不清
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1
        text_y = max(text_h + 4, y1 - 6)
        cv2.rectangle(canvas, (text_x, text_y - text_h - baseline - 4),
                      (text_x + text_w + 4, text_y + baseline), color, -1)
        cv2.putText(canvas, label, (text_x + 2, text_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas


def make_side_by_side(rgb_img, ir_img, fusion_img, titles=None):
    """【修复】将三幅图横向拼接，便于论文对比展示。"""
    if rgb_img is None or ir_img is None or fusion_img is None:
        return None

    if titles is None:
        titles = ["RGB", "IR", "Fusion"]

    imgs = [rgb_img, ir_img, fusion_img]
    # 统一高度，避免拼接时报错
    target_h = min(img.shape[0] for img in imgs)
    resized = []
    for img, title in zip(imgs, titles):
        scale = target_h / float(img.shape[0])
        new_w = max(1, int(round(img.shape[1] * scale)))
        r = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(r, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        resized.append(r)

    return cv2.hconcat(resized)


def load_ground_truth(label_path, img_width, img_height):
    """
    【修复】按YOLO标注格式读取GT：class_id cx cy w h（均为归一化坐标）
    返回格式固定为(N,5)：[x1, y1, x2, y2, class_id]
    """
    gt_boxes = []

    # 【修复】文件不存在时返回合规空数组
    if not os.path.exists(label_path):
        print(f"[DEBUG GT读取] 文件: {label_path} | 有效行人GT数量: 0")
        return np.empty((0, 5), dtype=np.float32)

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # 【修复】YOLO标签必须至少5列：class_id cx cy w h
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            # 【修复】只评估行人类别(class_id=0)
            if class_id != 0:
                continue

            # 【修复】归一化坐标必须位于[0,1]，否则按异常行过滤
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                continue

            # 【修复】按要求公式将YOLO归一化坐标转换为像素级xyxy
            x1 = (cx - w / 2.0) * float(img_width)
            y1 = (cy - h / 2.0) * float(img_height)
            x2 = (cx + w / 2.0) * float(img_width)
            y2 = (cy + h / 2.0) * float(img_height)

            # 【修复】边界裁剪，兼容越界标注
            x1 = max(0.0, min(float(img_width), x1))
            y1 = max(0.0, min(float(img_height), y1))
            x2 = max(0.0, min(float(img_width), x2))
            y2 = max(0.0, min(float(img_height), y2))

            # 【修复】过滤无效框和小目标
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue
            if box_w * box_h < 50:
                continue

            gt_boxes.append([x1, y1, x2, y2, float(class_id)])

    except Exception as e:
        print(f"⚠️  标注文件读取异常: {label_path}, 错误: {str(e)}")
        print(f"[DEBUG GT读取] 文件: {label_path} | 有效行人GT数量: 0")
        return np.empty((0, 5), dtype=np.float32)

    # 【修复】每个文件都打印读取结果，方便排查GT统计
    print(f"[DEBUG GT读取] 文件: {label_path} | 有效行人GT数量: {len(gt_boxes)}")
    if len(gt_boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    return np.array(gt_boxes, dtype=np.float32)


def get_label_path_from_visible_rel_path(img_rel_path):
    """【修复】从 visible 相对路径生成对应的 YOLO 标注路径。"""
    rel = img_rel_path.replace('\\', '/')
    rel = rel.replace('/visible/', '/')
    return os.path.splitext(rel)[0] + '.txt'


def get_image_pair_paths(img_rel_path):
    """【修复】由可见光相对路径生成 visible / lwir / label 三路路径。"""
    rel = img_rel_path.replace('\\', '/')
    visible_img_path = os.path.join(IMAGES_ROOT, rel)
    ir_img_path = os.path.join(IMAGES_ROOT, rel.replace('/visible/', '/lwir/'))
    label_rel_path = get_label_path_from_visible_rel_path(img_rel_path)
    label_path = os.path.join(ANNOTATION_ROOT, label_rel_path)
    return visible_img_path, ir_img_path, label_path


def assign_gt_ids(sequence_gt_list, iou_thresh=0.5):
    """为序列内的GT框分配全局唯一ID，用于跟踪评估"""
    sequence_gt_with_id = []
    next_gt_id = 0
    prev_gt_with_id = np.empty((0, 6), dtype=np.float32)  # 上一帧带ID的GT

    for current_gt in sequence_gt_list:
        # 空帧直接添加空数组，跳过处理
        if len(current_gt) == 0:
            sequence_gt_with_id.append(np.empty((0, 6), dtype=np.float32))
            continue

        current_gt_with_id = np.zeros((len(current_gt), 6), dtype=np.float32)
        current_gt_with_id[:, :5] = current_gt
        current_gt_with_id[:, 5] = -1  # 初始化为未分配ID

        if len(prev_gt_with_id) > 0 and len(current_gt) > 0:
            # 匈牙利算法匹配前后帧GT，分配相同ID
            iou_matrix = np.zeros((len(current_gt), len(prev_gt_with_id)), dtype=np.float32)
            for d, det in enumerate(current_gt):
                for t, trk in enumerate(prev_gt_with_id):
                    iou_matrix[d, t] = iou(det[:4], trk[:4])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            # 分配ID
            for d, t in zip(row_ind, col_ind):
                if iou_matrix[d, t] >= iou_thresh:
                    current_gt_with_id[d, 5] = prev_gt_with_id[t, 5]

        # 为未匹配的GT分配新ID
        for i in range(len(current_gt_with_id)):
            if current_gt_with_id[i, 5] == -1:
                current_gt_with_id[i, 5] = next_gt_id
                next_gt_id += 1

        sequence_gt_with_id.append(current_gt_with_id)
        prev_gt_with_id = current_gt_with_id.copy()

    return sequence_gt_with_id


def split_into_sequences():
    """自动遍历所有序列，兼容V000/V001，使用连续完整帧做MOT评估"""
    sequence_dict = {}
    total_sampled_frames = 0

    for set_name in EVAL_SETS:
        set_path = os.path.join(IMAGES_ROOT, set_name)
        if not os.path.exists(set_path):
            print(f"⚠️  未找到数据集文件夹: {set_path}")
            continue

        for seq_name in os.listdir(set_path):
            seq_path = os.path.join(set_path, seq_name)
            if not os.path.isdir(seq_path) or not seq_name.startswith('V'):
                continue

            visible_img_dir = os.path.join(seq_path, 'visible')
            if not os.path.exists(visible_img_dir):
                continue

            all_img_names = sorted([f for f in os.listdir(visible_img_dir) if f.lower().endswith(('.jpg', '.png'))])
            if len(all_img_names) == 0:
                continue

            # 【修复】MOT指标评估必须使用连续完整帧，不能跳帧
            sampled_img_names = all_img_names

            full_seq_key = f"{set_name}/{seq_name}"
            sequence_dict[full_seq_key] = [os.path.join(set_name, seq_name, 'visible', img) for img in
                                           sampled_img_names]
            total_sampled_frames += len(sampled_img_names)
            print(f"✅ 加载序列: {full_seq_key}, 有效帧数: {len(sampled_img_names)}/{len(all_img_names)}")

            # ===================== 【临时修改3】只保留第一个序列（set06/V000） =====================
            break  # 验证完删掉这行
            # ====================================================================================

    print(f"\n📊 【临时验证-仅set06/V000】共加载 {len(sequence_dict)} 个序列，总评估帧数: {total_sampled_frames}")
    return sequence_dict


def save_markdown_table(rows, save_path):
    """【修复】生成 Markdown 对比表格，便于直接粘贴论文。"""
    headers = ["检测方案", "MOTA(%)", "IDF1(%)", "IDSW", "TP", "FP", "FN", "FPS", "GT"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        idsw_val = row.get("idsw", row.get("ids", 0))
        fps_val = row.get("fps", row.get("FPS", 0.0))
        gt_val = row.get("gt", row.get("GT", 0))
        tp_val = row.get("tp", row.get("TP", 0))
        fp_val = row.get("fp", row.get("FP", 0))
        fn_val = row.get("fn", row.get("FN", 0))
        lines.append(
            f"| {row['mode']} | {float(row['mota']):.2f} | {float(row['idf1']):.2f} | {int(idsw_val)} | {int(tp_val)} | {int(fp_val)} | {int(fn_val)} | {float(fps_val):.2f} | {int(gt_val)} |"
        )

    md = "\n".join(lines) + "\n"
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(md)
    return md


def save_json(data, save_path):
    """保存 JSON 结果，便于论文归档和复现实验。"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows, save_path):
    """保存 CSV 结果，便于直接贴到论文表格/Excel。"""
    fieldnames = ["mode", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN", "FPS", "GT"]
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "mode": row["mode"],
                "MOTA": float(row.get("MOTA", row.get("mota", 0.0))),
                "IDF1": float(row.get("IDF1", row.get("idf1", 0.0))),
                "IDSW": int(row.get("IDSW", row.get("idsw", row.get("ids", 0)))),
                "TP": int(row.get("TP", row.get("tp", 0))),
                "FP": int(row.get("FP", row.get("fp", 0))),
                "FN": int(row.get("FN", row.get("fn", 0))),
                "FPS": float(row.get("FPS", row.get("fps", 0.0))),
                "GT": int(row.get("GT", row.get("gt", 0))),
            })


def build_parser():
    """命令行参数：支持论文复现实验的可控运行。"""
    parser = argparse.ArgumentParser(description="KAIST RGB-T MOT评估脚本（临时验证版）")
    # ===================== 【临时修改4】默认只跑 fusion 模式，验证更快 =====================
    parser.add_argument("--mode", type=str, default="fusion", choices=["all", "visible", "ir", "fusion"],
                        help="评估模式：all/visible/ir/fusion，默认 fusion（验证更快）")
    # ====================================================================================
    parser.add_argument("--save_dir", type=str, default=VIS_RESULTS_DIR,
                        help="结果保存根目录，默认 04_MOT_Tracking/vis_results")
    parser.add_argument("--run_name", type=str, default="debug_set06_v000",
                        help="实验名称（可选），用于区分不同论文实验")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认42")
    parser.add_argument("--enable_video_vis", action="store_true",
                        help="是否输出跟踪可视化视频（默认关闭）")
    return parser


def maybe_save_video(video_writer, frame_bgr):
    """【修复】可选保存视频帧，避免主逻辑被视频功能耦合。"""
    if video_writer is not None and frame_bgr is not None:
        video_writer.write(frame_bgr)


def make_video_writer(video_path, width, height, fps=20.0):
    """【修复】创建视频写入器，兼容不同 OpenCV 构建版本。"""
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_fn is None:
        return None
    fourcc = fourcc_fn(*"mp4v")
    return cv2.VideoWriter(video_path, fourcc, fps, (width, height))


def evaluate_tracking(detection_mode="fusion"):
    """评估指定检测模式的跟踪性能，符合MOTChallenge标准。"""
    if detection_mode == "ir":
        print(f"✅ 红外单模态模型加载完成: {IR_MODEL_PATH}")
    elif detection_mode == "visible":
        print(f"✅ 可见光单模态模型加载完成: {VISIBLE_MODEL_PATH}")
    else:
        print("✅ 多模态融合模式启用：late_fusion_single_image")

    sequence_dict = split_into_sequences()
    total_frames = sum(len(imgs) for imgs in sequence_dict.values())

    # 全局统计变量（MOT标准）
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_count = 0
    id_switch_total = 0

    # IDF1 相关统计
    idtp_total = 0
    idfp_total = 0
    idfn_total = 0

    # 耗时统计
    total_infer_time = 0.0

    for seq_name, seq_img_list in tqdm(sequence_dict.items(), desc=f"评估{detection_mode}跟踪性能"):
        # 可靠单序列跟踪逻辑：每个序列都重新初始化 SORT
        tracker = SORT(max_age=5, min_hits=2, iou_threshold=0.3)
        prev_track2gt = {}
        seq_gt_list = []

        seq_total_gt = 0
        seq_total_dets = 0
        seq_total_matches = 0
        seq_total_tp = 0
        seq_total_fp = 0
        seq_total_fn = 0

        video_writer = None
        video_path = ""
        if ENABLE_VIDEO_VIS:
            seq_vis_dir = os.path.join(VIS_RESULTS_DIR, detection_mode, seq_name.replace('/', '_'))
            ensure_dir(seq_vis_dir)
            video_path = os.path.join(seq_vis_dir, f"{seq_name.replace('/', '_')}_{detection_mode}.mp4")

        # -------------------- 第一步：预加载 GT --------------------
        for img_rel_path in seq_img_list:
            visible_img_path, _, label_path = get_image_pair_paths(img_rel_path)

            frame = cv2.imread(visible_img_path)
            if frame is None:
                seq_gt_list.append(np.empty((0, 5), dtype=np.float32))
                if ENABLE_DEBUG_LOG:
                    print(f"[DEBUG GT读取] 文件: {label_path} | 图片读取失败，按空GT处理")
                continue

            img_height, img_width = frame.shape[:2]
            gt_boxes = load_ground_truth(label_path, img_width, img_height)
            seq_gt_list.append(gt_boxes)

            total_gt_count += len(gt_boxes)
            seq_total_gt += len(gt_boxes)

        print(f"[DEBUG 序列完成] 序列名: {seq_name} | 该序列总GT数: {seq_total_gt}")

        # 分配序列内 GT ID，供 IDSW / IDF1 统计使用
        seq_gt_with_id = assign_gt_ids(seq_gt_list)

        # -------------------- 第二步：逐帧推理、跟踪、评估 --------------------
        for frame_idx, img_rel_path in enumerate(seq_img_list):
            visible_img_path, ir_img_path, _ = get_image_pair_paths(img_rel_path)

            frame = cv2.imread(visible_img_path)
            if frame is None:
                if ENABLE_DEBUG_LOG:
                    print(f"[DEBUG 帧] {seq_name} | 帧{frame_idx + 1}/{len(seq_img_list)} | 读取可见光失败，跳过")
                continue

            current_gt = seq_gt_with_id[frame_idx]

            # 融合检测结果先置空，便于可视化分支使用
            fusion_boxes = np.empty((0, 6), dtype=np.float32)

            start_time = time.time()

            # 1) 获取检测结果：复用 track_vis.py 可靠路径和 late_fusion 调用
            if detection_mode == "fusion":
                fusion_boxes, _, _, _ = late_fusion_single_image(visible_img_path, ir_img_path)
                dets = fusion_boxes[:, :5] if len(fusion_boxes) > 0 else np.empty((0, 5), dtype=np.float32)
            else:
                if detection_mode == "ir":
                    img = cv2.imread(ir_img_path)
                else:
                    img = cv2.imread(visible_img_path)

                if img is None:
                    dets = np.empty((0, 5), dtype=np.float32)
                    if ENABLE_DEBUG_LOG:
                        print(
                            f"[DEBUG 检测] {seq_name} | 帧{frame_idx + 1}/{len(seq_img_list)} | 检测图像读取失败，使用空检测")
                else:
                    if detection_mode == "ir":
                        results = yolo_infer(ir_model, img)
                    else:
                        results = yolo_infer(visible_model, img)

                    if len(results.boxes) > 0:
                        det_tensor = results.boxes.data
                        det_np = safe_cpu_numpy(det_tensor).astype(np.float32)
                        dets = det_np[:, :5]
                    else:
                        dets = np.empty((0, 5), dtype=np.float32)

            det_count = len(dets)
            seq_total_dets += det_count

            # 2) SORT 跟踪更新
            tracks = tracker.update(dets)

            infer_time = time.time() - start_time
            total_infer_time += infer_time

            # 3) 匈牙利匹配：tracks vs current_gt
            current_track2gt = {}
            gt_matched = np.zeros(len(current_gt), dtype=bool)
            track_matched = np.zeros(len(tracks), dtype=bool)

            if len(tracks) > 0 and len(current_gt) > 0:
                iou_matrix = np.zeros((len(tracks), len(current_gt)), dtype=np.float32)
                for t, track in enumerate(tracks):
                    for g, gt in enumerate(current_gt):
                        iou_matrix[t, g] = iou(track[:4], gt[:4])

                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                for t, g in zip(row_ind, col_ind):
                    if iou_matrix[t, g] >= MATCH_IOU_THRESH:
                        track_matched[t] = True
                        gt_matched[g] = True
                        current_track2gt[int(tracks[t, 4])] = int(current_gt[g, 5])

            match_count = int(np.sum(track_matched))
            seq_total_matches += match_count

            tp = int(np.sum(track_matched))
            fp = int(len(tracks) - tp)
            fn = int(len(current_gt) - np.sum(gt_matched))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            seq_total_tp += tp
            seq_total_fp += fp
            seq_total_fn += fn

            # 4) ID Switch：同一 track_id 连续帧匹配到不同 GT ID 才计数
            for track_id, gt_id in current_track2gt.items():
                if track_id in prev_track2gt and prev_track2gt[track_id] != gt_id:
                    id_switch_total += 1

            # 5) IDF1 简化统计
            idtp = len(current_track2gt)
            idfp = len(tracks) - idtp
            idfn = len(current_gt) - idtp
            idtp_total += idtp
            idfp_total += idfp
            idfn_total += idfn

            prev_track2gt = current_track2gt.copy()

            if ENABLE_DEBUG_LOG:
                print(
                    f"[DEBUG 帧] {seq_name} | 帧{frame_idx + 1}/{len(seq_img_list)} | "
                    f"GT={len(current_gt)} DET={det_count} MATCH={match_count} | TP={tp} FP={fp} FN={fn}"
                )

            # 可视化视频（可选）
            if ENABLE_VIDEO_VIS:
                if detection_mode == "fusion":
                    det_vis = draw_boxes(frame, dets, (0, 0, 255), "Fusion")
                elif detection_mode == "visible":
                    det_vis = draw_boxes(frame, dets, (0, 255, 0), "RGB")
                else:
                    det_vis = draw_boxes(frame, dets, (255, 0, 0), "IR")

                track_vis = draw_boxes(frame, tracks, (0, 255, 255), "Track", is_track=True)
                compare_vis = make_side_by_side(det_vis, track_vis, frame,
                                                [f"{detection_mode}-Det", f"{detection_mode}-Track", "Raw"])
                if compare_vis is not None:
                    if video_writer is None:
                        h, w = compare_vis.shape[:2]
                        video_writer = make_video_writer(video_path, w, h, fps=20.0)
                    maybe_save_video(video_writer, compare_vis)

        # 序列结束后输出序列统计，便于快速定位指标异常
        print(
            f"[DEBUG 序列统计] {seq_name} | 总帧数={len(seq_img_list)} | 总GT数={seq_total_gt} | "
            f"总检测框数={seq_total_dets} | 总匹配数={seq_total_matches} | TP={seq_total_tp} | FP={seq_total_fp} | FN={seq_total_fn}"
        )

        from sort_tracker import KalmanBoxTracker
        KalmanBoxTracker.count = 0

        if video_writer is not None:
            video_writer.release()

    # -------------------- 指标计算（标准MOT写法） --------------------
    denom_gt = max(total_gt_count, 1)
    MOTA = (1 - (total_fp + total_fn + id_switch_total) / denom_gt) * 100
    MOTA = max(min(MOTA, 100.0), -100.0)

    IDP = idtp_total / (idtp_total + idfp_total) if (idtp_total + idfp_total) > 0 else 0
    IDR = idtp_total / (idtp_total + idfn_total) if (idtp_total + idfn_total) > 0 else 0
    IDF1 = 2 * IDP * IDR / (IDP + IDR) * 100 if (IDP + IDR) > 0 else 0

    avg_fps = total_frames / total_infer_time if total_infer_time > 0 else 0

    print("=" * 80)
    print(f"【{detection_mode} 跟踪性能评估结果（临时验证版）】")
    print(f"MOTA: {MOTA:.2f}%")
    print(f"IDF1: {IDF1:.2f}%")
    print(f"ID Switch次数: {id_switch_total}")
    print(f"平均帧率FPS: {avg_fps:.2f}")
    print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn} | 总GT行人目标数: {total_gt_count}")
    print(
        f"[DEBUG 指标中间值] total_fp={total_fp}, total_fn={total_fn}, id_switch_total={id_switch_total}, total_gt_count={total_gt_count}")
    print("=" * 80)

    return {
        # 论文常用命名
        "MOTA": MOTA,
        "IDF1": IDF1,
        "IDSW": id_switch_total,
        "FPS": avg_fps,
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "GT": int(total_gt_count),
        # 兼容旧字段命名
        "mota": MOTA,
        "idf1": IDF1,
        "ids": id_switch_total,
        "idsw": id_switch_total,
        "fps": avg_fps,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "gt": int(total_gt_count),
    }


# 主程序入口
if __name__ == "__main__":
    args = build_parser().parse_args()

    # 同步全局可视化开关，保持 evaluate_tracking 内部逻辑不改
    ENABLE_VIDEO_VIS = bool(args.enable_video_vis)
    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.run_name}_{timestamp}" if args.run_name else f"mot_eval_{timestamp}"
    run_dir = os.path.join(args.save_dir, run_tag)
    ensure_dir(run_dir)

    modes = ["visible", "ir", "fusion"] if args.mode == "all" else [args.mode]
    mode_name_map = {
        "visible": "可见光单模态",
        "ir": "红外单模态",
        "fusion": "本文多模态融合方案",
    }

    print("=" * 120)
    print(f"📌 【临时验证】仅跑 set06/V000 | 运行模式: {modes} | 结果目录: {run_dir}")
    print(f"📌 随机种子: {args.seed} | 连续帧评估: 是")

    results = []
    for mode in modes:
        if mode == "visible":
            print("📌 开始评估可见光单模态跟踪性能...")
        elif mode == "ir":
            print("\n📌 开始评估红外单模态跟踪性能...")
        else:
            print("\n📌 开始评估多模态融合方案跟踪性能...")
        mode_res = evaluate_tracking(detection_mode=mode)
        results.append({"mode": mode_name_map[mode], "mode_key": mode, **mode_res})

    md_path = os.path.join(run_dir, "tracking_compare.md")
    csv_path = os.path.join(run_dir, "tracking_compare.csv")
    json_path = os.path.join(run_dir, "tracking_compare.json")
    meta_path = os.path.join(run_dir, "run_meta.json")

    save_markdown_table(results, md_path)
    save_csv(results, csv_path)
    save_json(results, json_path)

    meta_info = {
        "run_tag": run_tag,
        "timestamp": timestamp,
        "mode": args.mode,
        "evaluated_modes": modes,
        "seed": args.seed,
        "enable_video_vis": ENABLE_VIDEO_VIS,
        "paths": {
            "images_root": IMAGES_ROOT,
            "annotation_root": ANNOTATION_ROOT,
            "visible_model": VISIBLE_MODEL_PATH,
            "ir_model": IR_MODEL_PATH,
            "output_dir": run_dir,
        },
        "evaluation": {
            "eval_sets": EVAL_SETS,
            "match_iou_thresh": MATCH_IOU_THRESH,
            "continuous_frames": True,
            "img_size": IMG_SIZE,
            "conf_thresh": CONF_THRESH,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    save_json(meta_info, meta_path)

    print("\n" + "=" * 120)
    print("【临时验证结果（仅set06/V000）】")
    print(f"| 检测方案 | MOTA(%) | IDF1(%) | IDSW | 平均FPS |")
    print(f"|----------|---------|---------|------|---------|")
    for item in results:
        print(f"| {item['mode']} | {item['mota']:.2f} | {item['idf1']:.2f} | {item['idsw']} | {item['fps']:.2f} |")
    print("=" * 120)
    print(f"[INFO] Markdown结果已保存: {md_path}")
    print(f"[INFO] CSV结果已保存: {csv_path}")
    print(f"[INFO] JSON结果已保存: {json_path}")
    print(f"[INFO] 运行元数据已保存: {meta_path}")