import sys
import os

# ===================== 路径配置（仅需修改这里）=====================
# 【核心修改1】指向一个完整的连续序列文件夹（比如set09/V000）
# 注意：这里要用原始的连续帧，不要用你采样后的kaist_yolo/val
SEQ_VISIBLE_DIR = r"E:\KAIST_Dataset\images\set11\V000\visible"
SEQ_IR_DIR = r"E:\KAIST_Dataset\images\set11\V000\lwir"

# 结果保存路径
SAVE_VIDEO_PATH = "./kaist_continuous_sequence_tracking.mp4"
SAVE_IMAGE_DIR = "./continuous_seq_frames"

# 【新增】生成的MOT格式评估文件保存路径
# 这两个文件生成后，直接把路径填到track_eval.py里即可
OUT_GT_FILE = "./gt.txt"  # 生成的GT标注文件
OUT_TRACKER_FILE = "./tracker.txt"  # 生成的跟踪结果文件

# 【核心修改2】连续序列用10fps就是正常速度，不需要改慢
FPS = 10
# KAIST标注根目录（按帧txt）
ANNOTATION_ROOT = r"E:\KAIST_Dataset\annotations"
# 评估目标类别（建议与检测训练类别一致）
TARGET_CLASSES = {"person"}
# ====================================================================

# 导入late_fusion模块（保持不变）
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
fusion_module_dir = os.path.join(parent_dir, "03_Modal_Fusion")
sys.path.insert(0, fusion_module_dir)

# 预声明，避免静态分析器误报 try/except 导入告警
late_fusion_single_image = None
CONF_THRESH = 0.25
IMG_SIZE = 640

try:
    import late_fusion

    late_fusion_single_image = late_fusion.late_fusion_single_image
    CONF_THRESH = late_fusion.CONF_THRESH
    IMG_SIZE = late_fusion.IMG_SIZE
    print("✅ late_fusion模块导入成功！")
except ImportError as e:
    print(f"❌ late_fusion导入失败！错误信息：{e}")


    def late_fusion_single_image(*args, **kwargs):
        raise ImportError("late_fusion模块不可用，无法执行融合推理")


    sys.exit(1)

from sort_tracker import SORT
import cv2
import numpy as np
from tqdm import tqdm

# 可视化参数（保持不变）
BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
TITLE_BAR_HEIGHT = 60


def make_video_writer(path, fps, width, height):
    """兼容不同OpenCV版本创建视频写入器。"""
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_fn is None:
        raise RuntimeError("当前OpenCV缺少VideoWriter_fourcc，无法写入视频")
    fourcc = fourcc_fn(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def draw_tracking_result(frame, tracks):
    """在单张图像上绘制跟踪框+ID"""
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
        cv2.putText(
            canvas, text, (x1 + 2, text_y - 4),
            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS,
            lineType=cv2.LINE_AA
        )
    return canvas


def load_kaist_annotations(annot_dir, img_names, target_classes=None):
    """
    从KAIST按帧标注目录读取GT，返回:
      gt_data: {frame_id: [(gt_id, x1, y1, w, h), ...]}
      labeled_frames: 有GT标注文件且可解析的帧ID集合

    说明:
    - KAIST官方标注通常无跨帧ID，因此这里生成“全局唯一GT ID”避免伪ID切换。
    - 这能保证MOTA/FP/FN数值正确；ID相关指标在无真实ID场景下仅作参考。
    """
    if target_classes is None:
        target_classes = {"person"}

    gt_data = {}
    labeled_frames = set()
    next_gt_id = 1

    if not os.path.isdir(annot_dir):
        print(f"⚠️  警告：未找到标注目录: {annot_dir}")
        return gt_data, labeled_frames

    for frame_idx, img_name in enumerate(img_names):
        frame_id = frame_idx + 1
        stem, _ = os.path.splitext(img_name)
        ann_path = os.path.join(annot_dir, stem + ".txt")
        if not os.path.exists(ann_path):
            continue

        objs = []
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue

        for line in lines:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_name = parts[0].lower()
            if cls_name not in target_classes:
                continue

            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            if w <= 0 or h <= 0:
                continue

            objs.append((next_gt_id, x1, y1, w, h))
            next_gt_id += 1

        if objs:
            gt_data[frame_id] = objs
            labeled_frames.add(frame_id)

    print(f"✅ 成功加载GT标注帧: {len(labeled_frames)} / 总帧数: {len(img_names)}")
    return gt_data, labeled_frames


def track_continuous_sequence():
    """处理单个连续序列，生成流畅跟踪视频 + MOT评估文件"""
    # 1. 初始化SORT跟踪器
    tracker = SORT(max_age=5, min_hits=2, iou_threshold=0.3)

    # 2. 自动推导KAIST标注目录路径（按帧txt）
    seq_parent_dir = os.path.dirname(SEQ_VISIBLE_DIR)
    set_v_dir = os.path.basename(seq_parent_dir)
    set_dir = os.path.basename(os.path.dirname(seq_parent_dir))

    # 4. 读取并排序连续帧（按文件名数字排序，保证100%时序正确）
    img_names = sorted([
        f for f in os.listdir(SEQ_VISIBLE_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    if len(img_names) == 0:
        print(f"❌ 错误：在 {SEQ_VISIBLE_DIR} 未找到任何图片，请检查路径！")
        return
    print(f"✅ 找到连续序列，总帧数：{len(img_names)}")
    print(f"🎞️  视频播放帧率：{FPS}fps，预计播放时长：{len(img_names) / FPS:.1f}秒")

    # 4. 加载GT标注（按图像名对齐）
    gt_annot_dir = os.path.join(ANNOTATION_ROOT, set_dir, set_v_dir, "visible")
    gt_data, labeled_frames = load_kaist_annotations(
        gt_annot_dir,
        img_names,
        target_classes=TARGET_CLASSES
    )

    # 5. 初始化视频写入器
    first_vis_path = os.path.join(SEQ_VISIBLE_DIR, img_names[0])
    first_ir_path = os.path.join(SEQ_IR_DIR, img_names[0])
    first_vis_img = cv2.imread(first_vis_path)
    first_ir_img = cv2.imread(first_ir_path)
    if first_vis_img is None or first_ir_img is None:
        print(f"❌ 错误：第一帧图片读取失败，请检查路径")
        return

    h, w = first_vis_img.shape[:2]
    first_ir_img = cv2.resize(first_ir_img, (w, h), interpolation=cv2.INTER_LINEAR)
    total_width = w * 2
    total_height = h + TITLE_BAR_HEIGHT

    video_writer = make_video_writer(SAVE_VIDEO_PATH, FPS, total_width, total_height)
    os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

    # 【新增】打开MOT格式文件，准备写入
    gt_f = open(OUT_GT_FILE, 'w', encoding='utf-8')
    tracker_f = open(OUT_TRACKER_FILE, 'w', encoding='utf-8')

    # 6. 逐帧处理核心循环
    failed_frames = []
    for frame_idx, img_name in enumerate(tqdm(img_names, desc="连续序列跟踪处理中", total=len(img_names))):
        frame_id = frame_idx + 1  # MOT格式frame_id从1开始
        visible_img_path = os.path.join(SEQ_VISIBLE_DIR, img_name)
        ir_img_path = os.path.join(SEQ_IR_DIR, img_name)

        # -------------------------- 写入GT标注 --------------------------
        if frame_id in gt_data:
            for (gt_id, x1, y1, w_gt, h_gt) in gt_data[frame_id]:
                # MOT格式：frame, id, x1, y1, w, h, conf, -1, -1, -1
                line = f"{frame_id},{int(gt_id)},{x1:.1f},{y1:.1f},{w_gt:.1f},{h_gt:.1f},1.0,-1,-1,-1\n"
                gt_f.write(line)

        # -------------------------- 原有跟踪处理逻辑 --------------------------
        process_success = True
        try:
            fusion_boxes, _, _, visible_frame = late_fusion_single_image(visible_img_path, ir_img_path)
            ir_frame = cv2.imread(ir_img_path)
            if ir_frame is None:
                raise ValueError(f"红外帧读取失败: {ir_img_path}")
            if ir_frame.ndim == 2:
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            ir_frame = cv2.resize(ir_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            err_msg = f"帧{frame_idx}_{img_name} 处理失败：{str(e)}"
            failed_frames.append(err_msg)
            process_success = False
            continue

        dets = fusion_boxes[:, :5] if len(fusion_boxes) > 0 else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(dets)

        # -------------------------- 写入跟踪结果 --------------------------
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            w_track = x2 - x1
            h_track = y2 - y1
            # MOT格式：frame, id, x1, y1, w, h, conf, -1, -1, -1
            line = f"{frame_id},{int(track_id)},{x1:.1f},{y1:.1f},{w_track:.1f},{h_track:.1f},{conf:.1f},-1,-1,-1\n"
            tracker_f.write(line)

        # -------------------------- 原有可视化逻辑 --------------------------
        vis_with_track = draw_tracking_result(visible_frame, tracks)
        ir_with_track = draw_tracking_result(ir_frame, tracks)

        # 生成标题栏
        title_bar = np.zeros((TITLE_BAR_HEIGHT, total_width, 3), dtype=np.uint8)
        cv2.putText(
            title_bar, "Visible", (w // 2 - 60, TITLE_BAR_HEIGHT // 2 + 10),
            FONT, 0.9, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            title_bar, "Infrared", (w + w // 2 - 60, TITLE_BAR_HEIGHT // 2 + 10),
            FONT, 0.9, (255, 0, 0), 2, cv2.LINE_AA
        )

        dual_frame = cv2.hconcat([vis_with_track, ir_with_track])
        final_frame = cv2.vconcat([title_bar, dual_frame])

        video_writer.write(final_frame)
        cv2.imwrite(os.path.join(SAVE_IMAGE_DIR, img_name), final_frame)

    # 关闭文件句柄
    gt_f.close()
    tracker_f.close()
    video_writer.release()
    cv2.destroyAllWindows()

    from sort_tracker import KalmanBoxTracker
    KalmanBoxTracker.count = 0

    print("=" * 80)
    print(f"✅ 连续序列跟踪视频生成完成！")
    print(f"📹 视频保存路径：{os.path.abspath(SAVE_VIDEO_PATH)}")
    print(f"🎞️  视频帧率：{FPS}fps，总时长：{len(img_names) / FPS:.1f}秒")
    print(f"🖼️  单帧图片保存路径：{os.path.abspath(SAVE_IMAGE_DIR)}")
    print(f"✅ 成功处理：{len(img_names) - len(failed_frames)} / 总帧数：{len(img_names)}")
    if failed_frames:
        print(f"⚠️  失败帧数：{len(failed_frames)}，前5条失败详情：")
        for err in failed_frames[:5]:
            print(f"  - {err}")

    # 【新增】评估文件生成提示
    print("\n" + "=" * 50)
    print(f"✅ MOT评估文件已生成！")
    print(f"📄 GT标注文件：{os.path.abspath(OUT_GT_FILE)}")
    print(f"📄 跟踪结果文件：{os.path.abspath(OUT_TRACKER_FILE)}")
    print(f"📌 GT有标注帧数：{len(labeled_frames)} / {len(img_names)}")
    print(f"📌 GT类别过滤：{sorted(TARGET_CLASSES)}")
    print(f"💡 请将这两个路径填入track_eval.py的配置项：")
    print(f"   GT_FILE = r\"{os.path.abspath(OUT_GT_FILE)}\"")
    print(f"   TRACKER_FILE = r\"{os.path.abspath(OUT_TRACKER_FILE)}\"")
    print("=" * 50)


if __name__ == "__main__":
    track_continuous_sequence()