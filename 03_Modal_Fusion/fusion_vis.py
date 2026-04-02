# -*- coding: utf-8 -*-
"""
KAIST RGB-T 三栏可视化脚本

功能说明：
1) 左栏：RGB 单模态检测结果（绿色）
2) 中栏：IR/LWIR 单模态检测结果（蓝色）
3) 右栏：Late Fusion 融合结果（红色）

说明：
- 本脚本复用 late_fusion.py 中的模型路径、阈值配置和模型实例，避免参数漂移。
- 本脚本兼容 CUDA Tensor：所有检测框在转 numpy 前会先 .cpu()。
"""

import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

# 复用 late_fusion 中已经定义好的配置与模型实例
from late_fusion import (
    CONF_THRESH,
    IMG_SIZE,
    IR_MODEL_PATH,
    VISIBLE_MODEL_PATH,
    ir_model,
    late_fusion_single_image,
    visible_model,
)

# ===================== 可修改配置区 =====================
TEST_VISIBLE_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val"
TEST_LWIR_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val"

# 按需求统一保存到 ./vis_results
SAVE_DIR = "./vis_results"

# 绘图参数
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
BOX_THICKNESS = 2
TITLE_BAR_H = 56

# 颜色(BGR)
COLOR_RGB = (0, 255, 0)     # 绿色
COLOR_IR = (255, 0, 0)      # 蓝色
COLOR_FUSION = (0, 0, 255)  # 红色


# ===================== 工具函数 =====================
def _boxes_to_numpy(boxes_like) -> np.ndarray:
    """
    将任意检测框输入统一转成 [N, 6] 的 numpy.float32。

    支持输入：
    - ultralytics 的 Boxes.data (Tensor)
    - numpy.ndarray
    - list/tuple
    - None

    关键点：
    - 若输入是 CUDA Tensor，必须先 .cpu() 再 .numpy()，否则会报错。
    """
    if boxes_like is None:
        return np.empty((0, 6), dtype=np.float32)

    # PyTorch Tensor 或类似对象：优先走 .detach().cpu().numpy()
    if hasattr(boxes_like, "detach"):
        boxes_like = boxes_like.detach()
    if hasattr(boxes_like, "cpu"):
        boxes_like = boxes_like.cpu()
    if hasattr(boxes_like, "numpy"):
        arr = boxes_like.numpy()
    else:
        arr = np.asarray(boxes_like)

    arr = np.asarray(arr, dtype=np.float32)

    # 兼容空数组
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 兼容单框 [6] => [1, 6]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # 至少保证有前 6 列（x1,y1,x2,y2,conf,cls）
    if arr.shape[1] < 6:
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > 6:
        arr = arr[:, :6]

    return arr.astype(np.float32)


def _predict_single_modal(model, image: np.ndarray) -> np.ndarray:
    """
    使用单模态 YOLO 模型推理并返回 [N,6] 检测框。

    这里显式复用 late_fusion.py 中的 IMG_SIZE / CONF_THRESH 配置。
    """
    results = model(image, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
    if len(results.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return _boxes_to_numpy(results.boxes.data)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    color: tuple,
    prefix: str,
) -> np.ndarray:
    """
    在图像上绘制检测框与标签。

    标签格式："{prefix} {conf:.2f}"，例如：RGB 0.91
    对空框输入做鲁棒处理，直接返回原图副本。
    """
    canvas = image.copy()
    boxes_np = _boxes_to_numpy(boxes)

    if boxes_np.shape[0] == 0:
        return canvas

    h, w = canvas.shape[:2]
    for box in boxes_np:
        x1, y1, x2, y2, conf = box[:5]

        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # 画框
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # 标签文本：前缀 + 置信度
        label = f"{prefix} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # 避免标签画出图像边界
        text_top = max(0, y1 - th - 6)
        text_bottom = min(h - 1, y1)
        text_right = min(w - 1, x1 + tw + 4)

        cv2.rectangle(canvas, (x1, text_top), (text_right, text_bottom), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(th + 1, y1 - 4)),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return canvas


def _build_title_bar(single_w: int, total_w: int) -> np.ndarray:
    """生成三栏标题条，便于论文展示。"""
    bar = np.zeros((TITLE_BAR_H, total_w, 3), dtype=np.uint8)
    cv2.putText(bar, "RGB", (single_w // 2 - 30, 36), FONT, 0.85, COLOR_RGB, 2, cv2.LINE_AA)
    cv2.putText(bar, "IR", (single_w + single_w // 2 - 20, 36), FONT, 0.85, COLOR_IR, 2, cv2.LINE_AA)
    cv2.putText(bar, "Fusion", (single_w * 2 + single_w // 2 - 45, 36), FONT, 0.85, COLOR_FUSION, 2, cv2.LINE_AA)
    return bar


def _iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 xyxy 检测框的 IoU。"""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _unmatched_count(src_boxes: np.ndarray, ref_boxes: np.ndarray, iou_thresh: float) -> int:
    """
    统计 src 中无法与 ref 成功匹配的框数量。

    采用贪心匹配：每个 src 框最多匹配一个 ref 框，匹配条件 IoU >= iou_thresh。
    """
    src = _boxes_to_numpy(src_boxes)
    ref = _boxes_to_numpy(ref_boxes)

    if len(src) == 0:
        return 0
    if len(ref) == 0:
        return int(len(src))

    used = np.zeros(len(ref), dtype=bool)
    unmatched = 0

    for s in src:
        best_iou = 0.0
        best_j = -1
        for j, r in enumerate(ref):
            if used[j]:
                continue
            iou_val = _iou_xyxy(s[:4], r[:4])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_j >= 0 and best_iou >= iou_thresh:
            used[best_j] = True
        else:
            unmatched += 1

    return int(unmatched)


def has_three_way_difference(
    rgb_boxes: np.ndarray,
    ir_boxes: np.ndarray,
    fusion_boxes: np.ndarray,
    iou_thresh: float = 0.55,
    min_diff_boxes: int = 1,
) -> Tuple[bool, str]:
    """
    判断三路结果是否存在“可视化意义上的差异”。

    判定规则：
    1) 三路框数量不一致 => 有差异
    2) 数量一致时，再比较三组两两匹配的未匹配框数；
       若最大未匹配数 >= min_diff_boxes => 有差异
    """
    rgb = _boxes_to_numpy(rgb_boxes)
    ir = _boxes_to_numpy(ir_boxes)
    fusion = _boxes_to_numpy(fusion_boxes)

    if not (len(rgb) == len(ir) == len(fusion)):
        return True, f"count_diff(rgb={len(rgb)}, ir={len(ir)}, fusion={len(fusion)})"

    u_fr = _unmatched_count(fusion, rgb, iou_thresh)
    u_fi = _unmatched_count(fusion, ir, iou_thresh)
    u_ri = _unmatched_count(rgb, ir, iou_thresh)
    max_unmatched = max(u_fr, u_fi, u_ri)

    if max_unmatched >= int(max(1, min_diff_boxes)):
        return True, f"iou_diff(unmatched_max={max_unmatched}, iou_thresh={iou_thresh:.2f})"

    return False, "no_meaningful_diff"


# ===================== 核心函数（按你的命名要求） =====================
def vis_single_image_pair(
    visible_img_path: str,
    lwir_img_path: str,
    save_name: Optional[str] = None,
    save_dir: str = SAVE_DIR,
    only_save_diff: bool = False,
    diff_iou_thresh: float = 0.55,
    min_diff_boxes: int = 1,
    return_meta: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], bool, str]]:
    """
    生成单对 RGB/LWIR 图像的三栏对比图并保存。

    参数：
    - visible_img_path: 可见光图像路径
    - lwir_img_path: 红外图像路径
    - save_name: 输出文件名；为 None 时默认取可见光文件名

    返回：
    - 默认返回可视化图像(np.ndarray)；失败时返回 None
    - 当 return_meta=True 时，返回 (image_or_none, saved_flag, reason)
    """
    visible_img = cv2.imread(visible_img_path)
    lwir_img = cv2.imread(lwir_img_path)

    if visible_img is None:
        print(f"[vis] 可见光图像读取失败: {visible_img_path}")
        return None
    if lwir_img is None:
        print(f"[vis] 红外图像读取失败: {lwir_img_path}")
        return None

    # 若红外图像为灰度图，转换为3通道，确保可视化拼接尺寸一致
    if lwir_img.ndim == 2:
        lwir_img = cv2.cvtColor(lwir_img, cv2.COLOR_GRAY2BGR)

    # 1) 单模态推理（满足“调用单模态模型进行推理”要求）
    rgb_boxes = _predict_single_modal(visible_model, visible_img)
    ir_boxes = _predict_single_modal(ir_model, lwir_img)

    # 2) 融合推理（调用你的 late_fusion 逻辑）
    fusion_boxes, _, _, _ = late_fusion_single_image(visible_img_path, lwir_img_path)
    fusion_boxes = _boxes_to_numpy(fusion_boxes)

    # 3) 绘制三路结果
    rgb_panel = draw_boxes(visible_img, rgb_boxes, COLOR_RGB, "RGB")
    ir_panel = draw_boxes(lwir_img, ir_boxes, COLOR_IR, "IR")
    fusion_panel = draw_boxes(visible_img, fusion_boxes, COLOR_FUSION, "Fusion")

    # 4) 尺寸对齐后横向拼接
    h, w = rgb_panel.shape[:2]
    if ir_panel.shape[:2] != (h, w):
        ir_panel = cv2.resize(ir_panel, (w, h), interpolation=cv2.INTER_LINEAR)
    if fusion_panel.shape[:2] != (h, w):
        fusion_panel = cv2.resize(fusion_panel, (w, h), interpolation=cv2.INTER_LINEAR)

    merged = cv2.hconcat([rgb_panel, ir_panel, fusion_panel])
    title_bar = _build_title_bar(single_w=w, total_w=merged.shape[1])
    final_img = cv2.vconcat([title_bar, merged])

    # 5) 差异判定（仅在需要筛图时启用保存过滤）
    has_diff, diff_reason = has_three_way_difference(
        rgb_boxes,
        ir_boxes,
        fusion_boxes,
        iou_thresh=diff_iou_thresh,
        min_diff_boxes=min_diff_boxes,
    )

    # 6) 保存结果
    if save_name is None:
        save_name = os.path.basename(visible_img_path)

    if only_save_diff and not has_diff:
        print(f"[vis] 跳过(三路差异不明显): {save_name} | reason={diff_reason}")
        if return_meta:
            return final_img, False, diff_reason
        return final_img

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, final_img)
    print(f"[vis] 已保存: {save_path} | reason={diff_reason}")

    if return_meta:
        return final_img, True, diff_reason
    return final_img


def batch_vis_from_dirs(
    visible_dir: str = TEST_VISIBLE_DIR,
    lwir_dir: str = TEST_LWIR_DIR,
    save_dir: str = SAVE_DIR,
    max_images: Optional[int] = None,
    only_save_diff: bool = False,
    diff_iou_thresh: float = 0.55,
    min_diff_boxes: int = 1,
) -> None:
    """
    批量生成测试集三栏图，便于论文挑图。

    参数：
    - visible_dir/lwir_dir: 测试集路径
    - save_dir: 保存路径（默认 ./vis_results）
    - max_images: 仅处理前 N 张（None 表示全部）
    - only_save_diff: True 时仅保存三路结果有差异的样本
    - diff_iou_thresh: 差异判定时的 IoU 匹配阈值
    - min_diff_boxes: 至少多少个未匹配框才认为差异明显
    """
    if not os.path.isdir(visible_dir):
        print(f"[vis] 可见光目录不存在: {visible_dir}")
        return
    if not os.path.isdir(lwir_dir):
        print(f"[vis] 红外目录不存在: {lwir_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    names = sorted([n for n in os.listdir(visible_dir) if n.lower().endswith(valid_ext)])
    if max_images is not None:
        names = names[: max(0, int(max_images))]

    miss_count = 0
    fail_count = 0
    saved_count = 0
    no_diff_skip_count = 0

    for name in tqdm(names, desc="可视化生成中"):
        vis_path = os.path.join(visible_dir, name)
        ir_path = os.path.join(lwir_dir, name)

        if not os.path.isfile(ir_path):
            miss_count += 1
            print(f"[vis] 跳过(缺少红外配对): {name}")
            continue

        img, saved_flag, reason = vis_single_image_pair(
            vis_path,
            ir_path,
            save_name=name,
            save_dir=save_dir,
            only_save_diff=only_save_diff,
            diff_iou_thresh=diff_iou_thresh,
            min_diff_boxes=min_diff_boxes,
            return_meta=True,
        )
        if img is None:
            fail_count += 1
            continue
        if saved_flag:
            saved_count += 1
        elif reason == "no_meaningful_diff":
            no_diff_skip_count += 1

    ok_count = len(names) - miss_count - fail_count
    print(
        f"[vis] 批量完成: 可处理 {ok_count} / 总数 {len(names)} / 缺配对 {miss_count} / 失败 {fail_count}"
    )
    print(
        f"[vis] 保存统计: 已保存 {saved_count} / 无差异跳过 {no_diff_skip_count} "
        f"/ only_save_diff={only_save_diff}"
    )
    print(f"[vis] 结果目录: {save_dir}")


if __name__ == "__main__":
    # 打印当前复用的模型配置，便于确认与 late_fusion 完全一致
    print(f"[vis] VISIBLE_MODEL_PATH: {VISIBLE_MODEL_PATH}")
    print(f"[vis] IR_MODEL_PATH: {IR_MODEL_PATH}")

    # 单张示例：先用单张快速验证流程
    sample_visible = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val\set09_V000_I01259.jpg"
    sample_lwir = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val\set09_V000_I01259.jpg"
    vis_single_image_pair(sample_visible, sample_lwir, save_name="paper_compare_sample.png")

    # 批量示例1：保存全部可视化结果
    batch_vis_from_dirs()

    # 批量示例2：仅保存三路结果有明显差异的样本（推荐论文挑图）
    # batch_vis_from_dirs(max_images=300, only_save_diff=True, diff_iou_thresh=0.55, min_diff_boxes=1)


"""
===================== 运行说明（详细） =====================
1) 修改路径
   - 测试集路径在文件顶部可修改：
     TEST_VISIBLE_DIR = "..."
     TEST_LWIR_DIR = "..."
   - 结果默认保存到：SAVE_DIR = "./vis_results"

2) 单张测试（推荐先执行）
   - 修改 __main__ 中的 sample_visible 与 sample_lwir 为一对同名图像路径
   - 直接运行：python fusion_vis.py
   - 输出示例：./vis_results/paper_compare_sample.png

3) 批量测试集挑图
   - 先确认单张可运行
   - 打开 __main__，取消注释批量调用
   - 如需只跑前 N 张，可改为：batch_vis_from_dirs(max_images=100)
   - 如需仅保存三路有差异样本，可改为：
     batch_vis_from_dirs(max_images=300, only_save_diff=True, diff_iou_thresh=0.55, min_diff_boxes=1)
   - 输出全部保存到 ./vis_results，便于后续人工筛选论文图

4) 自动筛图开关说明（only_save_diff）
   - only_save_diff=False：保存所有样本（默认）
   - only_save_diff=True：仅保存三路结果有明显差异的样本
   - diff_iou_thresh：两框匹配阈值，越大越严格（常用 0.5~0.6）
   - min_diff_boxes：最少未匹配框数量，达到后判为“有差异”

5) 关于 CUDA 兼容
   - 本脚本对检测框统一走 _boxes_to_numpy：
     若检测框在 cuda:0，会先 .cpu() 再 .numpy()
   - 可避免常见报错：
     "TypeError: can't convert cuda:0 device type tensor to numpy"

6) 复用 late_fusion 配置
   - 本脚本导入并复用以下对象：
     IR_MODEL_PATH, VISIBLE_MODEL_PATH, IMG_SIZE, CONF_THRESH, ir_model, visible_model
   - 这样可确保单模态推理与融合推理使用一致的模型和阈值配置。
==========================================================
"""
