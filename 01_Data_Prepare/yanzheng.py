import cv2
import os
import yaml
import numpy as np
from tqdm import tqdm

# ====================== 【仅需核对修改这里的配置参数】 ======================
# 数据集根目录（images和labels文件夹所在的目录）
DATASET_ROOT = r"E:\KAIST_Dataset\kaist_yolo"
# 数据集类别yaml文件路径
YAML_PATH = r"../02_YOLO_Train/kaist_lwir.yaml"
# 可视化结果保存根目录
SAVE_ROOT = "kaist_full_label_visual_check"
# 是否全量验证所有样本（True=遍历全部图片，False=仅抽查，建议保持True确保全覆盖）
FULL_CHECK = True
# 抽查模式下，每个子集抽查的样本数量（仅FULL_CHECK=False时生效）
SAMPLE_NUM_PER_SUBSET = 500
# 是否仅保存有错误的样本可视化图（True=仅存错误样本，False=保存所有样本，节省空间建议开True）
SAVE_ONLY_ERROR = False
# 图片支持的后缀格式
IMG_SUFFIX = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
# YOLO标签标准列数：类别ID + cx cy w h 共5列
LABEL_COLUMN_NUM = 5


# =========================================================================

# 加载数据集类别名称
def load_classes(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"❌ 类别yaml文件不存在：{yaml_path}，默认使用['person']类别")
        return ["person"]
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']


# 兼容红外/可见光图片加载，支持16位红外图归一化
def load_image(img_path):
    # 无损读取所有格式图片
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # 处理16位红外图像，归一化到8位便于可视化
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 单通道灰度图转BGR，用于绘制彩色标注框
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# YOLO归一化坐标转像素坐标，同时校验坐标合法性
def yolo2pix_and_check(box, img_w, img_h):
    cx, cy, w, h = box
    # 校验归一化坐标是否在0-1合法区间
    is_valid = True
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        is_valid = False
    # 转换为像素坐标
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # 限制坐标不超出图片边界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    # 校验标注框是否有效（宽高必须大于0）
    if x2 <= x1 or y2 <= y1:
        is_valid = False
    return (x1, y1, x2, y2), is_valid


# 标签文件格式与内容校验
def check_label_file(txt_path, class_num):
    """
    校验标签文件
    返回：(是否合法, 标签行列表, 错误类型)
    错误类型：None=无错误, file_not_exist=文件缺失, empty=空标签, format_error=格式错误, class_error=类别ID非法
    """
    if not os.path.exists(txt_path):
        return False, [], "file_not_exist"

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 空标签文件
    if len(lines) == 0:
        return False, [], "empty"

    valid_lines = []
    for line in lines:
        data = line.split()
        # 列数不符合YOLO标准
        if len(data) != LABEL_COLUMN_NUM:
            return False, [], "format_error"
        # 转换数值类型
        try:
            data = list(map(float, data))
            cls_id = int(data[0])
            box = data[1:]
        except:
            return False, [], "format_error"
        # 类别ID超出范围
        if cls_id < 0 or cls_id >= class_num:
            return False, [], "class_error"
        valid_lines.append((cls_id, box))

    return True, valid_lines, None


# 单子集处理函数（子集：train/val，模态：lwir/visible）
def process_single_subset(subset_name, modal_name, class_names, save_dir):
    """
    处理单个数据集子集，完成标签校验、可视化绘制、结果保存与统计
    """
    # 中文名称映射，用于文件名标注
    subset_cn = "训练集" if subset_name == "train" else "测试集"
    modal_cn = "红外" if modal_name == "lwir" else "可见光"
    print(f"\n{'=' * 80}")
    print(f"🚀 开始校验【{subset_cn} - {modal_cn}】")
    print(f"{'=' * 80}")

    # 路径拼接
    img_dir = os.path.join(DATASET_ROOT, "images", modal_name, subset_name)
    label_dir = os.path.join(DATASET_ROOT, "labels", modal_name, subset_name)

    # 路径合法性校验
    if not os.path.exists(img_dir):
        print(f"❌ 图片目录不存在：{img_dir}，跳过该子集")
        return None
    if not os.path.exists(label_dir):
        print(f"❌ 标签目录不存在：{label_dir}，跳过该子集")
        return None

    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_SUFFIX)]
    total_img_num = len(img_files)
    if total_img_num == 0:
        print(f"❌ 图片目录未找到任何图片，跳过该子集")
        return None

    # 非全量模式下，随机抽取指定数量样本
    if not FULL_CHECK:
        import random
        random.shuffle(img_files)
        img_files = img_files[:SAMPLE_NUM_PER_SUBSET]
        print(f"📌 抽查模式：抽取 {len(img_files)} 张图片进行校验")
    else:
        print(f"📌 全量模式：共 {total_img_num} 张图片，将全部校验")

    # 统计变量初始化
    stat = {
        "total_img": len(img_files),
        "label_match": 0,
        "label_missing": 0,
        "label_empty": 0,
        "label_format_error": 0,
        "label_class_error": 0,
        "box_coordinate_error": 0,
        "img_read_fail": 0,
        "full_valid": 0
    }

    # 错误日志列表
    error_logs = {
        "label_missing": [],
        "label_empty": [],
        "label_format_error": [],
        "label_class_error": [],
        "box_coordinate_error": [],
        "img_read_fail": []
    }

    class_num = len(class_names)
    # 遍历所有图片处理
    for img_name in tqdm(img_files, desc=f"校验{subset_cn}-{modal_cn}"):
        # 初始化样本状态
        is_sample_valid = True
        sample_error_type = None
        # 匹配标签文件
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(label_dir, txt_name)

        # 1. 标签文件校验
        is_label_valid, label_lines, error_type = check_label_file(txt_path, class_num)
        if not is_label_valid:
            is_sample_valid = False
            sample_error_type = error_type
            if error_type == "file_not_exist":
                stat["label_missing"] += 1
                error_logs["label_missing"].append(img_name)
            elif error_type == "empty":
                stat["label_empty"] += 1
                error_logs["label_empty"].append(img_name)
            elif error_type == "format_error":
                stat["label_format_error"] += 1
                error_logs["label_format_error"].append(img_name)
            elif error_type == "class_error":
                stat["label_class_error"] += 1
                error_logs["label_class_error"].append(img_name)
        else:
            stat["label_match"] += 1

        # 2. 图片读取校验
        img_path = os.path.join(img_dir, img_name)
        img = load_image(img_path)
        if img is None:
            is_sample_valid = False
            sample_error_type = "img_read_fail"
            stat["img_read_fail"] += 1
            error_logs["img_read_fail"].append(img_name)
        else:
            img_h, img_w = img.shape[:2]

        # 3. 标注框坐标校验与绘制
        has_box_error = False
        if is_label_valid and img is not None:
            for (cls_id, box) in label_lines:
                (x1, y1, x2, y2), is_box_valid = yolo2pix_and_check(box, img_w, img_h)
                if not is_box_valid:
                    has_box_error = True
                    break
                # 绘制红色标注框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 绘制类别名称，避免文字超出图片边界
                text = class_names[cls_id]
                if y1 > 15:
                    text_y = y1 - 8
                else:
                    text_y = y2 + 18
                cv2.putText(img, text, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if has_box_error:
                is_sample_valid = False
                sample_error_type = "box_coordinate_error"
                stat["box_coordinate_error"] += 1
                error_logs["box_coordinate_error"].append(img_name)

        # 4. 完全合法样本统计
        if is_sample_valid:
            stat["full_valid"] += 1

        # 5. 可视化图片保存
        # 仅保存错误样本模式：有错误才保存
        if SAVE_ONLY_ERROR and not is_sample_valid:
            # 保存文件名格式：子集_中文名称_模态_中文名称_原文件名
            save_filename = f"{subset_name}_{subset_cn}_{modal_name}_{modal_cn}_{img_name}"
            save_path = os.path.join(save_dir, save_filename)
            # 错误样本左上角标注错误类型
            if sample_error_type:
                error_text = f"ERROR: {sample_error_type}"
                cv2.putText(img, error_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imwrite(save_path, img)
        # 全量保存模式：所有样本都保存
        elif not SAVE_ONLY_ERROR:
            save_filename = f"{subset_name}_{subset_cn}_{modal_name}_{modal_cn}_{img_name}"
            save_path = os.path.join(save_dir, save_filename)
            # 错误样本标注错误类型
            if not is_sample_valid and sample_error_type:
                error_text = f"ERROR: {sample_error_type}"
                cv2.putText(img, error_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imwrite(save_path, img)

    # 6. 子集校验结果输出
    print(f"\n✅ {subset_cn}-{modal_cn} 校验完成！统计结果：")
    print(f"  总校验图片数：{stat['total_img']}")
    print(f"  标签文件匹配成功：{stat['label_match']}")
    print(f"  标签文件缺失：{stat['label_missing']}")
    print(f"  空标签文件：{stat['label_empty']}")
    print(f"  标签格式错误：{stat['label_format_error']}")
    print(f"  类别ID非法：{stat['label_class_error']}")
    print(f"  标注框坐标非法：{stat['box_coordinate_error']}")
    print(f"  图片读取失败：{stat['img_read_fail']}")
    print(f"  ✅ 完全合法无错误样本数：{stat['full_valid']}")
    print(f"  📊 样本合法率：{stat['full_valid'] / stat['total_img'] * 100:.2f}%")

    # 7. 保存子集错误日志
    log_file = os.path.join(save_dir, f"{subset_name}_{modal_name}_error_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"===== {subset_cn} - {modal_cn} 标签校验错误日志 =====\n")
        f.write(f"总校验图片数：{stat['total_img']}\n")
        f.write(f"完全合法样本数：{stat['full_valid']}\n")
        f.write(f"样本合法率：{stat['full_valid'] / stat['total_img'] * 100:.2f}%\n\n")
        for error_type, file_list in error_logs.items():
            f.write(f"----- {error_type} 共{len(file_list)}个 -----\n")
            for file_name in file_list:
                f.write(f"{file_name}\n")
            f.write("\n")
    print(f"📋 详细错误日志已保存至：{log_file}")

    return stat, error_logs


def main():
    print("🚀 KAIST数据集全量标签-图片对齐校验脚本")
    print(f"📌 校验范围：训练集/测试集、红外/可见光全量样本")
    print(f"📌 校验内容：标签匹配、格式合法性、坐标有效性、可视化标注对齐")

    # 加载类别名称
    class_names = load_classes(YAML_PATH)
    print(f"📚 数据集类别：{class_names}")

    # 创建保存目录
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # 待校验的子集配置
    subset_configs = [
        ("train", "lwir"),
        ("train", "visible"),
        ("val", "lwir"),
        ("val", "visible")
    ]

    # 全局统计变量
    global_total_img = 0
    global_full_valid = 0
    all_stats = {}
    all_error_logs = {}

    # 遍历所有子集执行校验
    for subset_name, modal_name in subset_configs:
        result = process_single_subset(subset_name, modal_name, class_names, SAVE_ROOT)
        if result is not None:
            stat, error_logs = result
            all_stats[f"{subset_name}_{modal_name}"] = stat
            all_error_logs[f"{subset_name}_{modal_name}"] = error_logs
            global_total_img += stat["total_img"]
            global_full_valid += stat["full_valid"]
        print(f"\n{'-' * 80}\n")

    # 全局汇总结果输出
    print(f"\n{'=' * 80}")
    print(f"📈 全局校验汇总结果")
    print(f"{'=' * 80}")
    print(f"全局总校验图片数：{global_total_img}")
    print(f"全局完全合法无错误样本数：{global_full_valid}")
    print(f"全局整体样本合法率：{global_full_valid / global_total_img * 100:.2f}%")
    print(f"\n📂 所有可视化结果与错误日志已保存至：{SAVE_ROOT} 文件夹")

    # 保存全局汇总日志
    global_log_file = os.path.join(SAVE_ROOT, "global_check_summary.txt")
    with open(global_log_file, 'w', encoding='utf-8') as f:
        f.write("===== KAIST数据集全量标签校验全局汇总报告 =====\n")
        f.write(f"全局总校验图片数：{global_total_img}\n")
        f.write(f"全局完全合法无错误样本数：{global_full_valid}\n")
        f.write(f"全局整体样本合法率：{global_full_valid / global_total_img * 100:.2f}%\n\n")
        f.write("===== 各子集详细统计 =====\n")
        for subset_key, stat in all_stats.items():
            f.write(f"\n【{subset_key}】\n")
            for k, v in stat.items():
                f.write(f"  {k}: {v}\n")
    print(f"📋 全局汇总报告已保存至：{global_log_file}")
    print("🎉 全量校验任务执行完成！")


if __name__ == "__main__":
    main()