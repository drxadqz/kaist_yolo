import os
import shutil
import re
from tqdm import tqdm

# ====================== 【仅需修改这里的路径参数】 ======================
# KAIST原始数据集images根目录（setXX文件夹所在的目录）
RAW_IMG_ROOT = r"E:\KAIST_Dataset\images"
# 标签文件根目录（lwir/visible文件夹所在的目录）
LABEL_ROOT = r"E:\KAIST_Dataset\kaist_yolo\labels"
# 目标图片输出根目录
TARGET_IMG_ROOT = r"E:\KAIST_Dataset\kaist_yolo\images"
# 预期单模态val集总数量
EXPECTED_TOTAL = 2252
# 图片支持的后缀格式（覆盖KAIST数据集所有可能的图片格式）
IMG_SUFFIX = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
# 是否覆盖目标目录已存在的文件（True=覆盖，False=跳过已存在文件）
OVERWRITE_EXIST = False
# =========================================================================

# 标签文件名正则匹配：严格匹配 setXX_VXXX_IXXXXXX 格式
LABEL_NAME_PATTERN = re.compile(r'^(set\d+)_(V\d+)_(I\d+)$')


def parse_label_name(label_prefix):
    """
    从标签文件名前缀拆解出set目录、V目录、图片ID
    输入：标签文件名前缀（如 set00_V000_I01225）
    输出：(set_dir, v_dir, img_id) 匹配失败返回 None
    """
    match_result = LABEL_NAME_PATTERN.match(label_prefix)
    if not match_result:
        return None
    return match_result.groups()


def process_single_modal(modal_name):
    """
    处理单个模态（lwir/visible）的验证集图片匹配、复制、重命名
    """
    print(f"\n{'=' * 80}")
    print(f"🚀 开始处理【{modal_name}】模态验证集")
    print(f"{'=' * 80}")

    # 路径拼接
    label_dir = os.path.join(LABEL_ROOT, modal_name, "val")
    target_img_dir = os.path.join(TARGET_IMG_ROOT, modal_name, "val")
    # 路径合法性校验
    if not os.path.exists(label_dir):
        print(f"❌ 标签目录不存在：{label_dir}")
        return 0, [], []
    # 创建目标目录
    os.makedirs(target_img_dir, exist_ok=True)

    # 1. 获取所有标签文件，严格按文件名排序（保证顺序和标签目录完全一致）
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.txt')])
    total_label_num = len(label_files)
    print(f"📄 标签目录共找到 {total_label_num} 个标签文件")
    if total_label_num != EXPECTED_TOTAL:
        print(f"⚠️  标签文件数量与预期 {EXPECTED_TOTAL} 不符，请检查标签目录！")

    # 2. 统计变量初始化
    success_count = 0
    skip_count = 0
    missing_list = []
    format_error_list = []

    # 3. 遍历所有标签文件，按顺序处理
    for label_file in tqdm(label_files, desc=f"处理{modal_name}验证集图片"):
        # 拆解标签文件名
        label_prefix, label_suffix = os.path.splitext(label_file)
        parse_result = parse_label_name(label_prefix)

        # 文件名格式错误
        if not parse_result:
            format_error_list.append(label_file)
            missing_list.append(label_file)
            continue

        set_dir, v_dir, img_id = parse_result

        # 拼接原始图片目录路径
        raw_img_dir = os.path.join(RAW_IMG_ROOT, set_dir, v_dir, modal_name)
        if not os.path.exists(raw_img_dir):
            missing_list.append(label_file)
            continue

        # 查找对应ID的图片文件
        target_img_file = None
        for file in os.listdir(raw_img_dir):
            file_prefix, file_suffix = os.path.splitext(file)
            # 匹配图片ID，且后缀为支持的图片格式
            if file_prefix == img_id and file_suffix.lower() in IMG_SUFFIX:
                target_img_file = file
                break

        # 未找到对应图片
        if not target_img_file:
            missing_list.append(label_file)
            continue

        # 拼接原始图片完整路径、目标图片完整路径
        raw_img_path = os.path.join(raw_img_dir, target_img_file)
        img_suffix = os.path.splitext(target_img_file)[1]
        target_img_path = os.path.join(target_img_dir, f"{label_prefix}{img_suffix}")

        # 跳过已存在的文件
        if os.path.exists(target_img_path) and not OVERWRITE_EXIST:
            skip_count += 1
            success_count += 1
            continue

        # 复制文件
        try:
            shutil.copy2(raw_img_path, target_img_path)
            success_count += 1
        except Exception as e:
            print(f"\n❌ 复制失败：{label_file}，错误信息：{str(e)}")
            missing_list.append(label_file)

    # 4. 结果统计与输出
    print(f"\n✅ {modal_name} 模态验证集处理完成！")
    print(f"📊 统计结果：")
    print(f"  总标签数：{total_label_num}")
    print(f"  成功匹配复制：{success_count} 张")
    print(f"  跳过已存在文件：{skip_count} 张")
    print(f"  匹配失败/缺失：{len(missing_list)} 张")
    print(f"  文件名格式错误：{len(format_error_list)} 个")

    # 数量校验
    if success_count == EXPECTED_TOTAL:
        print(f"🎉 数量校验通过！与预期 {EXPECTED_TOTAL} 张完全一致")
    else:
        print(f"⚠️  数量校验不通过！预期 {EXPECTED_TOTAL} 张，实际成功 {success_count} 张")

    # 输出错误详情
    if format_error_list:
        print(f"\n📋 文件名格式错误的标签（前20个）：")
        for item in format_error_list[:20]:
            print(f"  - {item}")
        if len(format_error_list) > 20:
            print(f"  ... 剩余 {len(format_error_list) - 20} 个")

    if missing_list:
        print(f"\n📋 匹配失败的标签文件（前20个）：")
        for item in missing_list[:20]:
            print(f"  - {item}")
        if len(missing_list) > 20:
            print(f"  ... 剩余 {len(missing_list) - 20} 个")

    # 保存错误日志
    log_dir = "kaist_val_dataset_fix_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{modal_name}_val_fix_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"===== {modal_name} 验证集图片匹配日志 =====\n")
        f.write(f"总标签数：{total_label_num}\n")
        f.write(f"成功匹配数：{success_count}\n")
        f.write(f"跳过已存在数：{skip_count}\n")
        f.write(f"匹配失败数：{len(missing_list)}\n")
        f.write(f"格式错误数：{len(format_error_list)}\n\n")

        f.write("----- 匹配失败的标签文件 -----\n")
        for item in missing_list:
            f.write(f"{item}\n")
        f.write("\n----- 文件名格式错误的标签 -----\n")
        for item in format_error_list:
            f.write(f"{item}\n")
    print(f"\n📋 详细错误日志已保存至：{log_file}")

    return success_count, missing_list, format_error_list


def main():
    print("🚀 KAIST数据集验证集标签-图片精准对齐修复脚本")
    print(f"📌 核心规则：按标签文件名的 setXX_VXXX_IXXXXXX 层级匹配原始图片，保证100%一一对应")

    # 处理红外(lwir)和可见光(visible)两个模态
    modal_list = ["lwir", "visible"]
    total_success = 0
    all_missing = {}

    for modal in modal_list:
        sc, ml, fel = process_single_modal(modal)
        total_success += sc
        all_missing[modal] = ml
        print(f"\n{'-' * 80}\n")

    # 全局汇总
    print(f"📈 全局处理汇总结果")
    print(f"双模态总成功匹配图片数：{total_success}")
    total_missing = sum([len(v) for v in all_missing.values()])
    print(f"双模态总匹配失败数：{total_missing}")

    if total_missing == 0:
        print("🎉 所有模态验证集处理完成！标签与图片已100%一一对应，可直接用于YOLO验证")
    else:
        print("⚠️  存在匹配失败的文件，请查看上述日志排查原始数据集文件是否完整")


if __name__ == "__main__":
    main()