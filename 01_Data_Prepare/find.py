import os
import shutil

# ====================== 路径配置 ======================
# 标注文件所在文件夹
ANNOTATION_DIR = r"E:\KAIST行人数据集\KAIST_testset_annotations\annotations_KAIST_test_set"
# 源数据集-红外(lwir)图片根目录（自动递归所有子文件夹）
LWIR_SRC_ROOT = r"E:\KAIST_Dataset\kaist_full\images\lwir"
# 源数据集-可见光(visible)图片根目录（自动递归所有子文件夹）
VISIBLE_SRC_ROOT = r"E:\KAIST_Dataset\kaist_full\images\visible"
# 目标文件夹-红外val集
LWIR_DEST_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\lwir\val"
# 目标文件夹-可见光val集
VISIBLE_DEST_DIR = r"E:\KAIST_Dataset\kaist_yolo\images\visible\val"
# 预期单类别图片数量
EXPECTED_SINGLE_COUNT = 2252
# ========================================================

def build_simple_index(src_root_dir: str) -> dict:
    """
    递归扫描源目录，构建极简索引：key = 帧号(IXXXXX)，value = 图片完整路径
    """
    image_index = {}
    for root, _, files in os.walk(src_root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # 只取文件名（不含后缀）作为 key，例如 I01539.jpg -> key = "I01539"
                frame_name = os.path.splitext(file)[0]
                # 如果有重复帧号，只保留第一个找到的（避免覆盖）
                if frame_name not in image_index:
                    image_index[frame_name] = os.path.join(root, file)
    print(f"✅ 完成扫描，共在 {src_root_dir} 下找到 {len(image_index)} 张图片")
    return image_index

def main():
    # 1. 自动创建目标文件夹
    os.makedirs(LWIR_DEST_DIR, exist_ok=True)
    os.makedirs(VISIBLE_DEST_DIR, exist_ok=True)

    # 2. 扫描标注文件
    anno_file_list = [f for f in os.listdir(ANNOTATION_DIR) if f.lower().endswith(".txt")]
    total_anno_num = len(anno_file_list)
    print("\n" + "="*70)
    print(f"📄 扫描到标注文件总数: {total_anno_num} 个（预期 {EXPECTED_SINGLE_COUNT} 个）")
    print("="*70)

    # 3. 构建源图片索引（仅按帧号）
    print("\n🔍 正在扫描源数据集图片...")
    lwir_index = build_simple_index(LWIR_SRC_ROOT)
    visible_index = build_simple_index(VISIBLE_SRC_ROOT)

    # 4. 初始化统计
    lwir_success = 0
    visible_success = 0
    failed_log = []

    # 5. 遍历标注文件并复制
    print("\n📋 开始复制图片...")
    for filename in anno_file_list:
        # 提取文件名前缀，然后按下划线分割，取最后一段作为帧号（IXXXXX）
        file_prefix = os.path.splitext(filename)[0]
        parts = file_prefix.split("_")
        if not parts:
            failed_log.append(f"文件名解析失败: {filename}")
            continue
        frame_name = parts[-1] # 只取最后一段，例如 set06_V000_I01539 -> "I01539"

        # 目标文件名（保持完整前缀，避免重名覆盖）
        dest_filename = f"{file_prefix}.jpg"

        # 复制 lwir
        if frame_name in lwir_index:
            try:
                shutil.copy2(lwir_index[frame_name], os.path.join(LWIR_DEST_DIR, dest_filename))
                lwir_success += 1
            except Exception as e:
                failed_log.append(f"lwir复制失败 [{frame_name}]: {str(e)}")
        else:
            failed_log.append(f"lwir未找到帧号: {frame_name}")

        # 复制 visible
        if frame_name in visible_index:
            try:
                shutil.copy2(visible_index[frame_name], os.path.join(VISIBLE_DEST_DIR, dest_filename))
                visible_success += 1
            except Exception as e:
                failed_log.append(f"visible复制失败 [{frame_name}]: {str(e)}")
        else:
            failed_log.append(f"visible未找到帧号: {frame_name}")

    # 6. 输出结果
    print("\n" + "="*70)
    print("✅ 任务完成统计：")
    print(f"红外(lwir): {lwir_success} / {EXPECTED_SINGLE_COUNT}")
    print(f"可见光(visible): {visible_success} / {EXPECTED_SINGLE_COUNT}")
    print(f"总计: {lwir_success + visible_success} / {EXPECTED_SINGLE_COUNT * 2}")
    print("="*70)

    if lwir_success == EXPECTED_SINGLE_COUNT and visible_success == EXPECTED_SINGLE_COUNT:
        print("🎉 完美！数量完全符合预期！")
    else:
        print("\n⚠️ 存在缺失文件，明细如下：")
        for log in failed_log:
            print(f"- {log}")

if __name__ == "__main__":
    main()