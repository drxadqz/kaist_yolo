import os
import shutil

# ===================== 仅需修改这里的路径 =====================
# 之前转换好的YOLO标签根目录
ORIGIN_LABEL_ROOT = r"E:\KAIST_Dataset\kaist_yolo\labels"
# 整理后的标签保存目录
SAVE_LABEL_ROOT = r"E:\KAIST_Dataset\kaist_yolo\labels"
# =================================================================

# 官方标准划分
TRAIN_SETS = [f"set{i:02d}" for i in range(0, 6)]
VAL_SETS = [f"set{i:02d}" for i in range(6, 12)]

# 创建保存目录
os.makedirs(os.path.join(SAVE_LABEL_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_LABEL_ROOT, "val"), exist_ok=True)

print("开始整理标签（共用一套）...")
print("=" * 60)

# 处理训练集标签
count_train = 0
for set_name in TRAIN_SETS:
    set_path = os.path.join(ORIGIN_LABEL_ROOT, set_name)
    if not os.path.exists(set_path):
        continue

    video_folders = [f for f in os.listdir(set_path) if f.startswith("V")]
    for video_name in video_folders:
        label_dir = os.path.join(set_path, video_name)
        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue

            src = os.path.join(label_dir, label_file)
            dst = os.path.join(SAVE_LABEL_ROOT, "train", label_file)
            shutil.copy2(src, dst)
            count_train += 1
            print(f"训练集标签: {count_train} 个", end="\r")

# 处理验证集标签
count_val = 0
for set_name in VAL_SETS:
    set_path = os.path.join(ORIGIN_LABEL_ROOT, set_name)
    if not os.path.exists(set_path):
        continue

    video_folders = [f for f in os.listdir(set_path) if f.startswith("V")]
    for video_name in video_folders:
        label_dir = os.path.join(set_path, video_name)
        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue

            src = os.path.join(label_dir, label_file)
            dst = os.path.join(SAVE_LABEL_ROOT, "val", label_file)
            shutil.copy2(src, dst)
            count_val += 1
            print(f"验证集标签: {count_val} 个", end="\r")

print("\n" + "=" * 60)
print(f"✅ 标签整理完成！")
print(f"训练集标签：{count_train} 个")
print(f"验证集标签：{count_val} 个")
print("=" * 60)