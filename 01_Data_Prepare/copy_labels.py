import os
import shutil

# ===================== 仅需修改这里的路径 =====================
# 整理后的标签根目录（里面有train和val的那个）
LABEL_ROOT = r"E:\KAIST_Dataset\kaist_yolo\labels"
# =================================================================

# 两种模态
MODALITIES = ["lwir", "visible"]

print("开始复制标签，分成红外和可见光两份...")
print("=" * 60)

for modality in MODALITIES:
    # 创建目标目录
    target_train = os.path.join(LABEL_ROOT, modality, "train")
    target_val = os.path.join(LABEL_ROOT, modality, "val")
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_val, exist_ok=True)

    # 源目录
    src_train = os.path.join(LABEL_ROOT, "train")
    src_val = os.path.join(LABEL_ROOT, "val")

    # 复制训练集标签
    count_train = 0
    if os.path.exists(src_train):
        for f in os.listdir(src_train):
            if f.endswith(".txt"):
                shutil.copy2(os.path.join(src_train, f), os.path.join(target_train, f))
                count_train += 1

    # 复制验证集标签
    count_val = 0
    if os.path.exists(src_val):
        for f in os.listdir(src_val):
            if f.endswith(".txt"):
                shutil.copy2(os.path.join(src_val, f), os.path.join(target_val, f))
                count_val += 1

    print(f"✅ {modality} 标签复制完成：训练集 {count_train} 个，验证集 {count_val} 个")

print("=" * 60)
print("🎉 所有标签复制完成！")
print("=" * 60)