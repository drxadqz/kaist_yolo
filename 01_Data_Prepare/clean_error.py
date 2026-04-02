import os

# ====================== 路径配置 ======================
# 标签根目录
LABEL_ROOT = r"E:\KAIST_Dataset\kaist_yolo\labels"
# 待处理的模态
MODALS = ["lwir", "visible"]
# 数据集子集
SUBSET = "train"
# 待处理的标签文件名列表
ERROR_LABELS = [
    "set00_V004_I01233.txt",
    "set00_V007_I00263.txt",
    "set01_V000_I02017.txt",
    "set01_V003_I00905.txt",
    "set03_V001_I01005.txt",
    "set04_V001_I01229.txt",
    "set04_V001_I01231.txt",
    "set04_V001_I01233.txt"
]
# ======================================================

for modal in MODALS:
    label_dir = os.path.join(LABEL_ROOT, modal, SUBSET)
    print(f"\n开始处理【{modal}】模态...")
    for label_name in ERROR_LABELS:
        label_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_path):
            # 清空文件内容
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("")
            print(f"✅ 已清空：{label_path}")
        else:
            print(f"⚠️  文件不存在：{label_path}")

print("\n🎉 所有标签文件处理完成！")