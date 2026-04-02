import os

# 核心配置：你的kaist_yolo根目录，无需修改
root_dir = r"E:\KAIST_Dataset\kaist_yolo"

# 处理红外(lwir)和可见光(visible)两个模态
modalities = ["lwir", "visible"]

for modality in modalities:
    # 适配你的目录结构：标签/图片路径
    label_train_path = os.path.join(root_dir, "labels", modality, "train")
    image_train_path = os.path.join(root_dir, "images", modality, "train")

    # 路径合法性校验
    if not os.path.exists(label_train_path):
        print(f"跳过【{modality}】：标签路径不存在 {label_train_path}")
        continue
    if not os.path.exists(image_train_path):
        print(f"跳过【{modality}】：图片路径不存在 {image_train_path}")
        continue

    # 读取并排序标签（仅txt）、图片（仅jpg）
    label_list = sorted([f for f in os.listdir(label_train_path) if f.endswith(".txt")])
    image_list = sorted(
        [f for f in os.listdir(image_train_path) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])  # 按数字序号升序，适配你1.jpg/2.jpg的命名
    )

    # 数量一致性校验
    if len(label_list) != len(image_list):
        print(f"警告【{modality}】：标签数({len(label_list)})≠图片数({len(image_list)})，跳过处理")
        continue

    # 批量重命名
    print(f"开始处理【{modality}】训练集，共{len(label_list)}个样本")
    for img_file, label_file in zip(image_list, label_list):
        old_img_full = os.path.join(image_train_path, img_file)
        # 新图片名=标签名，后缀改为.jpg，保证和标签完全匹配
        new_img_name = os.path.splitext(label_file)[0] + ".jpg"
        new_img_full = os.path.join(image_train_path, new_img_name)

        os.rename(old_img_full, new_img_full)
        print(f"重命名完成：{img_file} → {new_img_name}")

    print(f"【{modality}】训练集处理完成\n")

print("所有模态重命名任务结束！")