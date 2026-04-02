import os

# ===================== 配置项（无需修改，适配你的路径） =====================
# 所有需要处理的标签文件夹
label_folders = [
    r"E:\KAIST_Dataset\kaist_yolo\labels\lwir\train",
    r"E:\KAIST_Dataset\kaist_yolo\labels\lwir\val",
    r"E:\KAIST_Dataset\kaist_yolo\labels\visible\train",
    r"E:\KAIST_Dataset\kaist_yolo\labels\visible\val"
]

# KAIST数据集固定图片尺寸（宽640px，高512px）
IMG_WIDTH = 640
IMG_HEIGHT = 512

# 类别映射：仅person一个类别，对应ID 0
CLASS_MAP = {"person": 0}

# ===================== 转换逻辑 =====================
for folder in label_folders:
    # 校验文件夹是否存在
    if not os.path.exists(folder):
        print(f"跳过不存在的文件夹：{folder}")
        continue

    print(f"\n========== 开始处理：{folder} ==========")
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    success_count = 0
    error_count = 0

    for txt_file in txt_files:
        txt_path = os.path.join(folder, txt_file)

        try:
            # 读取原始标签
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            yolo_lines = []
            for line in lines:
                line = line.strip()
                # 跳过空行、注释行
                if not line or line.startswith("%"):
                    continue

                # 拆分原始行，取前5个有效字段
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_name, x1, y1, w, h = parts[:5]
                # 只处理person类别
                if class_name not in CLASS_MAP:
                    continue

                # 转换为数字
                try:
                    x1 = float(x1)
                    y1 = float(y1)
                    w = float(w)
                    h = float(h)
                except ValueError:
                    continue

                # 转换为YOLO格式：中心坐标+归一化
                x_center = (x1 + w / 2) / IMG_WIDTH
                y_center = (y1 + h / 2) / IMG_HEIGHT
                w_norm = w / IMG_WIDTH
                h_norm = h / IMG_HEIGHT
                class_id = CLASS_MAP[class_name]

                # 拼接YOLO格式行，保留6位小数
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines.append(yolo_line)

            # 覆盖写入转换后的内容
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            success_count += 1

        except Exception as e:
            print(f"处理失败：{txt_file}，错误：{str(e)}")
            error_count += 1

    print(f"处理完成！成功：{success_count} 个，失败：{error_count} 个")

print("\n========== 所有标签转换完成！==========")