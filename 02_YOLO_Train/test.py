from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 训练（仅保留有效参数，无任何报错）
if __name__ == '__main__':
    model.train(
        data=r"E:\kaist_yolo\02_YOLO_Train\kaist_lwir.yaml",
        epochs=100,  # 训练轮数
        imgsz=640,  # 图片尺寸
        single_cls=True,  # 单类别必须开
        batch=8,
        device=0,
        workers=0
    )

    # 验证
    model.val()