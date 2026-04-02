from ultralytics import YOLO
import torch
import os

# 确保工作目录和脚本所在目录一致（不变）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def train_rgb_baseline_resume():
    # 1. CUDA设备校验（不变）
    print("="*50)
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*50)

    # 2. 加载你实际路径下的last.pt（核心修改）
    resume_weight_path = r"E:\kaist_yolo\02_YOLO_Train\runs\detect\runs\visible_baseline\kaist_visible_yolo11n\weights\last.pt"
    # 校验权重文件是否存在
    if not os.path.exists(resume_weight_path):
        raise FileNotFoundError(f"❌ 权重文件不存在！请检查路径：{resume_weight_path}")
    model = YOLO(resume_weight_path)

    # 3. 启动训练（仅新增resume=True，其余参数和原脚本完全一致）
    results = model.train(
        data="kaist_visible.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
        seed=42,
        deterministic=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        val=True,
        save=True,
        save_period=10,
        patience=50,
        amp=True,
        plots=True,
        project="runs/visible_baseline",
        name="kaist_visible_yolo11n",
        exist_ok=True,  # 必须保留
        iou=0.5,
        conf=0.001,
        max_det=300,
        resume=True,  # 核心：开启恢复训练
    )

    # 4. 输出最终训练结果（不变）
    print("="*50)
    print("✅ 可见光单模态基线继续训练完成！")
    print(f"最终验证集 mAP50: {results.box.map50:.4f}")
    print(f"最终验证集 mAP50-95: {results.box.map:.4f}")
    print(f"结果保存路径: {results.save_dir}")
    print("="*50)

    return results

if __name__ == "__main__":
    train_rgb_baseline_resume()