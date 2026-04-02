import torch
import cv2
import pandas
import matplotlib
import seaborn
import scipy
import yaml
import tqdm
import ultralytics

print("="*50)
print("✅ 所有库安装成功！环境检查清单：")
print("="*50)
print(f"1. PyTorch 版本: {torch.__version__}")
print(f"2. GPU 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"3. 你的显卡型号: {torch.cuda.get_device_name(0)}")
print(f"4. OpenCV 版本: {cv2.__version__}")
print(f"5. Ultralytics (YOLO) 版本: {ultralytics.__version__}")
print(f"6. Pandas 版本: {pandas.__version__}")
print(f"7. Matplotlib 版本: {matplotlib.__version__}")
print(f"8. Seaborn 版本: {seaborn.__version__}")
print(f"9. SciPy 版本: {scipy.__version__}")
print("="*50)
print("🚀 环境配置全部完成！可以开始做毕设的数据处理了！")
print("="*50)