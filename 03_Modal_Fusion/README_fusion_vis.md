# fusion_vis 使用说明

该目录下的 `fusion_vis.py` 用于生成 KAIST RGB-T 三栏对比图：
- 左栏：RGB 单模态
- 中栏：IR/LWIR 单模态
- 右栏：Late Fusion

## 依赖

```bash
pip install ultralytics opencv-python numpy tqdm
```

## 快速开始

1. 打开 `fusion_vis.py`，修改顶部路径配置：
   - `TEST_VISIBLE_DIR`
   - `TEST_LWIR_DIR`
2. 先在 `__main__` 中配置一对同名样例图路径。
3. 运行脚本：

```bash
python fusion_vis.py
```

输出默认保存到 `./vis_results`。

## 单张调用

```python
from fusion_vis import vis_single_image_pair

vis_single_image_pair(
    visible_img_path=r"E:\\KAIST_Dataset\\kaist_yolo\\images\\visible\\val\\set09_V000_I01259.jpg",
    lwir_img_path=r"E:\\KAIST_Dataset\\kaist_yolo\\images\\lwir\\val\\set09_V000_I01259.jpg",
    save_name="paper_compare_sample.png",
)
```

## 批量调用

```python
from fusion_vis import batch_vis_from_dirs

batch_vis_from_dirs(max_images=100)
```

仅保存三路结果有明显差异的样本：

```python
from fusion_vis import batch_vis_from_dirs

batch_vis_from_dirs(
    max_images=300,
    only_save_diff=True,
    diff_iou_thresh=0.55,
    min_diff_boxes=1,
)
```

## 差异判定规则（only_save_diff=True 时生效）

- 三路框数量不一致，判定为有差异。
- 数量一致时，基于 IoU 贪心匹配统计未匹配框；未匹配数量达到 `min_diff_boxes` 判定为有差异。
- `diff_iou_thresh` 越大越严格，推荐 `0.5 ~ 0.6`。

## 已做的鲁棒性处理

- 空检测框直接返回，不会崩溃。
- 检测框张量统一走 `.detach().cpu().numpy()`，避免 CUDA 转 numpy 报错。
- 自动检查 RGB/IR 配对缺失并跳过，打印统计信息。

