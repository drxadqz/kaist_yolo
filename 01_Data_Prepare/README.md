# 01_Data_Prepare

Data preprocessing scripts for KAIST multispectral dataset.

## Typical responsibilities

- Organize image folders for train/val splits
- Organize label folders to match image names
- Convert KAIST annotations to YOLO format
- Sanity-check labels and fix invalid boxes

## Suggested execution flow

1. `organize_images_train.py` / `orgnize_images_val.py`
2. `organize_labels.py` / `copy_labels.py`
3. `convert_kaist_to_yolo.py`
4. `yanzheng.py` and `clean_error.py`

## Output expectation

- Final YOLO-ready structure under:
  - `E:/KAIST_Dataset/kaist_yolo/images/...`
  - `E:/KAIST_Dataset/kaist_yolo/labels/...`

## Notes

- Keep train/val split fixed once detector training begins.
- Any label filtering policy should be documented in paper experiment settings.

