import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


VALID_COLOR = (0, 190, 0)
IGNORE_COLOR = (150, 150, 150)
TP_COLOR = (0, 120, 255)
FP_COLOR = (255, 0, 0)


def box_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def box_ioa_dt_xywh(dt_box, ignore_box):
    """KAIST ignore matching uses intersection over detection area."""
    dx1, dy1, dw, dh = dt_box
    gx1, gy1, gw, gh = ignore_box
    dx2, dy2 = dx1 + dw, dy1 + dh
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    dt_area = max(dw * dh, 1e-9)
    return inter / dt_area


def is_reasonable_ignore(ann):
    x, y, w, h = ann["bbox"]
    return (
        ann.get("ignore", 0)
        or h < 55
        or ann.get("occlusion", 0) not in (0, 1)
        or x < 5
        or y < 5
        or x + w > 635
        or y + h > 507
    )


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {im["id"]: im for im in data["images"]}
    valid = defaultdict(list)
    ignored = defaultdict(list)
    for ann in data["annotations"]:
        (ignored if is_reasonable_ignore(ann) else valid)[ann["image_id"]].append(ann["bbox"])
    return images, valid, ignored


def load_predictions(path):
    preds = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(v) for v in line.strip().split(",")]
            image_id = int(vals[0]) - 1
            x, y, w, h, score = vals[1:6]
            preds[image_id].append({"box": [x, y, w, h], "score": score})
    for image_id in preds:
        preds[image_id].sort(key=lambda p: p["score"], reverse=True)
    return preds


def classify_predictions(images, valid, ignored, preds, iou_thr=0.5):
    matched = defaultdict(set)
    fp_by_image = Counter()
    fn_by_image = Counter()
    high_fp_rows = []
    summary = Counter()

    for image_id in sorted(images):
        image_preds = preds.get(image_id, [])
        for pred in image_preds:
            box = pred["box"]
            best_iou = 0.0
            best_gt = -1
            for gt_idx, gt in enumerate(valid.get(image_id, [])):
                if gt_idx in matched[image_id]:
                    continue
                iou = box_iou_xywh(box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            is_day = image_id < 1455
            split = "day" if is_day else "night"
            if best_iou >= iou_thr:
                matched[image_id].add(best_gt)
                summary[f"{split}_tp"] += 1
                pred["kind"] = "tp"
            elif any(box_ioa_dt_xywh(box, gt) >= iou_thr for gt in ignored.get(image_id, [])):
                summary[f"{split}_ignored_match"] += 1
                pred["kind"] = "ignored"
            else:
                summary[f"{split}_fp"] += 1
                fp_by_image[image_id] += 1
                pred["kind"] = "fp"
                pred["best_iou"] = best_iou
                high_fp_rows.append((pred["score"], image_id, box, best_iou))

        miss = len(valid.get(image_id, [])) - len(matched[image_id])
        if miss > 0:
            fn_by_image[image_id] = miss
            summary[("day" if image_id < 1455 else "night") + "_fn"] += miss

    high_fp_rows.sort(reverse=True, key=lambda x: x[0])
    return summary, fp_by_image, fn_by_image, high_fp_rows, preds


def image_path_from_id(images_root, im_name):
    return images_root / (im_name.replace("/", "_") + ".jpg")


def draw_xywh(draw, box, color, width=3):
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def draw_case(image_path, out_path, title, valid_boxes, ignored_boxes, preds, min_score=0.05):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for box in ignored_boxes:
        draw_xywh(draw, box, IGNORE_COLOR, 1)
    for box in valid_boxes:
        draw_xywh(draw, box, VALID_COLOR, 3)
    for pred in preds:
        if pred["score"] < min_score or pred.get("kind") == "ignored":
            continue
        color = TP_COLOR if pred.get("kind") == "tp" else FP_COLOR
        draw_xywh(draw, pred["box"], color, 2)
        x, y, _, _ = pred["box"]
        draw.text((x, max(0, y - 16)), f"{pred['kind']} {pred['score']:.2f}", fill=color, font=font)

    draw.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="KAIST result.txt file")
    parser.add_argument("--data", default="./data/multispectral/KAIST_local.yaml")
    parser.add_argument("--ann", default="./evaluation_script/KAIST_annotation.json")
    parser.add_argument("--name", default="analysis")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dataset_root = Path(data["path"])
    images_root = dataset_root / "visible" / "test"

    images, valid, ignored = load_annotations(args.ann)
    preds = load_predictions(args.result)
    summary, fp_by_image, fn_by_image, high_fp_rows, preds = classify_predictions(images, valid, ignored, preds)

    out_dir = Path("runs/error_analysis") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "summary.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("Summary\n")
        for key in sorted(summary):
            f.write(f"{key}: {summary[key]}\n")
        f.write("\nTop FP images\n")
        for image_id, count in fp_by_image.most_common(args.topk):
            split = "day" if image_id < 1455 else "night"
            f.write(f"{image_id:04d} {split} fp={count} {images[image_id]['im_name']}\n")
        f.write("\nTop FN images\n")
        for image_id, count in fn_by_image.most_common(args.topk):
            split = "day" if image_id < 1455 else "night"
            f.write(f"{image_id:04d} {split} fn={count} {images[image_id]['im_name']}\n")
        f.write("\nTop high-confidence FP\n")
        for score, image_id, box, best_iou in high_fp_rows[: args.topk]:
            split = "day" if image_id < 1455 else "night"
            box_s = " ".join(f"{v:.1f}" for v in box)
            f.write(f"{score:.4f} {image_id:04d} {split} best_iou={best_iou:.3f} box=[{box_s}] {images[image_id]['im_name']}\n")

    selected = []
    for image_id, _ in fp_by_image.most_common(args.topk):
        selected.append(image_id)
    for _, image_id, _, _ in high_fp_rows[: args.topk]:
        selected.append(image_id)

    seen = set()
    for image_id in selected:
        if image_id in seen:
            continue
        seen.add(image_id)
        im_name = images[image_id]["im_name"]
        image_path = image_path_from_id(images_root, im_name)
        if not image_path.exists():
            continue
        split = "day" if image_id < 1455 else "night"
        title = f"{image_id:04d} {split} {im_name} | green=GT red=FP blue=TP gray=ignored"
        out_path = out_dir / f"{image_id:04d}_{split}_{im_name.replace('/', '_')}.jpg"
        draw_case(
            image_path,
            out_path,
            title,
            valid.get(image_id, []),
            ignored.get(image_id, []),
            preds.get(image_id, []),
            min_score=args.min_score,
        )

    print(f"Saved analysis to {out_dir}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
