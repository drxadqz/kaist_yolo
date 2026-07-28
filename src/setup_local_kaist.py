from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
DEFAULT_TARGET_ROOT = REPO_ROOT / "datasets" / "KAIST_local_strict"
EVAL_JSON = SOURCE_DIR / "evaluation_script" / "KAIST_annotation.json"
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 512.0
POSITIVE_LABELS = {"person"}


def _expected_image_links(image_source_root: Path) -> dict[Path, Path]:
    return {
        Path("visible") / "train": image_source_root / "images" / "visible" / "train",
        Path("visible") / "test": image_source_root / "images" / "visible" / "val",
        Path("infrared") / "train": image_source_root / "images" / "lwir" / "train",
        Path("infrared") / "test": image_source_root / "images" / "lwir" / "val",
    }


def _expected_annotation_links(train_annotation_root: Path, test_annotation_root: Path) -> dict[Path, Path]:
    return {
        Path("annotations_raw") / "train": train_annotation_root,
        Path("annotations_raw") / "test": test_annotation_root,
    }


def _same_target(link_path: Path, target_path: Path) -> bool:
    if not link_path.exists():
        return False
    try:
        return link_path.resolve(strict=True) == target_path.resolve(strict=True)
    except OSError:
        return False


def _create_dir_link(link_path: Path, target_path: Path) -> None:
    if not target_path.is_dir():
        raise FileNotFoundError(f"Missing source directory: {target_path}")

    if link_path.exists() or link_path.is_symlink():
        if _same_target(link_path, target_path):
            print(f"[Skip] {link_path} -> {target_path}")
            return
        raise FileExistsError(
            f"Target link path already exists and points elsewhere: {link_path}\n"
            f"Expected source: {target_path}"
        )

    link_path.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        cmd = ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        if stdout:
            print(stdout)
    else:
        os.symlink(target_path, link_path, target_is_directory=True)
        print(f"[Link] {link_path} -> {target_path}")


def _count_files(path: Path, suffix: str) -> int:
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower())


def _load_expected_test_order(eval_json_path: Path) -> list[str]:
    with eval_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    expected = []
    for item in data["images"]:
        image_name = item["im_name"].replace("/", "_")
        expected.append(f"{image_name}.txt")
    return expected


def _parse_annotation_file(annotation_path: Path) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            label = parts[0]
            x, y, w, h = map(float, parts[1:5])
            occ = int(float(parts[5])) if len(parts) > 5 else 0
            objects.append(
                {
                    "label": label,
                    "bbox": [x, y, w, h],
                    "occlusion": occ,
                    "raw_parts": parts,
                }
            )
    return objects


def _clip_box_xywh(box: list[float]) -> list[float]:
    x, y, w, h = box
    x1 = min(max(x, 0.0), IMAGE_WIDTH)
    y1 = min(max(y, 0.0), IMAGE_HEIGHT)
    x2 = min(max(x + w, 0.0), IMAGE_WIDTH)
    y2 = min(max(y + h, 0.0), IMAGE_HEIGHT)
    return [x1, y1, x2 - x1, y2 - y1]


def _xywh_to_yolo(box: list[float]) -> str:
    x, y, w, h = box
    cx = (x + w * 0.5) / IMAGE_WIDTH
    cy = (y + h * 0.5) / IMAGE_HEIGHT
    wn = w / IMAGE_WIDTH
    hn = h / IMAGE_HEIGHT
    return f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"


def _clear_txt_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in out_dir.glob("*.txt"):
        txt_file.unlink()


def _remove_cache_file(cache_path: Path) -> None:
    if cache_path.exists():
        cache_path.unlink()
        print(f"[Cache] removed {cache_path}")


def _build_split_labels(
    split_name: str,
    image_dir: Path,
    annotation_root: Path,
    labels_out_dir: Path,
    ignore_out_dir: Path,
) -> dict[str, object]:
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise RuntimeError(f"No images found for split '{split_name}' in {image_dir}")

    _clear_txt_dir(labels_out_dir)
    _clear_txt_dir(ignore_out_dir)

    raw_label_hist: Counter[str] = Counter()
    written_boxes = 0
    raw_objects = 0
    empty_label_files = 0
    missing_annotations = 0
    dropped_non_person = 0
    dropped_invalid_person = 0
    dropped_invalid_other = 0
    written_ignore_boxes = 0
    empty_ignore_files = 0

    for image_path in image_files:
        annotation_path = annotation_root / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            missing_annotations += 1
            raise FileNotFoundError(
                f"Missing annotation for {split_name} image {image_path.name}: {annotation_path}"
            )

        objects = _parse_annotation_file(annotation_path)
        raw_objects += len(objects)

        yolo_lines: list[str] = []
        ignore_lines: list[str] = []
        for obj in objects:
            label = str(obj["label"])
            raw_label_hist[label] += 1
            clipped = _clip_box_xywh(list(obj["bbox"]))  # type: ignore[arg-type]
            if clipped[2] <= 0.0 or clipped[3] <= 0.0:
                if label in POSITIVE_LABELS:
                    dropped_invalid_person += 1
                else:
                    dropped_invalid_other += 1
                continue

            if label in POSITIVE_LABELS:
                yolo_lines.append(_xywh_to_yolo(clipped))
                written_boxes += 1
            else:
                ignore_lines.append(_xywh_to_yolo(clipped))
                written_ignore_boxes += 1
                dropped_non_person += 1

        if not yolo_lines:
            empty_label_files += 1
        if not ignore_lines:
            empty_ignore_files += 1

        out_path = labels_out_dir / f"{image_path.stem}.txt"
        out_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        ignore_path = ignore_out_dir / f"{image_path.stem}.txt"
        ignore_path.write_text("\n".join(ignore_lines), encoding="utf-8")

    manifest = {
        "split": split_name,
        "image_dir": str(image_dir),
        "annotation_root": str(annotation_root),
        "label_dir": str(labels_out_dir),
        "ignore_label_dir": str(ignore_out_dir),
        "image_size": [int(IMAGE_WIDTH), int(IMAGE_HEIGHT)],
        "conversion_rule": {
            "positive_labels_written_to_yolo": sorted(POSITIVE_LABELS),
            "other_labels_are_written_to_labels_ignore": True,
            "boxes_are_clipped_to_image_bounds": True,
            "nonpositive_boxes_after_clipping_are_dropped": True,
            "raw_bbgt_annotations_are_preserved_via_annotations_raw_links": True,
        },
        "summary": {
            "images": len(image_files),
            "raw_objects": raw_objects,
            "positive_boxes_written": written_boxes,
            "ignore_boxes_written": written_ignore_boxes,
            "empty_label_files": empty_label_files,
            "empty_ignore_files": empty_ignore_files,
            "missing_annotations": missing_annotations,
            "dropped_non_person_objects": dropped_non_person,
            "dropped_invalid_person_boxes": dropped_invalid_person,
            "dropped_invalid_other_boxes": dropped_invalid_other,
            "raw_label_hist": dict(sorted(raw_label_hist.items())),
        },
    }
    print(
        f"[Labels:{split_name}] images={len(image_files)} written_boxes={written_boxes} "
        f"ignore_boxes={written_ignore_boxes} empty={empty_label_files} raw_objects={raw_objects}"
    )
    return manifest


def _write_manifests(target_root: Path, manifests: list[dict[str, object]]) -> None:
    meta_dir = target_root / "annotation_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    combined = {"manifests": manifests}
    for manifest in manifests:
        split = str(manifest["split"])
        (meta_dir / f"{split}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (meta_dir / "dataset_manifest.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Meta] wrote manifests to {meta_dir}")


def _verify_counts(target_root: Path) -> None:
    train_rgb = target_root / "visible" / "train"
    test_rgb = target_root / "visible" / "test"
    train_ir = target_root / "infrared" / "train"
    test_ir = target_root / "infrared" / "test"
    train_labels = target_root / "labels" / "train"
    test_labels = target_root / "labels" / "test"
    train_ignore = target_root / "labels_ignore" / "train"
    test_ignore = target_root / "labels_ignore" / "test"
    train_ann = target_root / "annotations_raw" / "train"
    test_ann = target_root / "annotations_raw" / "test"

    summary = {
        "visible/train": _count_files(train_rgb, ".jpg"),
        "visible/test": _count_files(test_rgb, ".jpg"),
        "infrared/train": _count_files(train_ir, ".jpg"),
        "infrared/test": _count_files(test_ir, ".jpg"),
        "labels/train": _count_files(train_labels, ".txt"),
        "labels/test": _count_files(test_labels, ".txt"),
        "labels_ignore/train": _count_files(train_ignore, ".txt"),
        "labels_ignore/test": _count_files(test_ignore, ".txt"),
        "annotations_raw/train": _count_files(train_ann, ".txt"),
        "annotations_raw/test": _count_files(test_ann, ".txt"),
    }

    print("[Counts]")
    for name, value in summary.items():
        print(f"  {name}: {value}")

    expected_train = 7601
    expected_test = 2252
    if (
        summary["visible/train"] != expected_train
        or summary["infrared/train"] != expected_train
        or summary["labels/train"] != expected_train
        or summary["labels_ignore/train"] != expected_train
        or summary["annotations_raw/train"] != expected_train
        or summary["visible/test"] != expected_test
        or summary["infrared/test"] != expected_test
        or summary["labels/test"] != expected_test
        or summary["labels_ignore/test"] != expected_test
        or summary["annotations_raw/test"] != expected_test
    ):
        raise RuntimeError(
            "Strict KAIST local layout counts do not match the expected 7601 train / 2252 test protocol."
        )


def _verify_test_order(target_root: Path, eval_json_path: Path) -> None:
    labels_test_dir = target_root / "labels" / "test"
    actual = sorted(p.name for p in labels_test_dir.glob("*.txt"))
    expected = _load_expected_test_order(eval_json_path)

    if len(actual) != len(expected):
        raise RuntimeError(
            f"Test label count mismatch: local={len(actual)} vs eval_json={len(expected)}"
        )

    mismatches = []
    for idx, (a_name, e_name) in enumerate(zip(actual, expected)):
        if a_name != e_name:
            mismatches.append((idx, a_name, e_name))
            if len(mismatches) >= 10:
                break

    if mismatches:
        details = "\n".join(
            f"  idx={idx}: local={local_name}, expected={expected_name}"
            for idx, local_name, expected_name in mismatches
        )
        raise RuntimeError(
            "Local test-label ordering does not match DeformCAT KAIST evaluation order.\n"
            + details
        )

    print(f"[Order] labels/test matches {len(expected)}-image KAIST evaluation order")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a strict KAIST local dataset view for DeformCAT: "
            "reuse local RGB/LWIR images, regenerate train labels from sanitized training annotations, "
            "regenerate test labels from the official 2252-image test annotations, and preserve raw bbGt files."
        )
    )
    parser.add_argument(
        "--image-source-root",
        type=Path,
        required=True,
        help="Existing local KAIST RGB/LWIR image root",
    )
    parser.add_argument(
        "--train-annotation-root",
        type=Path,
        required=True,
        help="Flat sanitized KAIST training annotation root",
    )
    parser.add_argument(
        "--test-annotation-root",
        type=Path,
        required=True,
        help="Flat official KAIST 2252-image test annotation root",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Strict DeformCAT-local dataset root to create",
    )
    parser.add_argument(
        "--skip-order-check",
        action="store_true",
        help="Skip the strict 2252-image order verification against evaluation_script/KAIST_annotation.json",
    )
    args = parser.parse_args()

    image_source_root = args.image_source_root.resolve()
    train_annotation_root = args.train_annotation_root.resolve()
    test_annotation_root = args.test_annotation_root.resolve()
    target_root = args.target_root.resolve()

    print(f"[Image Source] {image_source_root}")
    print(f"[Train Annotations] {train_annotation_root}")
    print(f"[Test Annotations] {test_annotation_root}")
    print(f"[Target] {target_root}")

    for relative_link, source_path in _expected_image_links(image_source_root).items():
        _create_dir_link(target_root / relative_link, source_path)

    for relative_link, source_path in _expected_annotation_links(
        train_annotation_root, test_annotation_root
    ).items():
        _create_dir_link(target_root / relative_link, source_path)

    train_manifest = _build_split_labels(
        split_name="train",
        image_dir=target_root / "visible" / "train",
        annotation_root=train_annotation_root,
        labels_out_dir=target_root / "labels" / "train",
        ignore_out_dir=target_root / "labels_ignore" / "train",
    )
    test_manifest = _build_split_labels(
        split_name="test",
        image_dir=target_root / "visible" / "test",
        annotation_root=test_annotation_root,
        labels_out_dir=target_root / "labels" / "test",
        ignore_out_dir=target_root / "labels_ignore" / "test",
    )
    _write_manifests(target_root, [train_manifest, test_manifest])

    _remove_cache_file(target_root / "labels" / "train.cache")
    _remove_cache_file(target_root / "labels" / "test.cache")

    _verify_counts(target_root)

    if not args.skip_order_check:
        _verify_test_order(target_root, EVAL_JSON)

    print("[Done] Strict local KAIST dataset layout is ready for DeformCAT")
    print(f"[YAML] Use {REPO_ROOT / 'configs' / 'datasets' / 'kaist.example.yaml'}")


if __name__ == "__main__":
    main()
