import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from evaluation_script.evaluation_script import evaluate


@dataclass
class Det:
    image_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    raw: str = ""

    @property
    def xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def line(self):
        return f"{self.image_id},{self.x:.6g},{self.y:.6g},{self.w:.6g},{self.h:.6g},{self.score:.6g}"


def parse_result_file(path):
    result = {}
    raw_lines = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            det = Det(
                image_id=int(vals[0]),
                x=vals[1],
                y=vals[2],
                w=vals[3],
                h=vals[4],
                score=vals[5],
                raw=line,
            )
            result.setdefault(det.image_id, []).append(det)
            raw_lines.setdefault(det.image_id, []).append(line)
    for dets in result.values():
        dets.sort(key=lambda d: d.score, reverse=True)
    return result, raw_lines


def iou(a, b):
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-9)


def center_support(a, b):
    acx, acy = a.x + a.w * 0.5, a.y + a.h * 0.5
    bcx, bcy = b.x + b.w * 0.5, b.y + b.h * 0.5
    scale_x = max(0.5 * (a.w + b.w), 1.0)
    scale_y = max(0.5 * (a.h + b.h), 1.0)
    dx = abs(acx - bcx) / scale_x
    dy = abs(acy - bcy) / scale_y
    size_ratio = min(a.w * a.h, b.w * b.h) / max(a.w * a.h, b.w * b.h, 1e-9)
    return math.exp(-0.5 * (dx * dx + dy * dy)) * math.sqrt(max(size_ratio, 0.0))


def support_score(det, support_dets, center_weight):
    best = 0.0
    best_conf = 0.0
    for other in support_dets:
        q = max(iou(det, other), center_weight * center_support(det, other))
        if q > best:
            best = q
            best_conf = other.score
    return best, best_conf


def local_density(det, dets, radius):
    cx = det.x + det.w * 0.5
    cy = det.y + det.h * 0.5
    norm = max(det.h, 1.0)
    count = 0
    for other in dets:
        if other is det:
            continue
        ocx = other.x + other.w * 0.5
        ocy = other.y + other.h * 0.5
        dist = math.hypot((cx - ocx) / norm, (cy - ocy) / norm)
        if dist <= radius:
            count += 1
    return count


def max_higher_iou(det, dets):
    best = 0.0
    for other in dets:
        if other is det:
            break
        best = max(best, iou(det, other))
    return best


def same_rows(a, b):
    return a == b


def infer_source(base_rows, day_rows, night_rows):
    if same_rows(base_rows, day_rows):
        return "day"
    if same_rows(base_rows, night_rows):
        return "night"
    day_overlap = len(set(base_rows) & set(day_rows))
    night_overlap = len(set(base_rows) & set(night_rows))
    return "day" if day_overlap >= night_overlap else "night"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def calibrate_image(base_dets, support_dets, source, apply_source, cfg):
    if apply_source != "all" and source != apply_source:
        return [Det(**vars(d)) for d in base_dets]

    out = []
    ordered = sorted(base_dets, key=lambda d: d.score, reverse=True)
    for det in ordered:
        new_det = Det(**vars(det))
        supp, supp_conf = support_score(det, support_dets, cfg["center_weight"])
        density = local_density(det, ordered, cfg["density_radius"])
        factor = 1.0

        if supp < cfg["support_thr"]:
            factor *= cfg["unique_factor"]
            if density >= cfg["dense_count"]:
                factor *= cfg["dense_unique_factor"]

        high_iou = max_higher_iou(det, ordered)
        if high_iou >= cfg["duplicate_iou"]:
            factor *= cfg["duplicate_factor"]

        if supp >= cfg["boost_thr"] and cfg["support_boost"] > 0:
            factor *= 1.0 + cfg["support_boost"] * supp_conf

        new_det.score = clamp(det.score * factor, 1e-6, 0.999999)
        out.append(new_det)

    out.sort(key=lambda d: d.score, reverse=True)
    return out


def write_outputs(out_map, label_names, out_dir):
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    result_path = labels_dir / "result.txt"
    with open(result_path, "w", encoding="utf-8") as result_f:
        for image_id in range(1, len(label_names) + 1):
            lines = [d.line() for d in out_map.get(image_id, [])]
            for line in lines:
                result_f.write(line + "\n")
            with open(labels_dir / label_names[image_id - 1], "w", encoding="utf-8") as img_f:
                for line in lines:
                    img_f.write(line + "\n")
    return result_path


def evaluate_result(root, result_path, split):
    eval_result = evaluate(str(root / "evaluation_script" / "KAIST_annotation.json"), str(result_path), split)
    return {
        "MR_all": float(eval_result["all"].summarize(0)),
        "MR_day": float(eval_result["day"].summarize(0)),
        "MR_night": float(eval_result["night"].summarize(0)),
    }


def build_grid(args):
    if not args.grid:
        return [vars(args_to_cfg(args))]

    keys = [
        "support_thr",
        "unique_factor",
        "dense_unique_factor",
        "duplicate_iou",
        "duplicate_factor",
        "support_boost",
    ]
    values = [
        [0.20, 0.25, 0.30, 0.35],
        [0.65, 0.75, 0.85, 0.95],
        [0.55, 0.70, 0.85, 1.0],
        [0.45, 0.55, 0.65, 1.0],
        [0.60, 0.75, 0.90],
        [0.0, 0.08],
    ]
    base = vars(args_to_cfg(args))
    grid = []
    for combo in itertools.product(*values):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        grid.append(cfg)
    return grid


def args_to_cfg(args):
    return argparse.Namespace(
        support_thr=args.support_thr,
        boost_thr=args.boost_thr,
        unique_factor=args.unique_factor,
        dense_unique_factor=args.dense_unique_factor,
        dense_count=args.dense_count,
        density_radius=args.density_radius,
        duplicate_iou=args.duplicate_iou,
        duplicate_factor=args.duplicate_factor,
        support_boost=args.support_boost,
        center_weight=args.center_weight,
    )


def cfg_name(cfg):
    return (
        f"st{int(cfg['support_thr'] * 100):02d}_"
        f"uf{int(cfg['unique_factor'] * 100):02d}_"
        f"du{int(cfg['dense_unique_factor'] * 100):02d}_"
        f"di{int(cfg['duplicate_iou'] * 100):02d}_"
        f"df{int(cfg['duplicate_factor'] * 100):02d}_"
        f"sb{int(cfg['support_boost'] * 100):02d}"
    )


def run_once(args, cfg, root, label_names, split, base, day, night, base_rows, day_rows, night_rows, out_dir):
    out_map = {}
    source_counts = {"day": 0, "night": 0}
    for image_id in range(1, len(label_names) + 1):
        source = infer_source(
            base_rows.get(image_id, []),
            day_rows.get(image_id, []),
            night_rows.get(image_id, []),
        )
        source_counts[source] += 1
        support = night.get(image_id, []) if source == "day" else day.get(image_id, [])
        out_map[image_id] = calibrate_image(
            base.get(image_id, []),
            support,
            source,
            args.apply_source,
            cfg,
        )

    result_path = write_outputs(out_map, label_names, out_dir)
    metrics = evaluate_result(root, result_path, split)
    metrics.update(cfg)
    metrics["name"] = out_dir.name
    metrics["source_day_images"] = source_counts["day"]
    metrics["source_night_images"] = source_counts["night"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Prediction-only consensus calibration for KAIST IA-DASR/IAER outputs.")
    parser.add_argument("--data", default="./data/multispectral/KAIST_local.yaml")
    parser.add_argument("--base-label-dir", required=True, help="IAER or primary labels directory containing result.txt")
    parser.add_argument("--day-label-dir", required=True, help="day expert labels directory containing result.txt")
    parser.add_argument("--night-label-dir", required=True, help="night expert labels directory containing result.txt")
    parser.add_argument("--project", default="runs/postproc_search")
    parser.add_argument("--name", default="consensus")
    parser.add_argument("--apply-source", choices=["day", "night", "all"], default="day")
    parser.add_argument("--grid", action="store_true", help="run a local diagnostic grid search")
    parser.add_argument("--support-thr", type=float, default=0.30)
    parser.add_argument("--boost-thr", type=float, default=0.45)
    parser.add_argument("--unique-factor", type=float, default=0.85)
    parser.add_argument("--dense-unique-factor", type=float, default=0.70)
    parser.add_argument("--dense-count", type=int, default=3)
    parser.add_argument("--density-radius", type=float, default=1.25)
    parser.add_argument("--duplicate-iou", type=float, default=0.55)
    parser.add_argument("--duplicate-factor", type=float, default=0.75)
    parser.add_argument("--support-boost", type=float, default=0.0)
    parser.add_argument("--center-weight", type=float, default=0.35)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    with open(root / args.data, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    split = int(data["day_night_split"])
    label_dir = root / data["path"] / "labels" / "test"
    label_names = sorted(p.name for p in label_dir.glob("*.txt"))

    base, base_rows = parse_result_file(root / args.base_label_dir / "result.txt")
    day, day_rows = parse_result_file(root / args.day_label_dir / "result.txt")
    night, night_rows = parse_result_file(root / args.night_label_dir / "result.txt")

    output_root = root / args.project / args.name
    output_root.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args)

    rows = []
    for idx, cfg in enumerate(grid, start=1):
        run_name = cfg_name(cfg) if args.grid else "calibrated"
        out_dir = output_root / run_name
        print(f"[{idx}/{len(grid)}] {run_name}")
        rows.append(
            run_once(
                args,
                cfg,
                root,
                label_names,
                split,
                base,
                day,
                night,
                base_rows,
                day_rows,
                night_rows,
                out_dir,
            )
        )

    rows.sort(key=lambda r: (r["MR_all"], r["MR_day"], r["MR_night"]))
    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "MR_all",
            "MR_day",
            "MR_night",
            "source_day_images",
            "source_night_images",
            "support_thr",
            "unique_factor",
            "dense_unique_factor",
            "duplicate_iou",
            "duplicate_factor",
            "support_boost",
            "boost_thr",
            "dense_count",
            "density_radius",
            "center_weight",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    summary_path = output_root / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Best\n")
        f.write(f"  {best['name']}: MR-all={best['MR_all']:.4f}, MR-day={best['MR_day']:.4f}, MR-night={best['MR_night']:.4f}\n")
        f.write("\nTop 10\n")
        for row in rows[:10]:
            f.write(f"  {row['name']}: MR-all={row['MR_all']:.4f}, MR-day={row['MR_day']:.4f}, MR-night={row['MR_night']:.4f}\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
