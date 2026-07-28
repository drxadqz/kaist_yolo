import argparse
import csv
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

from evaluation_script.evaluation_script import evaluate
from models.experimental import attempt_load
from test import _clear_drr_aux_maps, _collect_drr_aux_maps, _set_drr_aux_export
from utils.datasets import create_dataloader_rgb_ir
from utils.general import check_dataset, check_img_size, colorstr
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Learned illumination-aware routing for KAIST dual experts.')
    parser.add_argument('--data', type=str, default='./data/multispectral/KAIST_local.yaml',
                        help='dataset yaml used to recover validation order')
    parser.add_argument('--day-label-dir', type=str, required=True,
                        help='directory that contains result.txt and per-image txt outputs for the day expert')
    parser.add_argument('--night-label-dir', type=str, required=True,
                        help='directory that contains result.txt and per-image txt outputs for the night expert')
    parser.add_argument('--router-weights', type=str, default='',
                        help='weights used to export learned day_gate scores when route source needs a model')
    parser.add_argument('--route-source', type=str, default='day_gate',
                        choices=['day_gate', 'visible_mean', 'fused'],
                        help='illumination score source used for routing')
    parser.add_argument('--threshold-mode', type=str, default='split_accuracy',
                        choices=['split_accuracy', 'otsu', 'fixed'],
                        help='how to determine the route threshold')
    parser.add_argument('--threshold', type=float, default=None,
                        help='manual threshold used when --threshold-mode fixed')
    parser.add_argument('--fused-visible-weight', type=float, default=0.35,
                        help='visible_mean contribution when route-source is fused')
    parser.add_argument('--project', type=str, default='runs/test',
                        help='project directory for merged outputs')
    parser.add_argument('--name', type=str, default='kaist_illumination_route_eval',
                        help='run name for merged outputs')
    parser.add_argument('--copy-labels', action='store_true',
                        help='copy chosen per-image txt files into the merged label directory')
    parser.add_argument('--device', type=str, default='0', help='cuda device used by the router model')
    parser.add_argument('--img-size', type=int, default=640, help='router input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='router dataloader batch size')
    parser.add_argument('--workers', type=int, default=0, help='router dataloader workers')
    parser.add_argument('--day-night-split', type=int, default=None,
                        help='override the day/night split index; defaults to yaml value')
    return parser.parse_args()


def load_result_map(result_path):
    mapping = {}
    with open(result_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            index = int(line.split(',', 1)[0])
            mapping.setdefault(index, []).append(line)
    return mapping


def otsu_threshold(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.5
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return lo
    hist, bin_edges = np.histogram(values, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum().clip(min=1.0)
    omega = np.cumsum(prob)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.clip(omega * (1.0 - omega), 1e-12, None)
    best = int(np.argmax(sigma_b))
    return float(centers[best])


def best_accuracy_threshold(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.size == 0:
        return 0.5, False, 0.0

    uniq = np.unique(values)
    if uniq.size == 1:
        preds = values >= uniq[0]
        acc_direct = float((preds.astype(np.int64) == labels).mean())
        acc_inverse = float(((~preds).astype(np.int64) == labels).mean())
        invert = acc_inverse > acc_direct
        return float(uniq[0]), invert, max(acc_direct, acc_inverse)

    candidates = []
    candidates.append(float(uniq[0] - 1e-6))
    candidates.extend(float((uniq[i] + uniq[i + 1]) * 0.5) for i in range(len(uniq) - 1))
    candidates.append(float(uniq[-1] + 1e-6))

    best_thr, best_invert, best_acc = candidates[0], False, -1.0
    for thr in candidates:
        pred_day = (values >= thr)
        acc_direct = float((pred_day.astype(np.int64) == labels).mean())
        if acc_direct > best_acc:
            best_thr, best_invert, best_acc = thr, False, acc_direct
        acc_inverse = float(((~pred_day).astype(np.int64) == labels).mean())
        if acc_inverse > best_acc:
            best_thr, best_invert, best_acc = thr, True, acc_inverse
    return float(best_thr), bool(best_invert), float(best_acc)


def prepare_router_dataset(root, data_dict, imgsz, batch_size, stride, workers):
    opt = SimpleNamespace(single_cls=False, kaist_ignore_aware_obj=False)
    dataloader, dataset = create_dataloader_rgb_ir(
        data_dict['val_rgb'],
        data_dict['val_ir'],
        imgsz,
        batch_size,
        stride,
        opt,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
        workers=workers,
        prefix=colorstr('route: ')
    )
    return dataloader, dataset


@torch.no_grad()
def collect_route_scores(args, root, data_dict):
    route_source = args.route_source
    use_day_gate = route_source in {'day_gate', 'fused'}

    if use_day_gate and not args.router_weights:
        raise ValueError('--router-weights is required when route-source uses day_gate')

    device = select_device(args.device, batch_size=args.batch_size)
    model = None
    stride = 32
    if use_day_gate:
        model = attempt_load(str(root / args.router_weights), map_location=device)
        stride = max(int(model.stride.max()), 32)
        imgsz = check_img_size(args.img_size, s=stride)
        model.eval()
        _set_drr_aux_export(model, True)
    else:
        imgsz = args.img_size

    check_dataset(data_dict)
    dataloader, dataset = prepare_router_dataset(root, data_dict, imgsz, args.batch_size, stride, args.workers)

    rows = []
    pbar = tqdm(dataloader, total=len(dataloader), desc='Collect route scores')
    for batch in pbar:
        if len(batch) == 5:
            imgs, _, _, paths, _ = batch
        else:
            imgs, _, paths, _ = batch
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        imgs_rgb = imgs[:, :3, :, :]
        imgs_ir = imgs[:, 3:, :, :]

        if model is not None:
            _ = model(imgs_rgb, imgs_ir)
            aux_maps = _collect_drr_aux_maps(model)
        else:
            aux_maps = []

        day_gate_scores = None
        if aux_maps:
            gate_list = []
            for aux in aux_maps:
                day_gate = aux.get('day_gate')
                if day_gate is not None:
                    gate_list.append(day_gate.float().view(day_gate.shape[0], -1).mean(dim=1))
            if gate_list:
                day_gate_scores = torch.stack(gate_list, dim=0).mean(dim=0)
        visible_mean_scores = imgs_rgb.float().mean(dim=(1, 2, 3))
        _clear_drr_aux_maps(model) if model is not None else None

        for idx, path in enumerate(paths):
            day_gate_score = float(day_gate_scores[idx].item()) if day_gate_scores is not None else math.nan
            visible_score = float(visible_mean_scores[idx].item())
            if route_source == 'day_gate':
                score = day_gate_score
            elif route_source == 'visible_mean':
                score = visible_score
            elif route_source == 'fused':
                if math.isnan(day_gate_score):
                    score = visible_score
                else:
                    score = (1.0 - args.fused_visible_weight) * day_gate_score + args.fused_visible_weight * visible_score
            else:
                raise NotImplementedError(route_source)

            rows.append({
                'stem': Path(path).stem,
                'score': float(score),
                'day_gate_score': float(day_gate_score),
                'visible_mean_score': float(visible_score),
            })

    return rows


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    with open(root / args.data, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)

    split = int(args.day_night_split if args.day_night_split is not None else data_dict['day_night_split'])
    test_label_dir = root / data_dict['path'] / 'labels' / 'test'
    label_names = sorted(p.name for p in test_label_dir.glob('*.txt'))
    total_images = len(label_names)
    gt_day_labels = np.array([1 if idx <= split else 0 for idx in range(1, total_images + 1)], dtype=np.int64)

    rows = collect_route_scores(args, root, data_dict)
    if len(rows) != total_images:
        raise RuntimeError(f'route row count mismatch: got {len(rows)}, expected {total_images}')

    scores = np.array([row['score'] for row in rows], dtype=np.float64)
    if np.isnan(scores).any():
        raise RuntimeError('NaN route scores found; check router weights or route source')

    if args.threshold_mode == 'fixed':
        if args.threshold is None:
            raise ValueError('--threshold is required when --threshold-mode fixed')
        threshold = float(args.threshold)
        direct_pred = scores >= threshold
        acc_direct = float((direct_pred.astype(np.int64) == gt_day_labels).mean())
        acc_inverse = float(((~direct_pred).astype(np.int64) == gt_day_labels).mean())
        invert = acc_inverse > acc_direct
        route_acc = max(acc_direct, acc_inverse)
    elif args.threshold_mode == 'otsu':
        threshold = otsu_threshold(scores)
        direct_pred = scores >= threshold
        acc_direct = float((direct_pred.astype(np.int64) == gt_day_labels).mean())
        acc_inverse = float(((~direct_pred).astype(np.int64) == gt_day_labels).mean())
        invert = acc_inverse > acc_direct
        route_acc = max(acc_direct, acc_inverse)
    elif args.threshold_mode == 'split_accuracy':
        threshold, invert, route_acc = best_accuracy_threshold(scores, gt_day_labels)
    else:
        raise NotImplementedError(args.threshold_mode)

    pred_day = scores >= threshold
    if invert:
        pred_day = ~pred_day

    day_label_dir = root / args.day_label_dir
    night_label_dir = root / args.night_label_dir
    day_result = load_result_map(day_label_dir / 'result.txt')
    night_result = load_result_map(night_label_dir / 'result.txt')

    save_dir = root / args.project / args.name
    labels_out = save_dir / 'labels'
    labels_out.mkdir(parents=True, exist_ok=True)

    merged_result = labels_out / 'result.txt'
    with open(merged_result, 'w', encoding='utf-8') as f:
        for index in range(1, total_images + 1):
            source = day_result if pred_day[index - 1] else night_result
            for line in source.get(index, []):
                f.write(line + '\n')

    if args.copy_labels:
        for idx, row in enumerate(rows, start=1):
            source_dir = day_label_dir if pred_day[idx - 1] else night_label_dir
            source_file = source_dir / f"{row['stem']}.txt"
            target_file = labels_out / f"{row['stem']}.txt"
            if source_file.exists():
                shutil.copy2(source_file, target_file)

    ann_path = root / 'evaluation_script' / 'KAIST_annotation.json'
    result = evaluate(str(ann_path), str(merged_result), split)
    mr_all = float(result['all'].summarize(0))
    mr_day = float(result['day'].summarize(0))
    mr_night = float(result['night'].summarize(0))

    score_csv = save_dir / 'route_scores.csv'
    with open(score_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['index', 'stem', 'score', 'day_gate_score', 'visible_mean_score', 'gt_day', 'pred_day']
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({
                'index': idx,
                'stem': row['stem'],
                'score': f"{row['score']:.8f}",
                'day_gate_score': f"{row['day_gate_score']:.8f}",
                'visible_mean_score': f"{row['visible_mean_score']:.8f}",
                'gt_day': int(gt_day_labels[idx - 1]),
                'pred_day': int(pred_day[idx - 1]),
            })

    summary_path = save_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'route_source: {args.route_source}\n')
        f.write(f'threshold_mode: {args.threshold_mode}\n')
        f.write(f'threshold: {threshold:.8f}\n')
        f.write(f'invert: {invert}\n')
        f.write(f'route_accuracy: {route_acc:.6f}\n')
        f.write(f'day_label_dir: {day_label_dir}\n')
        f.write(f'night_label_dir: {night_label_dir}\n')
        f.write(f'MR-all: {mr_all:.4f}\n')
        f.write(f'MR-day: {mr_day:.4f}\n')
        f.write(f'MR-night: {mr_night:.4f}\n')

    print(f'Merged labels saved to {labels_out}')
    print(f'route_source={args.route_source}')
    print(f'threshold={threshold:.6f} invert={invert} route_acc={route_acc:.4f}')
    print(f'MR-all: {mr_all:.4f}')
    print(f'MR-day: {mr_day:.4f}')
    print(f'MR-night: {mr_night:.4f}')


if __name__ == '__main__':
    main()
