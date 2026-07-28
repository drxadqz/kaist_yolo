import argparse
import shutil
from pathlib import Path

import yaml

from evaluation_script.evaluation_script import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Route KAIST day/night predictions from two expert runs.')
    parser.add_argument('--data', type=str, default='./data/multispectral/KAIST_local.yaml',
                        help='dataset yaml used to recover day/night split and label order')
    parser.add_argument('--day-label-dir', type=str, required=True,
                        help='directory that contains result.txt and per-image txt outputs for the day expert')
    parser.add_argument('--night-label-dir', type=str, required=True,
                        help='directory that contains result.txt and per-image txt outputs for the night expert')
    parser.add_argument('--day-night-split', type=int, default=None,
                        help='override the day/night split index; defaults to yaml value')
    parser.add_argument('--project', type=str, default='runs/test',
                        help='project directory for merged outputs')
    parser.add_argument('--name', type=str, default='kaist_dual_expert_route_eval',
                        help='run name for merged outputs')
    parser.add_argument('--copy-labels', action='store_true',
                        help='copy the chosen per-image txt files into the merged label directory')
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


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    with open(root / args.data, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    split = args.day_night_split
    if split is None:
        split = int(data['day_night_split'])

    test_label_dir = root / data['path'] / 'labels' / 'test'
    label_names = sorted(p.name for p in test_label_dir.glob('*.txt'))
    total_images = len(label_names)

    day_label_dir = Path(args.day_label_dir)
    night_label_dir = Path(args.night_label_dir)
    day_result = load_result_map(day_label_dir / 'result.txt')
    night_result = load_result_map(night_label_dir / 'result.txt')

    save_dir = root / args.project / args.name
    labels_out = save_dir / 'labels'
    labels_out.mkdir(parents=True, exist_ok=True)

    merged_result = labels_out / 'result.txt'
    with open(merged_result, 'w', encoding='utf-8') as f:
        for index in range(1, total_images + 1):
            source = day_result if index <= split else night_result
            for line in source.get(index, []):
                f.write(line + '\n')

    if args.copy_labels:
        for idx, label_name in enumerate(label_names, start=1):
            source_dir = day_label_dir if idx <= split else night_label_dir
            source_file = source_dir / label_name
            target_file = labels_out / label_name
            if source_file.exists():
                shutil.copy2(source_file, target_file)

    ann_path = root / 'evaluation_script' / 'KAIST_annotation.json'
    result = evaluate(str(ann_path), str(merged_result), split)
    mr_all = float(result['all'].summarize(0))
    mr_day = float(result['day'].summarize(0))
    mr_night = float(result['night'].summarize(0))

    summary_path = save_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'day_label_dir: {day_label_dir}\n')
        f.write(f'night_label_dir: {night_label_dir}\n')
        f.write(f'day_night_split: {split}\n')
        f.write(f'MR-all: {mr_all:.4f}\n')
        f.write(f'MR-day: {mr_day:.4f}\n')
        f.write(f'MR-night: {mr_night:.4f}\n')

    print(f'Merged labels saved to {labels_out}')
    print(f'MR-all: {mr_all:.4f}')
    print(f'MR-day: {mr_day:.4f}')
    print(f'MR-night: {mr_night:.4f}')


if __name__ == '__main__':
    main()
