# KAIST dataset contract

## Source and license

The KAIST Multispectral Pedestrian Dataset must be obtained from the
[official benchmark repository](https://github.com/SoonminHwang/rgbt-ped-detection).
The dataset is not bundled here. The official repository identifies the dataset
license as CC BY-NC-SA 4.0 and its toolbox code as BSD-2-Clause; users must review
and accept those terms themselves.

## Strict local layout

The default release configuration expects:

```text
datasets/KAIST_local_strict/
├── visible/
│   ├── train/
│   └── test/
├── infrared/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
├── labels_ignore/
│   ├── train/
│   └── test/
├── annotations_raw/
│   ├── train/
│   └── test/
└── manifests/
```

Visible and thermal filenames must remain paired and ordered. The data adapter
uses directory links/junctions for images and regenerates YOLO label files from
the supplied annotations.

## Prepare the view

All source paths are explicit CLI arguments; the repository contains no
machine-specific defaults:

```bash
python src/setup_local_kaist.py \
  --image-source-root /path/to/organized/kaist_yolo \
  --train-annotation-root /path/to/sanitized/train/annotations \
  --test-annotation-root /path/to/official/test/annotations \
  --target-root datasets/KAIST_local_strict
```

Expected image source layout:

```text
<image-source-root>/
└── images/
    ├── visible/
    │   ├── train/
    │   └── val/
    └── lwir/
        ├── train/
        └── val/
```

Use `--skip-order-check` only for debugging. Formal evaluation requires the
2,252-image KAIST test ordering.

## Protocol manifest

The strict evaluation snapshot used by this project contains:

- 2,252 paired test images;
- 3,390 positive `person` instances;
- 864 ignore regions;
- day/night split index 1,455.

The evaluator annotation JSON in
[`src/evaluation_script/KAIST_annotation.json`](../src/evaluation_script/KAIST_annotation.json)
is byte-identical to the file published in the DeformCAT repository (Git blob
`b1622092a340f59305379f87b5b1c2b13e066f16`). It is retained for evaluator
compatibility, treated as KAIST dataset material under CC BY-NC-SA 4.0, and
attributed in [`ATTRIBUTION.md`](ATTRIBUTION.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Data hygiene

Do not commit:

- source images or copied dataset archives;
- generated labels or cache files;
- train/validation split text files containing local paths;
- prediction dumps;
- annotation sources not explicitly licensed for redistribution.

The root `.gitignore` excludes the complete `datasets/` tree.

## Annotation interpretation

Only `person` is a positive detection class for this release. Ignored labels and
non-countable regions are written separately so `--kaist-ignore-aware-obj` can
mask them during negative objectness supervision.

Changing image order, label sanitization, ignored-region handling, day/night
split, or ROI filtering changes the evaluation protocol. Such a result must use
a new experiment ID rather than silently replacing an existing result card.
