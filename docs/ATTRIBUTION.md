# Attribution and third-party components

## License

This repository is distributed under AGPL-3.0-only. It contains derivative
YOLOv5/DeformCAT source, and both current upstream repositories identify their
code as AGPL-3.0. The exact historical imported revisions are not recoverable,
so this release uses the conservative AGPL license instead of claiming a more
permissive relicensing right. The root `LICENSE` controls repository code except
where a file or asset states a separate license.

## DeformCAT

- Repository: <https://github.com/jiongger/DeformCAT>
- License: AGPL-3.0
- Role: dual-stream multispectral detector, deformable cross-modal interaction,
  model graph, training/evaluation scaffold, and KAIST evaluator integration.
- Locally modified areas include `src/models/common.py`,
  `src/models/yolo_test.py`, `src/train.py`, `src/test.py`, and transformer model
  configs.

The exact imported upstream commit is not recoverable from the original local
workspace metadata. This is recorded as a provenance limitation rather than
inventing a revision.

## YOLOv5

- Repository: <https://github.com/ultralytics/yolov5>
- Current upstream license: AGPL-3.0
- Role: YOLO-style backbone/head components, training utilities, data loading,
  loss, plotting, checkpoint, and NMS infrastructure.

The released files retain their upstream-style comments. Original project
contributions are enumerated in [`ARCHITECTURE.md`](ARCHITECTURE.md) and are not
described as authorship of YOLOv5.

## COCO API

- Repository: <https://github.com/cocodataset/cocoapi>
- License: Simplified BSD / BSD-2-Clause
- Files: `src/evaluation_script/coco.py` and
  `src/evaluation_script/cocoeval.py`.

The full copyright notice is preserved in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## KAIST benchmark

- Repository: <https://github.com/SoonminHwang/rgbt-ped-detection>
- Dataset license identified by the official repository: CC BY-NC-SA 4.0
- Toolbox code license identified by the official repository: BSD-2-Clause
- Citation:

  > Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon.
  > “Multispectral Pedestrian Detection: Benchmark Dataset and Baselines.”
  > CVPR, 2015.

The repository does not redistribute the full dataset. The evaluator annotation
JSON retained for metric compatibility and the selected qualitative montage
containing adapted KAIST frames are separately marked CC BY-NC-SA 4.0 in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and
[`assets/README.md`](../assets/README.md).

## Project contributions

The project owner's work includes:

- three-scale lightweight DARP-Net integration;
- consensus/detail and detail-scene reliability modules;
- ignore-aware and protocol-semantic supervision;
- KAIST-specific bounded calibration experiments;
- expert routing and post-processing experiments;
- cross-dataset diagnostics and fine-tuning;
- experiment cards, canonical result schema, model cards, release validation,
  and evidence-grounded LLM reporting.

This attribution file is a best-effort software provenance record, not a
substitute for legal advice.
