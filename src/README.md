# Source layout

This directory contains the IA-DASR research implementation:

```text
train.py                 training entrypoint
test.py                  inference and KAIST evaluation entrypoint
setup_local_kaist.py     portable strict-dataset adapter
models/                  YOLO-style graph and RGB-T fusion modules
utils/                   data, loss, metric, plotting, and runtime utilities
evaluation_script/       KAIST/COCO-style miss-rate evaluator
```

The code descends from DeformCAT and YOLOv5 and retains a script-oriented
research layout for checkpoint compatibility. See
[`docs/ATTRIBUTION.md`](../docs/ATTRIBUTION.md) before reusing individual files.

Formal project additions are documented in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). Model and dataset paths are
supplied through repository-relative configs or explicit CLI arguments; no
private machine path is required.
