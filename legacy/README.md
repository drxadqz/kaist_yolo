# Legacy prototype

The first public version of this repository used a numbered 01–06 pipeline:

```text
01 data preparation
02 YOLO11n visible/LWIR baselines
03 box-level late fusion
04 SORT/ByteTrack experiments
05 small mid-fusion score refiner
06 mid-fusion tracking scaffold
```

Those directories were removed from the current release tree because they
contained duplicated training artifacts and 33 ordinary Git checkpoint blobs
totalling about 291 MiB. They remain recoverable from Git history at commit
`f4a03bb`.

Verified legacy detector results:

| Modality | Best mAP@.5 | Best mAP@.5:.95 |
|---|---:|---:|
| Visible | 60.139% | 26.357% |
| LWIR | 71.010% | 33.467% |

The best values occur at different epochs and are retained only as historical
baselines.

Not promoted as verified improvements:

- late fusion had implementation code but no committed evaluation output;
- the mid-fusion refiner had no checkpoint, training log, or formal metric;
- the mid-fusion tracker had no committed output.

The legacy tracking diagnostic is not a valid full MOT benchmark because its GT
contains only 93 non-empty frames and uses a unique identity for every box.
Identity metrics are therefore invalid, and empty-frame false positives are
excluded by the old evaluator.

The current IA-DASR structure replaces this artifact-oriented pipeline with
portable configs, canonical results, model cards, tests, and explicit evidence
boundaries.
