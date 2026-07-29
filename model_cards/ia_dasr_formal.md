# DARP-Fusion protocol-neutral predecessor model card

## Summary

| Field | Value |
|---|---|
| Task | paired RGB-thermal pedestrian detection |
| Dataset | KAIST Multispectral Pedestrian Dataset |
| Protocol | Reasonable |
| Input | paired 640-pixel RGB/LWIR images |
| Architecture | dual YOLOv5s-style streams + DCAF + CDR + DSRE |
| Training | ignore-aware objectness, seed 0 research run |
| Setting | formal single model |
| License | AGPL-3.0-only |

## Intended use

Research on multispectral pedestrian detection, cross-modal alignment, modality
reliability, and protocol-aware evaluation.

Not intended as a safety-certified production detector or as evidence of an
LLM/VLM system.

## Metrics

Canonical repository re-evaluation:

| Metric | Value |
|---|---:|
| MR-all | 7.137% |
| MR-day | 7.917% |
| MR-night | 4.159% |
| mAP@.5 | 71.741% |
| mAP@.5:.95 | 35.376% |
| Recall-all | 98.351% |
| Recorded total latency | 59.52 ms/image |

The historical result card rounds the same project line to
`7.13 / 7.91 / 4.17%`. Public automated outputs use the full-precision canonical
CSV so rounding remains stable.

## Training configuration

- model: `configs/models/ia_dasr_formal.yaml`
- data: `configs/datasets/kaist.example.yaml`
- hyperparameters: `configs/experiments/hyp.formal_stage2.yaml`
- input size: 640
- epochs: 10
- batch size: 1
- initialization: stage-1 `kaist_dasr_iao_strict_v2` best checkpoint
- required switches:
  - `--kaist-day-roi-filter`
  - `--kaist-ignore-aware-obj`
  - `--drr-aux-weight 0.03`

## Limitations

- Primary evidence is from KAIST.
- The runtime snapshot is hardware/environment-specific.
- Training budgets across all intermediate ablations are not perfectly
  normalized.
- No public checkpoint URL is claimed yet.
- The exact stage-1 initialization checkpoint/hash is not yet published, so
  from-scratch reproducibility of the validated continuation remains incomplete.
- This detector has no language encoder and is not an LLM/VLM.

## Ethical and safety considerations

Pedestrian detection can affect privacy and safety. Evaluate demographic,
weather, camera, geographic, and deployment-domain shifts before use. Do not
treat benchmark recall as a safety guarantee.
