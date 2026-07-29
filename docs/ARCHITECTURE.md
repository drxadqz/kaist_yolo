# Architecture and method scope

## Problem

RGB-T pedestrian detection is not ordinary channel fusion. Visible and thermal
cameras can be locally misaligned, their reliability changes with illumination,
and each sensor may contain modality-specific clutter. KAIST evaluation also
distinguishes countable people from ignored regions. A detector that simply
concatenates the two images can therefore learn confident but protocol-invalid
responses.

IA-DASR addresses three coupled failure modes:

1. **local geometric mismatch** between the visible and thermal streams;
2. **modality conflict and scene-dependent reliability**;
3. **incorrect negative supervision** inside ignore/non-countable regions.

## Data flow

Let paired inputs be \(I^v\) and \(I^t\). Two YOLOv5s-style backbones produce
visible and thermal features at three detection levels:

\[
F_l^v = B_l^v(I^v), \qquad
F_l^t = B_l^t(I^t), \qquad l \in \{3,4,5\}.
\]

Each level is processed by the same conceptual stack:

```text
RGB feature ─┐
             ├─ DCAF ─ CDR ─ DSRE ─ fused feature ─ detection pyramid
LWIR feature ┘

KAIST ignore masks ───────── ignore-aware objectness supervision
```

The implementation lives primarily in
[`src/models/common.py`](../src/models/common.py), with graph construction in
[`src/models/yolo_test.py`](../src/models/yolo_test.py) and experiment switches
in [`src/train.py`](../src/train.py) and [`src/test.py`](../src/test.py).

## DCAF: deformable cross-modal alignment and fusion

For a visible query at position \(p\), the thermal stream is sampled at learned
local offsets:

\[
\widetilde F_k^t(p) = F^t(p + \Delta p_k(p)),
\qquad
\Delta p_k(p) =
\tanh\left(\phi_k([F^v(p),F^t(p)])\right)s .
\]

The bounded \(\tanh\) and scale \(s\) keep sampling local. Cross-modal attention
then aggregates sampled thermal values:

\[
a_k^v(p) =
\operatorname{softmax}_k
\left(
\frac{q^v(p)^\top k_k^t(p)}{\sqrt d}
+ b(\Delta p_k(p))
\right),
\]

\[
G^v(p) = F^v(p) + \sum_k a_k^v(p)W_v\widetilde F_k^t(p).
\]

The thermal-to-visible direction is symmetric. The two updated streams are
projected to one scale feature. This differs from global attention: offsets and
keys are constrained to a local sampling field, reducing cost and making the
operator explicitly target small RGB-T registration errors.

## CDR: consensus-detail refinement

Alignment alone does not decide whether the two sensors agree. CDR creates
projected visible, thermal, and fused representations:

\[
R=\psi_v(F^v), \quad T=\psi_t(F^t), \quad Z=\psi_f(F^f).
\]

It exposes agreement and conflict cues:

\[
A=R\odot T,\qquad D=|R-T|,
\qquad \delta Z=Z-\operatorname{AvgPool}_{3\times3}(Z).
\]

`A` captures cross-modal consensus, `D` captures disagreement, and
\(\delta Z\) retains local detail that global or low-frequency fusion may erase.

## DSRE: detail-scene reliability estimation

DSRE predicts bounded maps for foreground support, background conflict,
ambiguity, groupness, uncertainty, day context, and modality preference. Two
representative maps are:

\[
M_{fg}=\sigma(\phi_{fg}([Z,A,\delta Z])),
\]

\[
M_{conf}=
\sigma(\phi_{conf}([D,\delta Z,Z]))(1-M_{fg}).
\]

The output uses residual modulation and lower clamps. Reliability estimates can
reshape a feature but cannot freely erase the backbone representation:

\[
\widehat F^f =
F^f + \gamma\,
\phi_o([Z^\*, Z^\*M_{fg}, F_{detail}]).
\]

This conservative design is important because a reliability branch is itself
uncertain, especially at night or under heavy occlusion.

## Ignore-aware supervision

KAIST contains ignored and non-countable regions. Treating those regions as
background produces contradictory objectness targets. With
`--kaist-ignore-aware-obj`, anchors that overlap ignore regions are masked out
of negative objectness supervision while valid positive assignments are
preserved.

This is a training-time correction, not a post-hoc metric trick. The exact
dataset and evaluation protocol still matter; see
[`DATASET.md`](DATASET.md).

## Formal mainline

The formal public method is:

```text
DCAF + CDR + DSRE + ignore-aware objectness
```

Its source alias is
[`configs/models/ia_dasr_formal.yaml`](../configs/models/ia_dasr_formal.yaml).
Its canonical same-protocol re-evaluation is:

```text
MR-all / MR-day / MR-night = 7.137 / 7.917 / 4.159 %
```

This is the safest single-model claim because its setting is explicit and its
source is in the release.

## Protocol-aware Round 2I+ branch

Round 2I+ retains the normal box, objectness, and class outputs, then adds a
protocol-semantic branch for humanness \(H\), countability \(C\), groupness
\(G\), uncertainty \(U\), and bounded protocol bias \(B\):

\[
q = c + B - \alpha_G G - \alpha_U U,
\qquad C=\sigma(q), \qquad S_{sem}=H C.
\]

At inference, the semantic score is allowed to apply only a weak bounded factor:

\[
r = \operatorname{clip}
\left(
1+\alpha\tanh(\beta(S_{sem}-\mu)),
r_{\min},r_{\max}
\right),
\qquad o' = or.
\]

The archived runtime used:

```text
alpha=0.05, r_min=0.95, r_max=1.06, mu=0.25, beta=4.0
```

Round 2I+ also uses a KAIST daytime ROI filter and night score calibration.
Therefore its `6.909%` official reload MR-all is reported as a
**protocol-optimized single checkpoint**, not as the formal generic mainline.

## System extensions

IAER routes between expert predictions using estimated illumination. ECR
calibrates and reranks already-generated predictions. These are useful system
experiments, but neither changes the formal IA-DASR architecture claim:

```text
formal single model         IA-DASR                         7.137% MR-all
protocol-optimized model    IA-DASR Round 2I+               6.909% MR-all
routed system               IA-DASR + IAER                  6.900% MR-all
post-processing system      IA-DASR + IAER + ECR            6.841% MR-all
```

## Original contribution boundary

This repository does not claim authorship of YOLOv5, DeformCAT, COCO API, or
KAIST. Original project work centers on:

- integrating three-scale deformable RGB-T interaction into a lightweight
  YOLOv5s-style dual stream;
- consensus/detail and scene-reliability refinement;
- ignore-aware objectness and protocol-semantic supervision;
- bounded inference-time semantic calibration experiments;
- illumination-aware expert routing and prediction calibration;
- the result-provenance, model-card, and reproducibility workflow.

See [`ATTRIBUTION.md`](ATTRIBUTION.md) for upstream code ownership.

## Complexity and trade-offs

The same-protocol timing snapshot reports:

| Model | Total latency (ms/image) | Approx. throughput |
|---|---:|---:|
| Early Fusion 6-channel | 19.04 | 52.5 FPS |
| DCAF | 44.53 | 22.5 FPS |
| DCAF + CDR | 85.57 | 11.7 FPS |
| IA-DASR formal | 59.52 | 16.8 FPS |

These numbers are environment-specific and are not a hardware-neutral benchmark.
They make the core trade-off visible: IA-DASR improves low-FPPI miss rate but is
roughly 3.1× slower than the early-fusion baseline in the recorded snapshot.
