# Résumé and interview guide

## One-line positioning

Use:

> DARP-Net is a reproducible Transformer-based RGB-T detector that combines
> deformable cross-modal alignment, reliability modeling, and protocol-aware
> learning for KAIST pedestrian detection.

Do not use:

> An LLM-based pedestrian detector.

## Chinese résumé bullets

### Recommended compact version

> 设计双流 RGB–热红外检测模型 DARP-Net，在 P3/P4/P5 三尺度实现多头双向
> 可变形 Transformer cross-attention，并结合可靠性建模、ignore-aware 监督与
> 协议因子化检测头；KAIST Reasonable 官方复评 MR-all/day/night 为
> 6.909/7.370/4.844%，Recall-all 98.42%。

> 建立单模态、早期融合、DCAF/CDR/DSRE 消融、协议感知单 checkpoint、专家路由和
> 后处理系统的分层评测矩阵；以 result card、checkpoint SHA-256、机器可读
> CSV、CI 和 claim boundary 保证实验可复现、结论可审计。

> 以 canonical claim table、result card、checkpoint SHA-256 和 CI 维护指标
> 来源与适用边界；另实现可选的结构化实验报告工具，但该工具不参与模型训练
> 或推理，也不作为检测增益来源。

### Best-result version

> 归档 DARP-Net 锁定单 checkpoint（SHA-256 可追溯）：官方 reload/fused
> 复评 MR-all/day/night 为 6.909/7.370/4.844%，相比 6 通道 Early Fusion
> 的 MR-all 13.535 低 49.0%；明确标注 KAIST ROI、夜间校准与有界 BPSC 的
> 协议适用边界。

## English résumé bullets

> Designed DARP-Net, a dual-stream RGB-thermal detector with three-scale,
> multi-head bidirectional deformable Transformer cross-attention, reliability
> modeling, and ignore-aware supervision; achieved 6.909/7.370/4.844% MR
> all/day/night and 98.42% recall on the KAIST Reasonable evaluation.

> Built an auditable experiment stack spanning single-modality baselines,
> fusion ablations, protocol-aware calibration, expert routing, and
> post-processing; tracked result cards, checkpoint hashes, canonical metrics,
> CI tests, and explicit claim boundaries.

> Maintained canonical claim tables, result cards, checkpoint hashes, and CI
> checks for auditable evaluation; added an optional structured report utility
> as downstream tooling, separate from detector training and inference.

## Interview topics

Be ready to explain:

1. Why local deformable attention is preferable to global attention for small
   sensor misalignment.
2. Why RGB and LWIR reliability changes between day and night.
3. Why mAP@.5 can decline while low-FPPI MR improves.
4. How ignore-aware objectness changes targets rather than merely filtering
   final detections.
5. Why DARP-Net's KAIST-specific calibration must be separated from portable variants.
6. Why routed experts and prediction reranking cannot be called one model.
7. What the LLM copilot does—and, equally important, what it does not do.
8. Why FLIR zero-shot results limit generalization claims.

## Mapping to large-model roles

Credible transferable themes:

- multimodal token/feature alignment;
- cross-attention and conditional computation;
- reliability and uncertainty gating;
- weak/bounded adapters that preserve a pretrained backbone;
- evaluation provenance and safe model reporting;
- structured outputs and LLM tool integration.

Missing from this detector:

- tokenizer or language encoder;
- instruction tuning or preference optimization;
- retrieval or agent loops;
- language-conditioned detection;
- vision-language contrastive pretraining.

Use [SP-WFM](https://github.com/drxadqz/SP-WFM) as the primary direct LLM
post-training project and DARP-Net as the multimodal perception/research-
engineering project. Together they form a stronger and more honest narrative
than relabeling DARP-Net as an LLM.
