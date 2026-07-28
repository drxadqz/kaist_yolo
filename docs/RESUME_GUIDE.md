# Résumé and interview guide

## One-line positioning

Use:

> IA-DASR is a reproducible RGB-T multimodal detector that combines deformable
> cross-modal alignment, scene-reliability learning, and ignore-aware
> supervision for KAIST pedestrian detection.

Do not use:

> An LLM-based pedestrian detector.

## Chinese résumé bullets

### Recommended compact version

> 设计轻量双流 RGB–热红外检测模型 IA-DASR，在 P3/P4/P5 三尺度引入双向
> 可变形跨模态注意力、场景可靠性调制与 ignore-aware 监督；在 KAIST
> Reasonable 同协议重评中取得 7.14% MR-all，相较 6 通道 Early Fusion
> baseline 相对下降 47.3%。

> 建立单模态、早期融合、DCAF/CDR 消融、协议优化单 checkpoint、专家路由和
> 后处理系统的分层评测矩阵；以 result card、checkpoint SHA-256、机器可读
> CSV、CI 和 claim boundary 保证实验可复现、结论可审计。

> 实现 evidence-grounded LLM 实验助手：基于 OpenAI Responses API
> structured outputs 将验证指标生成报告/简历草稿，并通过 scope guard
> 阻止把 protocol/system/postprocess 结果误写成单模型结论。

### Best-result version

> 归档 KAIST 协议优化单 checkpoint：官方 reload/fused 复评 MR-all/day/night
> 为 6.909/7.370/4.844%，Recall-all 98.42%；明确隔离 KAIST ROI、夜间校准与
> 有界 PCSF 的协议特化影响，避免跨数据集或通用性过度声明。

## English résumé bullets

> Designed IA-DASR, a dual-stream RGB-thermal detector with three-scale
> bidirectional deformable cross-modal attention, scene-reliability modulation,
> and ignore-aware supervision; achieved 7.14% MR on KAIST Reasonable, a 47.3%
> relative reduction over a same-protocol 6-channel early-fusion baseline.

> Built an auditable experiment stack spanning single-modality baselines,
> fusion ablations, protocol-aware calibration, expert routing, and
> post-processing; tracked result cards, checkpoint hashes, canonical metrics,
> CI tests, and explicit claim boundaries.

> Implemented an evidence-grounded experiment copilot with the OpenAI Responses
> API and structured outputs, converting verified CSV metrics into reports and
> résumé drafts while preventing system-level results from becoming
> single-model claims.

## Interview topics

Be ready to explain:

1. Why local deformable attention is preferable to global attention for small
   sensor misalignment.
2. Why RGB and LWIR reliability changes between day and night.
3. Why mAP@.5 can decline while low-FPPI MR improves.
4. How ignore-aware objectness changes targets rather than merely filtering
   final detections.
5. Why Round 2I+ is a protocol-optimized result and not the generic mainline.
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
post-training project and IA-DASR as the multimodal perception/research-
engineering project. Together they form a stronger and more honest narrative
than relabeling IA-DASR as an LLM.
