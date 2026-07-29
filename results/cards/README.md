# Experiment cards

Each card binds one public result to a model setting, dataset protocol, input
size, seed, metrics, and evidence source.

The current public cards are:

- B0: RGB-only baseline;
- B1: LWIR-only baseline;
- B2: 6-channel early-fusion baseline;
- M4: DCAF + CDR + DSRE ablation;
- M5: formal IA-DASR single model;
- IAER: routed system.

The protocol-optimized Round 2I+ checkpoint has a full model card in
[`model_cards/ia_dasr_round2i_plus.md`](../../model_cards/ia_dasr_round2i_plus.md).

Cards are supporting records. When a rounded value conflicts with the canonical
CSV, [`results/benchmark_summary.csv`](../benchmark_summary.csv) controls
automated public output.
