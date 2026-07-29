# Round 2I+ evidence snapshot

This directory contains the selected, non-weight evidence copied from the
archived Round 2I+ run:

- `results.csv`: three training/validation rows;
- `opt.yaml`: sanitized run options;
- `hyp.yaml`: hyperparameters;
- `official_reload_metrics.source.json`: path-sanitized official reload summary
  with source and checkpoint hashes;
- `artifacts/`: selected curves and confusion matrix.

The public model card is
[`model_cards/ia_dasr_round2i_plus.md`](../../model_cards/ia_dasr_round2i_plus.md).

## Official reload result

```text
MR-all / MR-day / MR-night = 6.909 / 7.370 / 4.844 %
Recall-all = 98.42 %
```

The inference checkpoint is not committed. Its verified SHA-256 is:

```text
626D279F6C0D547F3559B3483009D9C34F860D1D9A1D970F9950F38BAC03B7C6
```

This is a protocol-optimized single-checkpoint result. It includes a KAIST
daytime ROI filter, night score calibration, and bounded PCSF semantic score
factor. It is not the protocol-neutral formal mainline and is not an LLM/VLM
result.
