# Checkpoint policy

Model weights are not stored as ordinary Git blobs. This keeps clones small,
avoids duplicate optimizer states, and makes every published checkpoint an
explicit release artifact.

## Planned artifacts

| File | Purpose | SHA-256 |
|---|---|---|
| `ia_dasr_stage1_best.pt` | initialization for the validated formal stage-2 continuation | not yet published in this release |
| `ia_dasr_stage2_e10_best.pt` | formal single-model mainline | not yet published in this release |
| `ia_dasr_round2i_plus_best.pt` | protocol-optimized inference checkpoint | `626D279F6C0D547F3559B3483009D9C34F860D1D9A1D970F9950F38BAC03B7C6` |

The archived Round 2I+ inference file is approximately 65.6 MB. Raw training
checkpoints containing optimizer state are intentionally excluded.

## Verification

Windows PowerShell:

```powershell
Get-FileHash .\checkpoints\ia_dasr_round2i_plus_best.pt -Algorithm SHA256
```

Linux/macOS:

```bash
sha256sum checkpoints/ia_dasr_round2i_plus_best.pt
```

Only load PyTorch checkpoint files from a trusted source. Traditional `.pt`
files may use Python pickle and can execute code while loading. A future release
should prefer a state-dict-only or `safetensors` artifact where the model graph
permits it.

No download URL is advertised until the corresponding GitHub Release exists and
its checksum has been verified.
