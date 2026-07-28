# Contributing

Contributions that improve portability, evaluation integrity, documentation, or
RGB-T modeling are welcome.

1. Create a focused branch.
2. Keep datasets, checkpoints, predictions, and credentials outside Git.
3. Add or update a result card for every metric-changing contribution.
4. State whether a result is a single model, protocol-optimized model, routed
   system, or post-processing result.
5. Run:

   ```bash
   python scripts/validate_release.py
   python -m unittest discover -s tests -v
   ```

6. Describe the exact protocol, seed, checkpoint hash, and evidence source in
   the pull request.

Please preserve the AGPL-3.0 license and upstream attribution.
