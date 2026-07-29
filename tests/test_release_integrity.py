from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_release.py"
SPEC = importlib.util.spec_from_file_location("validate_release", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT}")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class ReleaseIntegrityTests(unittest.TestCase):
    def test_canonical_scope_counts(self) -> None:
        rows = validator.load_rows()
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["setting"]] = counts.get(row["setting"], 0) + 1
        self.assertEqual(
            counts,
            {
                "single_model": 11,
                "protocol_optimized": 1,
                "system": 1,
                "postprocess": 1,
            },
        )

    def test_headline_improvement_is_reproducible(self) -> None:
        rows = {row["record_id"]: row for row in validator.load_rows()}
        early = float(rows["KAIST_B2_EARLY6"]["mr_all_pct"])
        formal = float(rows["KAIST_M5_FORMAL"]["mr_all_pct"])
        relative = 100 * (early - formal) / early
        self.assertAlmostEqual(relative, 47.27, places=2)

    def test_release_validator_has_no_findings(self) -> None:
        self.assertEqual(validator.run_validation(), [])


if __name__ == "__main__":
    unittest.main()
