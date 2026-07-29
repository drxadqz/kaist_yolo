#!/usr/bin/env python3
"""Validate the lightweight public release without importing ML dependencies."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "results" / "benchmark_summary.csv"
ALLOWED_SETTINGS = {
    "single_model",
    "protocol_optimized",
    "system",
    "postprocess",
}
ALLOWED_EVIDENCE = {
    "reproduced",
    "official_reload",
    "validated_summary",
}
MACHINE_READABLE_SOURCE_IDS = {
    "KAIST_M5_FORMAL",
    "KAIST_R2I_PLUS",
    "KAIST_IAER_ROUTER",
    "KAIST_IAER_ECR",
}
REQUIRED_COLUMNS = {
    "record_id",
    "dataset",
    "protocol",
    "setting",
    "model",
    "input_size",
    "seed",
    "mr_all_pct",
    "mr_day_pct",
    "mr_night_pct",
    "evidence_level",
    "source",
    "claim_boundary",
}
REQUIRED_PATHS = (
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "configs/datasets/kaist.example.yaml",
    "configs/models/ia_dasr_formal.yaml",
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARKS.md",
    "docs/LLM_COPILOT.md",
    "model_cards/ia_dasr_formal.md",
    "model_cards/ia_dasr_round2i_plus.md",
    "results/benchmark_summary.csv",
    "results/postprocess/ecr_bestall_metrics.source.csv",
    "results/protocol_aware_best/official_reload_metrics.source.json",
    "results/system/iaer_router_metrics.source.json",
    "scripts/experiment_copilot.py",
)
NUMERIC_COLUMNS = (
    "mr_all_pct",
    "mr_day_pct",
    "mr_night_pct",
    "map50_pct",
    "map50_95_pct",
    "recall_all_pct",
)
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/])")
PRIVATE_UNIX = re.compile(r"/home/(?:shen|fqy|dell)(?:/|\\b)", re.IGNORECASE)


class ValidationError(RuntimeError):
    pass


def git_tracked_files() -> set[str] | None:
    if not (REPO_ROOT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }


def load_rows(path: Path = BENCHMARK) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValidationError(f"benchmark columns missing: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValidationError("benchmark has no rows")
    return rows


def validate_benchmark(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    tracked = git_tracked_files()
    for line, row in enumerate(rows, start=2):
        record_id = row["record_id"].strip()
        if not record_id:
            errors.append(f"CSV line {line}: empty record_id")
        elif record_id in ids:
            errors.append(f"CSV line {line}: duplicate record_id {record_id}")
        ids.add(record_id)

        if row["setting"] not in ALLOWED_SETTINGS:
            errors.append(f"CSV line {line}: unsupported setting {row['setting']!r}")
        if row["evidence_level"] not in ALLOWED_EVIDENCE:
            errors.append(
                f"CSV line {line}: unsupported evidence_level "
                f"{row['evidence_level']!r}"
            )
        if not row["claim_boundary"].strip():
            errors.append(f"CSV line {line}: empty claim_boundary")

        for column in NUMERIC_COLUMNS:
            raw = row.get(column, "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"CSV line {line}: {column} is not numeric")
                continue
            if not 0 <= value <= 100:
                errors.append(f"CSV line {line}: {column} outside [0, 100]")

        source_relative = row["source"].strip()
        source = REPO_ROOT / source_relative
        if not source.is_file():
            errors.append(
                f"CSV line {line}: evidence source does not exist: {source_relative}"
            )
        elif tracked is not None and source_relative not in tracked:
            errors.append(
                f"CSV line {line}: evidence source is not tracked: {source_relative}"
            )
        if (
            record_id in MACHINE_READABLE_SOURCE_IDS
            and source.suffix.lower() not in {".csv", ".json"}
        ):
            errors.append(
                f"CSV line {line}: key result source is not machine-readable: "
                f"{source_relative}"
            )

    expected = {
        "KAIST_M5_FORMAL",
        "KAIST_R2I_PLUS",
        "KAIST_IAER_ROUTER",
        "KAIST_IAER_ECR",
    }
    missing_ids = sorted(expected - ids)
    if missing_ids:
        errors.append("benchmark missing key records: " + ", ".join(missing_ids))

    by_id = {row["record_id"]: row for row in rows}
    if {"KAIST_B2_EARLY6", "KAIST_M5_FORMAL"} <= by_id.keys():
        early = float(by_id["KAIST_B2_EARLY6"]["mr_all_pct"])
        formal = float(by_id["KAIST_M5_FORMAL"]["mr_all_pct"])
        relative = 100 * (early - formal) / early
        if abs(relative - 47.27) > 0.02:
            errors.append(f"unexpected formal-vs-early relative gain: {relative:.3f}%")

    return errors


def iter_public_files() -> list[Path]:
    ignored_roots = {
        ".git",
        "artifacts",
        "datasets",
        "outputs",
        "runs",
        "wandb",
    }
    ignored_anywhere = {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "venv",
    }
    return [
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(REPO_ROOT).parts[0] not in ignored_roots
        and not (
            set(path.relative_to(REPO_ROOT).parts)
            & ignored_anywhere
        )
    ]


def validate_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    tracked = git_tracked_files()
    for relative in REQUIRED_PATHS:
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"required file missing: {relative}")
        elif tracked is not None and relative not in tracked:
            errors.append(f"required file is not tracked by Git: {relative}")

    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"}:
            errors.append(f"weight/model binary committed: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB release limit: {relative}")

        if suffix not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitattributes",
            ".gitignore",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"text file is not UTF-8: {relative}")
            continue

        if WINDOWS_ABSOLUTE.search(text):
            errors.append(f"machine-specific Windows path in {relative}")
        if PRIVATE_UNIX.search(text):
            errors.append(f"machine-specific Unix path in {relative}")
        placeholder_pattern = r"\b(?:" + "your" + r"-name|T" + "BD" + r")\b"
        if re.search(placeholder_pattern, text, flags=re.IGNORECASE):
            errors.append(f"unresolved publishing placeholder in {relative}")

    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    if 'license: "AGPL-3.0-only"' not in citation:
        errors.append("CITATION.cff license is not AGPL-3.0-only")
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_text:
        errors.append("root LICENSE is not AGPL")

    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("auths/", "*.db", ".env", "*.pt", "datasets/", "runs/"):
        if rule not in ignore_text:
            errors.append(f".gitignore missing safety rule: {rule}")
    return errors


def validate_markdown_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(REPO_ROOT.resolve())
            except ValueError:
                errors.append(
                    f"{path.relative_to(REPO_ROOT)}: link leaves repository: {target}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(REPO_ROOT)}: broken local link: {target}"
                )
    return errors


def run_validation() -> list[str]:
    errors: list[str] = []
    try:
        rows = load_rows()
    except (OSError, ValidationError) as exc:
        errors.append(str(exc))
        rows = []
    if rows:
        errors.extend(validate_benchmark(rows))
    files = iter_public_files()
    errors.extend(validate_files(files))
    errors.extend(validate_markdown_links(files))
    return sorted(set(errors))


def main() -> int:
    errors = run_validation()
    if errors:
        print("Release validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    rows = load_rows()
    print(
        f"Release validation passed: {len(rows)} benchmark rows, "
        f"{len(iter_public_files())} public files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
