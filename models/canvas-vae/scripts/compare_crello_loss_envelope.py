"""Compare production trainer loss traces on the verified training stream."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, median
from typing import TypedDict, cast

MIN_PRODUCTION_UPDATES = 1500


class _TraceUpdate(TypedDict):
    record: str
    update: int
    objective: float
    batch_sha256: str


class _SeedSummary(TypedDict):
    original_median: float
    package_median: float
    aligned_logger_events: int


class _LossEnvelopeSummary(TypedDict):
    updates_per_run: int
    logger_interval_updates: int
    logger_smoothing: str
    matched_training_seeds: list[int]
    batch_sequence_mismatches: int
    original_seed_medians: list[float]
    original_range: list[float]
    original_width: float
    expanded_original_range: list[float]
    package_seed_medians: list[float]
    package_summary: float
    per_seed: dict[str, _SeedSummary]
    passed: bool


def _read_trace(
    path: Path, *, system: str, seed: int, updates: int
) -> list[_TraceUpdate]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not isinstance(rows[0], dict):
        raise ValueError(f"{path}: trace metadata is missing")

    metadata = rows[0]
    if (
        metadata.get("record") != "metadata"
        or metadata.get("system") != system
        or metadata.get("seed") != seed
    ):
        raise ValueError(f"{path}: trace metadata does not match {system} seed {seed}")

    if len(rows) != updates + 1:
        raise ValueError(
            f"{path}: expected {updates} update records, got {len(rows) - 1}"
        )

    trace: list[_TraceUpdate] = []
    for expected_update, row in enumerate(rows[1:], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: update {expected_update} is not an object")

        update = cast(_TraceUpdate, row)
        if (
            update.get("record") != "update"
            or type(update.get("update")) is not int
            or update["update"] != expected_update
            or type(update.get("objective")) not in {int, float}
            or not math.isfinite(update["objective"])
            or not isinstance(update.get("batch_sha256"), str)
            or len(update["batch_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in update["batch_sha256"]
            )
        ):
            raise ValueError(f"{path}: invalid update record {expected_update}")

        trace.append(update)

    return trace


def compare_loss_envelope(
    original_dir: Path,
    package_dir: Path,
    *,
    seeds: Sequence[int],
    updates: int,
    logger_interval: int,
) -> _LossEnvelopeSummary:
    """Apply the protocol gate to logger-aligned per-update traces."""
    if (
        len(seeds) != 3
        or len(set(seeds)) != len(seeds)
        or updates < MIN_PRODUCTION_UPDATES
        or logger_interval < 1
        or updates % logger_interval
    ):
        raise ValueError(
            f"updates must be at least {MIN_PRODUCTION_UPDATES} and a multiple of logger_interval"
        )

    per_seed: dict[str, _SeedSummary] = {}
    sequence_mismatches = 0
    original_medians: list[float] = []
    package_medians: list[float] = []
    for seed in seeds:
        original = _read_trace(
            original_dir / f"seed-{seed}" / "updates.jsonl",
            system="original",
            seed=seed,
            updates=updates,
        )
        package = _read_trace(
            package_dir / f"seed-{seed}" / "updates.jsonl",
            system="package",
            seed=seed,
            updates=updates,
        )
        sequence_mismatches += sum(
            original_update["batch_sha256"] != package_update["batch_sha256"]
            for original_update, package_update in zip(original, package, strict=True)
        )

        original_events = [
            fmean(
                update["objective"]
                for update in original[start : start + logger_interval]
            )
            for start in range(0, updates, logger_interval)
        ]
        package_events = [
            fmean(
                update["objective"]
                for update in package[start : start + logger_interval]
            )
            for start in range(0, updates, logger_interval)
        ]
        original_median = float(median(original_events))
        package_median = float(median(package_events))
        original_medians.append(original_median)
        package_medians.append(package_median)
        per_seed[str(seed)] = {
            "original_median": original_median,
            "package_median": package_median,
            "aligned_logger_events": len(original_events),
        }

    original_min = min(original_medians)
    original_max = max(original_medians)
    original_width = original_max - original_min
    lower_bound = original_min - original_width
    upper_bound = original_max + original_width
    package_summary = float(median(package_medians))
    return {
        "updates_per_run": updates,
        "logger_interval_updates": logger_interval,
        "logger_smoothing": "arithmetic mean over each original epoch's fixed-size batches",
        "matched_training_seeds": list(seeds),
        "batch_sequence_mismatches": sequence_mismatches,
        "original_seed_medians": original_medians,
        "original_range": [original_min, original_max],
        "original_width": original_width,
        "expanded_original_range": [lower_bound, upper_bound],
        "package_seed_medians": package_medians,
        "package_summary": package_summary,
        "per_seed": per_seed,
        "passed": sequence_mismatches == 0
        and lower_bound <= package_summary <= upper_bound,
    }


def main() -> None:
    """Read two trace directories and write their loss-envelope result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--updates", type=int, default=1501)
    parser.add_argument("--logger-interval", type=int, required=True)
    args = parser.parse_args()
    result = compare_loss_envelope(
        args.original_dir,
        args.package_dir,
        seeds=args.seeds,
        updates=args.updates,
        logger_interval=args.logger_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
