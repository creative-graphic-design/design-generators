#!/usr/bin/env python3
"""Compute the CGL label-size package/vendor comparison from score files."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from statistics import stdev
from typing import TypedDict

import yaml
from scipy.stats import ttest_ind


METRIC_ORDER = (
    "R_{shm} (vgg distance)",
    "alignment-LayoutGAN++",
    "occlusion",
    "overlap-LayoutGAN++",
    "overlay",
    "test_coverage_layout",
    "test_density_layout",
    "test_fid_layout",
    "test_precision_layout",
    "test_recall_layout",
    "underlay_effectiveness_loose",
    "underlay_effectiveness_strict",
    "unreadability",
    "utilization",
    "validity",
)


class ComparisonRow(TypedDict):
    """One package/vendor result row."""

    package_values: list[float]
    vendor_values: list[float]
    package_mean: float
    package_sd: float
    vendor_mean: float
    vendor_sd: float
    package_mean_in_vendor_range: bool
    vendor_mean_in_package_range: bool
    welch_p: float | None
    permutation_p: float
    permutation_total: int
    mean_difference: float
    standardized_difference: float


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def _score_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("generated_samples_*/scores_all.yaml"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one scores_all.yaml below {directory}, found {candidates}"
        )
    return candidates[0]


def _read_scores(directory: Path) -> tuple[Path, dict[str, float]]:
    path = _score_file(directory)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    scores = {key: float(value) for key, value in document["average"]["test"].items()}
    if set(scores) != set(METRIC_ORDER):
        raise RuntimeError(f"unexpected metric set in {path}: {sorted(scores)}")
    return path, scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _range_membership(first: list[float], second: list[float]) -> bool:
    mean = sum(first) / len(first)
    return min(second) <= mean <= max(second)


def _standardized_difference(
    mean_difference: float, first_sd: float, second_sd: float
) -> float:
    pooled_sd = math.sqrt((first_sd**2 + second_sd**2) / 2)
    if pooled_sd == 0:
        return 0.0 if mean_difference == 0 else math.inf
    return abs(mean_difference) / pooled_sd


def _permutation_p(values: list[float], first_size: int) -> tuple[float, int]:
    observed = abs(
        sum(values[:first_size]) / first_size
        - sum(values[first_size:]) / (len(values) - first_size)
    )
    failures = 0
    total = 0
    indexes = range(len(values))
    for first_indexes in itertools.combinations(indexes, first_size):
        first_set = set(first_indexes)
        second_indexes = [index for index in indexes if index not in first_set]
        first_mean = sum(values[index] for index in first_indexes) / first_size
        second_mean = sum(values[index] for index in second_indexes) / (
            len(values) - first_size
        )
        failures += abs(first_mean - second_mean) >= observed
        total += 1
    if total != math.comb(len(values), first_size):
        raise RuntimeError(f"expected all relabelings, got {total}")
    return failures / total, total


def _comparison(
    package: dict[int, dict[str, float]],
    vendor: dict[int, dict[str, float]],
) -> dict[str, ComparisonRow]:
    result: dict[str, ComparisonRow] = {}
    package_values = {
        metric: [row[metric] for row in package.values()] for metric in METRIC_ORDER
    }
    vendor_values = {
        metric: [row[metric] for row in vendor.values()] for metric in METRIC_ORDER
    }
    for metric in METRIC_ORDER:
        package_row = package_values[metric]
        vendor_row = vendor_values[metric]
        package_mean = sum(package_row) / len(package_row)
        vendor_mean = sum(vendor_row) / len(vendor_row)
        package_sd = stdev(package_row)
        vendor_sd = stdev(vendor_row)
        permutation_value, permutation_total = _permutation_p(
            package_row + vendor_row, len(package_row)
        )
        welch_p = float(ttest_ind(package_row, vendor_row, equal_var=False).pvalue)
        welch_value = welch_p if math.isfinite(welch_p) else None
        result[metric] = {
            "package_values": package_row,
            "vendor_values": vendor_row,
            "package_mean": package_mean,
            "package_sd": package_sd,
            "vendor_mean": vendor_mean,
            "vendor_sd": vendor_sd,
            "package_mean_in_vendor_range": _range_membership(package_row, vendor_row),
            "vendor_mean_in_package_range": _range_membership(vendor_row, package_row),
            "welch_p": welch_value,
            "permutation_p": permutation_value,
            "permutation_total": permutation_total,
            "mean_difference": package_mean - vendor_mean,
            "standardized_difference": _standardized_difference(
                package_mean - vendor_mean, package_sd, vendor_sd
            ),
        }
    return result


def main() -> None:
    """Write the label-size comparison JSON artifact."""
    root = _repo_root()
    eval_root = root / ".cache/ralf/training-reproduction/cgl/label_size/s5/evals"
    package_dirs = {seed: eval_root / f"package-seed-{seed}" for seed in range(1, 6)}
    vendor_dirs = {seed: eval_root / f"vendor-seed-{seed}" for seed in range(1, 6)}
    loaded: dict[str, dict[int, dict[str, float]]] = {"package": {}, "vendor": {}}
    artifacts: dict[str, dict[int, dict[str, str]]] = {"package": {}, "vendor": {}}
    for name, directories in (("package", package_dirs), ("vendor", vendor_dirs)):
        for seed, directory in directories.items():
            path, scores = _read_scores(directory)
            loaded[name][seed] = scores
            artifacts[name][seed] = {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
            }

    metrics = _comparison(loaded["package"], loaded["vendor"])
    largest_metric = max(
        metrics, key=lambda metric: metrics[metric]["standardized_difference"]
    )
    document = {
        "artifacts": artifacts,
        "comparisons": metrics,
        "metric_order": METRIC_ORDER,
        "method": {
            "package_n": 5,
            "vendor_n": 5,
            "range_test": "inclusive package and vendor mean membership in the other group's min-max range",
            "welch_test": "two-sided scipy.stats.ttest_ind with equal_var=False",
            "permutation_test": "two-sided absolute package-vendor mean difference over every C(10,5) relabeling",
        },
        "summary": {
            "both_direction_range_pass": sum(
                row["package_mean_in_vendor_range"]
                and row["vendor_mean_in_package_range"]
                for row in metrics.values()
            ),
            "welch_p_below_0_05": sum(
                row["welch_p"] is not None and row["welch_p"] < 0.05
                for row in metrics.values()
            ),
            "permutation_p_below_0_05": sum(
                row["permutation_p"] < 0.05 for row in metrics.values()
            ),
            "largest_standardized_difference": {
                "metric": largest_metric,
                "value": metrics[largest_metric]["standardized_difference"],
            },
        },
    }
    output = (
        root
        / ".cache/ralf/training-reproduction/cgl/label_size/s5/comparison/comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    print(json.dumps(document["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
