#!/usr/bin/env python3
"""Recompute the CGL unconditional three-seed comparison from score files."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any  # noqa: TID251 - YAML score payloads are heterogeneous.

import yaml
from scipy.stats import t as student_t


METRICS = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(value: Any) -> Fraction:
    return Fraction(str(value))


def decimal(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 24
        result = Decimal(value.numerator) / Decimal(value.denominator)
        text = format(result, "f").rstrip("0").rstrip(".")
    return text or "0"


def score_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("scores_all.yaml"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one score file under {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def averages(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    scores = payload["average"]["test"]
    missing = [metric for metric in METRICS if metric not in scores]
    if missing:
        raise KeyError(f"missing metrics in {path}: {missing}")
    return {metric: scores[metric] for metric in METRICS}


def welch_p(package: list[Fraction], vendor: list[Fraction]) -> float:
    package_values = [float(value) for value in package]
    vendor_values = [float(value) for value in vendor]
    package_mean = sum(package_values) / len(package_values)
    vendor_mean = sum(vendor_values) / len(vendor_values)
    package_variance = sum((value - package_mean) ** 2 for value in package_values) / (
        len(package_values) - 1
    )
    vendor_variance = sum((value - vendor_mean) ** 2 for value in vendor_values) / (
        len(vendor_values) - 1
    )
    standard_error = math.sqrt(
        package_variance / len(package_values) + vendor_variance / len(vendor_values)
    )
    if standard_error == 0:
        return 1.0 if package_mean == vendor_mean else 0.0
    statistic = (package_mean - vendor_mean) / standard_error
    numerator = (
        package_variance / len(package_values) + vendor_variance / len(vendor_values)
    ) ** 2
    denominator = package_variance**2 / (
        len(package_values) ** 2 * (len(package_values) - 1)
    ) + vendor_variance**2 / (len(vendor_values) ** 2 * (len(vendor_values) - 1))
    degrees_of_freedom = numerator / denominator if denominator else 1.0
    return float(2 * student_t.sf(abs(statistic), degrees_of_freedom))


def permutation_count(
    package: list[Fraction], vendor: list[Fraction]
) -> tuple[int, int]:
    pooled = package + vendor
    observed = abs(
        sum(package, Fraction()) / len(package) - sum(vendor, Fraction()) / len(vendor)
    )
    count = 0
    for package_indices in itertools.combinations(range(len(pooled)), len(package)):
        package_index_set = set(package_indices)
        candidate_package = [
            value for index, value in enumerate(pooled) if index in package_index_set
        ]
        candidate_vendor = [
            value
            for index, value in enumerate(pooled)
            if index not in package_index_set
        ]
        difference = abs(
            sum(candidate_package, Fraction()) / len(candidate_package)
            - sum(candidate_vendor, Fraction()) / len(candidate_vendor)
        )
        count += difference >= observed
    return count, math.comb(len(pooled), len(package))


def metric_result(
    package_values: list[Any], vendor_values: list[Any]
) -> dict[str, Any]:
    package = [fraction(value) for value in package_values]
    vendor = [fraction(value) for value in vendor_values]
    package_mean = sum(package, Fraction()) / len(package)
    vendor_mean = sum(vendor, Fraction()) / len(vendor)
    package_float = [float(value) for value in package]
    vendor_float = [float(value) for value in vendor]
    package_sd = math.sqrt(
        sum((value - float(package_mean)) ** 2 for value in package_float)
        / (len(package) - 1)
    )
    vendor_sd = math.sqrt(
        sum((value - float(vendor_mean)) ** 2 for value in vendor_float)
        / (len(vendor) - 1)
    )
    permutation_k, permutation_n = permutation_count(package, vendor)
    pooled_sd = math.sqrt((package_sd**2 + vendor_sd**2) / 2)
    standardized_difference = (
        (float(package_mean) - float(vendor_mean)) / pooled_sd if pooled_sd else 0.0
    )
    return {
        "package_values": package_values,
        "vendor_values": vendor_values,
        "package_mean": decimal(package_mean),
        "package_sd": package_sd,
        "vendor_mean": decimal(vendor_mean),
        "vendor_sd": vendor_sd,
        "package_mean_in_vendor_range": min(vendor) <= package_mean <= max(vendor),
        "vendor_mean_in_package_range": min(package) <= vendor_mean <= max(package),
        "welch_p": welch_p(package, vendor),
        "permutation_k": permutation_k,
        "permutation_n": permutation_n,
        "permutation_fraction": f"{permutation_k}/{permutation_n}",
        "standardized_difference": standardized_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    systems: dict[str, dict[str, Any]] = {}
    for system, root in (("package", args.package_root), ("vendor", args.vendor_root)):
        seeds: dict[str, Any] = {}
        for seed in range(1, 4):
            path = score_file(root / f"seed-{seed}")
            seeds[str(seed)] = {
                "path": str(path),
                "sha256": sha256(path),
                "metrics": averages(path),
            }
        systems[system] = seeds

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        package_values = [
            systems["package"][str(seed)]["metrics"][metric] for seed in range(1, 4)
        ]
        vendor_values = [
            systems["vendor"][str(seed)]["metrics"][metric] for seed in range(1, 4)
        ]
        metrics[metric] = metric_result(package_values, vendor_values)

    summary = {
        "both_direction_range_passes": sum(
            result["package_mean_in_vendor_range"]
            and result["vendor_mean_in_package_range"]
            for result in metrics.values()
        ),
        "welch_p_below_0.05": sum(
            result["welch_p"] < 0.05 for result in metrics.values()
        ),
        "permutation_p_below_0.05": sum(
            result["permutation_k"] / result["permutation_n"] < 0.05
            for result in metrics.values()
        ),
        "permutation_relabelings": math.comb(6, 3),
    }
    result = {
        "seed_scope": "training-seed n=3",
        "systems": systems,
        "metrics": metrics,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}))


if __name__ == "__main__":
    main()
