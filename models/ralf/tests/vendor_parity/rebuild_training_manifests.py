#!/usr/bin/env python3
"""Rebuild retrospective RALF final-stage manifests from campaign records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, cast  # noqa: TID251 - record payloads are heterogeneous.

import yaml


CHECKPOINT_RULE = "final checkpoint named in the condition's launch manifest"
EVALUATOR_COMMAND = "original inference plus `eval.py`; package checkpoints enter through the evaluator adapter"
SOURCE_COMMIT_RE = re.compile(
    r"[\"']?source_commit[\"']?\s*[:=]\s*[\"']?([0-9a-f]{7,40})",
    re.IGNORECASE,
)
GIT_HEAD_RE = re.compile(
    r"[\"']?git_head[\"']?\s*[:=]\s*[\"']?([0-9a-f]{7,40})",
    re.IGNORECASE,
)
COMMIT_FALLBACK_RE = re.compile(
    r"[\"']?commit[\"']?\s*[:=]\s*[\"']?([0-9a-f]{7,40})",
    re.IGNORECASE,
)
SOURCE_JSON_RE = re.compile(r'"source_commit"\s*:\s*"([0-9a-f]{7,40})"')
LAUNCH_RE = re.compile(
    r"(?:launched_at|launched_utc|launch_utc|launch_started_utc|launch_time_utc|started_utc|prelaunch_utc)\s*[:=]\s*(?:[\"']?)([^\s\"']+)"
)
RUN_RE = re.compile(
    r"^(?:package|vendor)(?:-[a-z0-9]+)*-seed-\d+(?:-run-\d+)?$|^seed-\d+(?:-run-\d+)?$"
)
STAGE_RECORD_SUFFIXES = {".csv", ".json", ".log", ".record", ".txt", ".yaml", ".yml"}


def iter_files(root: Path) -> Iterable[Path]:
    """Walk a campaign cache without entering evaluator or cache symlinks."""

    for directory, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [
            name
            for name in directories
            if name not in {".git", "eval-vendor-source"}
            and not (Path(directory) / name).is_symlink()
        ]
        for name in filenames:
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_label(path: Path, campaign_root: Path) -> str:
    """Return a stable label without exposing a campaign worktree path."""

    parts = path.parts
    marker = "training-reproduction"
    if marker in parts:
        relative = Path(*parts[parts.index(marker) + 1 :])
    else:
        relative = Path(path.name)
    return ".cache/ralf/training-reproduction/" + "/".join(relative.parts)


def sanitize(value: Any) -> Any:
    """Remove absolute paths from resolved configuration values."""

    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return "$RALF_CACHE_DIR" if value.count("/") == 3 else "$ABSOLUTE_PATH"
        return value
    return value


def record_files(campaign_root: Path) -> list[Path]:
    names = {
        "evaluation-ledger.txt",
        "launch-manifest.txt",
        "prelaunch.v2.record",
        "prelaunch.record",
        "result.record",
        "source-gate.log",
        "conversion-source-gate.log",
        "train.log",
    }
    return sorted(
        path
        for path in iter_files(campaign_root)
        if path.is_file() and path.name in names
    )


def config_files(campaign_root: Path) -> list[Path]:
    return sorted(
        path
        for path in iter_files(campaign_root)
        if path.is_file() and path.name == "config.yaml"
    )


def stage_record_files(campaign_root: Path) -> list[Path]:
    """Return machine-readable pre-final-stage records used for provenance."""

    return sorted(
        path
        for path in iter_files(campaign_root)
        if path.is_file()
        and any(stage in path.parts for stage in ("s0", "s1", "s2", "s3", "s4"))
        and "comparison" not in path.parts
        and path.suffix.lower() in STAGE_RECORD_SUFFIXES
    )


def run_name(path: Path, campaign_root: Path) -> str | None:
    try:
        relative_parts = path.relative_to(campaign_root).parts
    except ValueError:
        return None
    for part in reversed(relative_parts):
        if RUN_RE.fullmatch(part):
            return part
    return None


def text_values(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = [
        {"commit": commit, "field": "source_commit"}
        for commit in SOURCE_COMMIT_RE.findall(text)
    ]
    if not source:
        source = [
            {"commit": commit, "field": "git_head"}
            for commit in GIT_HEAD_RE.findall(text)
        ]
    if not source:
        source = [
            {"commit": commit, "field": "commit"}
            for commit in COMMIT_FALLBACK_RE.findall(text)
        ]
    timestamps = LAUNCH_RE.findall(text)
    unique_source = list(
        {(item["commit"], item["field"]): item for item in source}.values()
    )
    return unique_source, list(dict.fromkeys(timestamps))


def _unique_dicts(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    result = []
    for value in values:
        key = tuple(sorted(value.items()))
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def run_records(files: Iterable[Path], campaign_root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in files:
        name = run_name(path, campaign_root)
        if name is None:
            continue
        entry = grouped.setdefault(
            name,
            {"run": name, "source_commits": [], "launched_at": [], "records": []},
        )
        commits, timestamps = text_values(path)
        entry["records"].append(
            {"path": relative_label(path, campaign_root), "sha256": sha256(path)}
        )
        entry["source_commits"].extend(
            {
                "commit": item["commit"],
                "record_path": relative_label(path, campaign_root),
                "field": item["field"],
            }
            for item in commits
        )
        entry["launched_at"].extend(timestamps)
    result = []
    for entry in grouped.values():
        entry["source_commits"] = _unique_dicts(entry["source_commits"])
        entry["launched_at"] = sorted(set(entry["launched_at"]))
        entry["records"] = sorted(entry["records"], key=lambda item: item["path"])
        result.append(entry)
    return sorted(result, key=lambda item: item["run"])


def stage_records(files: Iterable[Path], campaign_root: Path) -> list[dict[str, Any]]:
    result = []
    for path in files:
        commits, timestamps = text_values(path)
        result.append(
            {
                "path": relative_label(path, campaign_root),
                "sha256": sha256(path),
                "source_commits": [
                    {
                        **item,
                        "record_path": relative_label(path, campaign_root),
                    }
                    for item in commits
                ],
                "launched_at": timestamps,
            }
        )
    return result


def checkpoint_records(
    campaign_root: Path, runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_name = {entry["run"]: entry for entry in runs}
    for path in iter_files(campaign_root):
        if not path.is_file() or path.suffix not in {".ckpt", ".pt"}:
            continue
        name = run_name(path, campaign_root)
        if name in by_name:
            by_name[name].setdefault("checkpoints", []).append(
                {"path": relative_label(path, campaign_root), "sha256": sha256(path)}
            )
    for entry in runs:
        entry["checkpoints"] = sorted(
            entry.get("checkpoints", []), key=lambda item: item["path"]
        )
    return runs


def commit_provenance(
    commit: str, head: str, *, dataset: str, condition: str
) -> dict[str, Any]:
    resolved = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not resolved:
        return {"commit": commit, "resolved": False, "ancestor_of_head": False}
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolved, head], check=False
        ).returncode
        == 0
    )
    result: dict[str, Any] = {
        "commit": commit,
        "resolved": True,
        "resolved_commit": resolved,
        "ancestor_of_head": ancestor,
    }
    if not ancestor:
        diff = subprocess.run(
            ["git", "diff", "--stat", resolved, head, "--", "models/ralf/src"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["diff_stat_models_ralf_src_to_head"] = diff.splitlines()
        result["training_evaluation_path_argument"] = (
            f"The {dataset} {condition} run is attributed to its own recorded historical "
            "source commit. Its training checkpoint and the condition-specific vendor "
            "inference plus eval.py path are the evidence path; current package-source "
            "changes are not retroactively attributed to that run."
        )
    return result


def resolved_configs(paths: list[Path], campaign_root: Path) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for path in paths:
        content_hash = sha256(path)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result.append(
            {
                "path": relative_label(path, campaign_root),
                "sha256": content_hash,
                "values": sanitize(payload),
            }
        )
    return result


def build_manifest(
    *,
    dataset: str,
    condition: str,
    campaign_root: Path,
    output: Path,
    head: str,
    comparison_path: Path | None,
    parity_path: Path,
) -> None:
    s5_root = campaign_root / "s5"
    if not s5_root.is_dir():
        raise FileNotFoundError(f"missing S5 campaign directory: {s5_root}")
    records = record_files(s5_root)
    stage_files = stage_record_files(campaign_root)
    configs = config_files(s5_root)
    run_data = checkpoint_records(s5_root, run_records(records, s5_root))
    stage_data = stage_records(stage_files, campaign_root)
    all_record_artifacts = [
        {
            "path": relative_label(path, campaign_root),
            "sha256": sha256(path),
            "kind": "record",
        }
        for path in records
    ]
    comparison_artifact = None
    if comparison_path is not None:
        if not comparison_path.is_file():
            raise FileNotFoundError(comparison_path)
        comparison_artifact = {
            "path": relative_label(comparison_path, campaign_root),
            "sha256": sha256(comparison_path),
        }
    parity_artifact = None
    if parity_path.is_file():
        parity_artifact = {
            "path": relative_label(parity_path, campaign_root),
            "sha256": sha256(parity_path),
        }
    campaign_records = []
    campaign_record_paths = {
        path
        for path in (
            campaign_root / "launch-manifest.txt",
            s5_root / "launch-manifest.txt",
        )
        if path.is_file()
    }
    for path in sorted(campaign_record_paths):
        if not path.is_file():
            continue
        commits_for_path, timestamps = text_values(path)
        campaign_records.append(
            {
                "path": relative_label(path, campaign_root),
                "sha256": sha256(path),
                "source_commits": [
                    {
                        **item,
                        "record_path": relative_label(path, campaign_root),
                    }
                    for item in commits_for_path
                ],
                "launched_at": timestamps,
            }
        )
    commits = {
        provenance["commit"]
        for entry in run_data
        for provenance in entry["source_commits"]
    }
    commits.update(
        item["commit"]
        for record in campaign_records
        for item in cast(list[dict[str, str]], record["source_commits"])
    )
    condition_entry: dict[str, Any] = {
        "condition": condition,
        "campaign_root": "read-only campaign worktree; path withheld",
        "runs": run_data,
        "campaign_records": campaign_records,
        "stage_records": stage_data,
        "source_commits": [
            commit_provenance(commit, head, dataset=dataset, condition=condition)
            for commit in sorted(commits)
        ],
        "source_commit": sorted(commits),
        "launched_at": sorted(
            {timestamp for entry in run_data for timestamp in entry["launched_at"]}
            | {timestamp for entry in stage_data for timestamp in entry["launched_at"]}
            | {
                timestamp
                for entry in campaign_records
                for timestamp in entry["launched_at"]
            }
        ),
        "resolved_config": resolved_configs(configs, campaign_root),
        "checkpoint_rule": CHECKPOINT_RULE,
        "evaluator_command": EVALUATOR_COMMAND,
        "records": all_record_artifacts,
        "comparison": comparison_artifact,
        "evaluation_path_parity": parity_artifact,
    }
    artifacts: list[dict[str, str]] = list(all_record_artifacts)
    artifacts.extend(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "kind": "campaign_record",
        }
        for item in cast(list[dict[str, str]], campaign_records)
    )
    artifacts.extend(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "kind": "stage_record",
        }
        for item in cast(list[dict[str, str]], stage_data)
    )
    artifacts.extend(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "kind": "resolved_config",
        }
        for item in cast(list[dict[str, str]], condition_entry["resolved_config"])
    )
    artifacts.extend(
        {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "kind": "checkpoint",
        }
        for run in run_data
        for checkpoint in cast(list[dict[str, str]], run["checkpoints"])
    )
    for artifact, kind in (
        (comparison_artifact, "comparison"),
        (parity_artifact, "evaluation_path_parity"),
    ):
        if artifact is not None:
            artifacts.append({**artifact, "kind": kind})
    condition_entry["artifacts"] = {item["path"]: item["sha256"] for item in artifacts}
    condition_entry["artifact_records"] = artifacts
    manifest: dict[str, Any] = {}
    if output.is_file():
        loaded_manifest = json.loads(output.read_text(encoding="utf-8"))
        if (
            isinstance(loaded_manifest, dict)
            and loaded_manifest.get("manifest_version") == 2
            and isinstance(loaded_manifest.get("conditions"), dict)
        ):
            manifest = loaded_manifest
    if not manifest:
        manifest = {
            "manifest_type": "launch-manifest",
            "manifest_version": 2,
            "status": "reconstructed after the runs from each condition's own launch, source-gate, training, and evaluation records",
            "reconstructed_after_runs": True,
            "grouping": "One manifest groups all package and vendor full runs for the dataset; each condition entry retains its own run records, source commits, timestamps, resolved configurations, checkpoint hashes, comparison hash, and parity hash.",
            "dataset": dataset,
            "checkpoint_rule": CHECKPOINT_RULE,
            "evaluator_command": EVALUATOR_COMMAND,
            "conditions": {},
        }
    conditions = cast(dict[str, Any], manifest["conditions"])
    conditions[condition] = condition_entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_spec(value: str) -> tuple[str, str, Path, Path | None]:
    key, root, comparison = value.split("|", 2)
    dataset, condition = key.split("/", 1)
    if condition == "label_size":
        condition = "label-size"
    return dataset, condition, Path(root), Path(comparison) if comparison else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--parity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec", action="append", required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace the two dataset manifests before adding condition entries",
    )
    args = parser.parse_args()
    if args.replace:
        outputs = {
            args.output_root / dataset / "s5" / "manifest.json"
            for dataset, _condition, _root, _comparison in map(parse_spec, args.spec)
        }
        for output in outputs:
            output.unlink(missing_ok=True)
    for value in args.spec:
        dataset, condition, campaign_root, comparison_path = parse_spec(value)
        parity_path = (
            args.parity_root / dataset / condition / "evaluation-path-parity.json"
        )
        output = args.output_root / dataset / "s5" / "manifest.json"
        build_manifest(
            dataset=dataset,
            condition=condition,
            campaign_root=campaign_root.resolve(),
            output=output,
            head=args.head,
            comparison_path=comparison_path.resolve() if comparison_path else None,
            parity_path=parity_path.resolve(),
        )


if __name__ == "__main__":
    main()
