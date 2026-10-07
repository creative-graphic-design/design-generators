#!/usr/bin/env python3
"""Audit recorded SHA-256 file references in RALF evidence documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import cast


SHA_PATTERN = re.compile(r"(?i)(?<![0-9a-f])([0-9a-f]{64,})(?![0-9a-f])")
PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_$])((?:\.cache|models|scripts|docs)/[A-Za-z0-9_./=-]+)"
)
DOC_PATHS = (Path("models/ralf/TRAINING.md"), Path("PR270_BODY.md"))
MANIFEST_PATHS = (
    Path(".cache/ralf/training-reproduction/cgl/s5/manifest.json"),
    Path(".cache/ralf/training-reproduction/pku/s5/manifest.json"),
)
PARITY_ROOT = Path(".cache/ralf/training-reproduction/evaluation-path-parity-003")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def line_for_hash(path: Path, value: str) -> int:
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if value in line:
            return number
    return 0


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def search_roots(repo_root: Path, search_root: Path) -> list[Path]:
    return unique_paths([repo_root, *sorted(search_root.glob("ralf-*"))])


def candidate_paths(
    raw: str,
    *,
    base: Path,
    roots: list[Path],
    dataset: str | None = None,
    condition: str | None = None,
) -> list[Path]:
    path = Path(raw.rstrip("`.,;:)]}"))
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    elif raw.startswith(".cache/") or raw.startswith(("models/", "scripts/", "docs/")):
        candidates.extend(root / path for root in roots)
    else:
        candidates.append(base / path)

    if dataset and condition:
        condition_names = {condition, condition.replace("-", "_")}
        suffix = raw.removeprefix("campaign/")
        for root in roots:
            for name in condition_names:
                candidates.append(
                    root / ".cache/ralf/training-reproduction" / dataset / name / suffix
                )
    return unique_paths(candidates)


def resolve(
    raw: str,
    *,
    base: Path,
    roots: list[Path],
    dataset: str | None = None,
    condition: str | None = None,
) -> Path | None:
    for path in candidate_paths(
        raw, base=base, roots=roots, dataset=dataset, condition=condition
    ):
        if path.is_file():
            return path
    return None


def report_hash(
    *,
    source: Path,
    line: int,
    recorded: str,
    target: Path | None,
    counts: dict[str, int],
) -> None:
    if len(recorded) != 64:
        print(f"INVALID {source}:{line} length={len(recorded)} value={recorded}")
        counts["invalid"] += 1
        return
    if target is None:
        counts["unresolved"] += 1
        return
    actual = digest(target)
    counts["checked"] += 1
    if actual != recorded:
        print(
            f"MISMATCH {source}:{line} expected={recorded} actual={actual} "
            f"path={target}"
        )
        counts["mismatch"] += 1


def audit_json(
    path: Path, *, repo_root: Path, roots: list[Path], counts: dict[str, int]
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset") if isinstance(payload, dict) else None
    condition = payload.get("condition") if isinstance(payload, dict) else None

    def walk(value: object, parent: dict[str, object] | None = None) -> None:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            raw_path = mapping.get("path")
            recorded = mapping.get("sha256")
            if isinstance(raw_path, str) and isinstance(recorded, str):
                target = resolve(
                    raw_path,
                    base=path.parent,
                    roots=roots,
                    dataset=dataset if isinstance(dataset, str) else None,
                    condition=condition if isinstance(condition, str) else None,
                )
                report_hash(
                    source=path,
                    line=line_for_hash(path, recorded),
                    recorded=recorded,
                    target=target,
                    counts=counts,
                )
            for child in mapping.values():
                walk(child, mapping)
        elif isinstance(value, list):
            for child in cast(list[object], value):
                walk(child, parent)

    walk(payload)

    if isinstance(payload, dict):
        evaluator = payload.get("evaluator")
        if isinstance(evaluator, dict):
            raw_path = evaluator.get("runtime_freeze_record")
            recorded = evaluator.get("runtime_freeze_sha256")
            if isinstance(raw_path, str) and isinstance(recorded, str):
                report_hash(
                    source=path,
                    line=line_for_hash(path, recorded),
                    recorded=recorded,
                    target=resolve(raw_path, base=repo_root, roots=roots),
                    counts=counts,
                )
        runner = payload.get("runner")
        if isinstance(runner, dict):
            raw_path = runner.get("path")
            recorded = runner.get("sha256")
            if isinstance(raw_path, str) and isinstance(recorded, str):
                report_hash(
                    source=path,
                    line=line_for_hash(path, recorded),
                    recorded=recorded,
                    target=resolve(raw_path, base=repo_root, roots=roots),
                    counts=counts,
                )


def audit_document(
    path: Path, *, repo_root: Path, roots: list[Path], counts: dict[str, int]
) -> None:
    audit_document_text(
        path.read_text(encoding="utf-8"),
        source=path,
        repo_root=repo_root,
        roots=roots,
        counts=counts,
    )


def audit_document_text(
    text: str,
    *,
    source: Path,
    repo_root: Path,
    roots: list[Path],
    counts: dict[str, int],
) -> None:
    for line_number, line in enumerate(text.splitlines(), 1):
        hashes = list(SHA_PATTERN.finditer(line))
        paths = list(PATH_PATTERN.finditer(line))
        for match in hashes:
            recorded = match.group(1)
            nearest = min(
                paths, key=lambda item: abs(item.start() - match.start()), default=None
            )
            target = None
            if nearest is not None:
                target = resolve(nearest.group(1), base=repo_root, roots=roots)
            report_hash(
                source=source,
                line=line_number,
                recorded=recorded,
                target=target,
                counts=counts,
            )


def read_revision(path: Path, revision: str, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def audit_revision_documents(
    revision: str, *, repo_root: Path, roots: list[Path], counts: dict[str, int]
) -> None:
    for path in DOC_PATHS:
        text = read_revision(path, revision, repo_root)
        audit_document_text(
            text,
            source=Path(f"{revision}:{path}"),
            repo_root=repo_root,
            roots=roots,
            counts=counts,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    search_root = args.search_root.resolve() if args.search_root else repo_root.parent
    roots = search_roots(repo_root, search_root)
    counts = {"checked": 0, "mismatch": 0, "invalid": 0, "unresolved": 0}

    for path in DOC_PATHS:
        audit_document(path, repo_root=repo_root, roots=roots, counts=counts)
    for path in MANIFEST_PATHS:
        audit_json(path, repo_root=repo_root, roots=roots, counts=counts)
    for path in sorted(PARITY_ROOT.glob("**/evaluation-path-parity.json")):
        audit_json(path, repo_root=repo_root, roots=roots, counts=counts)
    if args.revision:
        audit_revision_documents(
            args.revision, repo_root=repo_root, roots=roots, counts=counts
        )

    print(
        "SUMMARY checked={checked} mismatches={mismatch} invalid={invalid} "
        "unresolved={unresolved}".format(**counts)
    )
    raise SystemExit(1 if counts["mismatch"] or counts["invalid"] else 0)


if __name__ == "__main__":
    main()
