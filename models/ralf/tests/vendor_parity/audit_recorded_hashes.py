#!/usr/bin/env python3
"""Audit recorded SHA-256 file references in RALF evidence documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import cast


SHA_PATTERN = re.compile(r"(?i)(?<![0-9a-f])([0-9a-f]{64,})(?![0-9a-f])")
PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_$])((?:\.cache|models|scripts|docs|evals)/[A-Za-z0-9_./=-]+|\$RALF_CACHE_DIR/[A-Za-z0-9_./=-]+)"
)
DOC_PATHS = (Path("models/ralf/TRAINING.md"), Path("PR270_BODY.md"))
MANIFEST_PATHS = (
    Path(".cache/ralf/training-reproduction/cgl/s5/manifest.json"),
    Path(".cache/ralf/training-reproduction/pku/s5/manifest.json"),
)
PARITY_ROOT = Path(".cache/ralf/training-reproduction/evaluation-path-parity-003")
DIGEST_CACHE: dict[Path, str] = {}


def digest(path: Path) -> str:
    resolved = path.resolve()
    cached = DIGEST_CACHE.get(resolved)
    if cached is not None:
        return cached
    value = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    result = value.hexdigest()
    DIGEST_CACHE[resolved] = result
    return result


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
    return unique_paths([repo_root, search_root, *sorted(search_root.glob("ralf-*"))])


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
    if raw.startswith("$RALF_CACHE_DIR/"):
        cache_root = os.environ.get("RALF_CACHE_DIR")
        if cache_root:
            candidates.append(Path(cache_root) / raw.removeprefix("$RALF_CACHE_DIR/"))
    elif path.is_absolute():
        candidates.append(path)
    elif raw.startswith(".cache/") or raw.startswith(("models/", "scripts/", "docs/")):
        candidates.extend(root / path for root in roots)
    else:
        candidates.append(base / path)

    if raw.startswith(("evals/", "training_logs/")):
        for root in roots:
            candidates.extend(
                root.glob(f".cache/ralf/training-reproduction/*/*/*/{raw}")
            )
            candidates.append(root / path)

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
        print(f"UNRESOLVED {source}:{line} value={recorded}")
        counts["unresolved"] += 1
        return
    actual = digest(target)
    counts["checked"] += 1
    if actual != recorded:
        label = target.as_posix()
        marker = "/training-reproduction/"
        if marker in label:
            label = ".cache/ralf/training-reproduction/" + label.split(marker, 1)[1]
        print(
            f"MISMATCH {source}:{line} expected={recorded} actual={actual} path={label}"
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


def rewrite_document(path: Path, *, repo_root: Path, roots: list[Path]) -> int:
    """Replace mismatched explicit file hashes with hashes computed here."""

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    replacements = 0
    for line_number, original in enumerate(lines, 1):
        line = original.rstrip("\n")
        hashes = list(SHA_PATTERN.finditer(line))
        paths = list(PATH_PATTERN.finditer(line))
        for match in reversed(hashes):
            target = document_target(
                line,
                match,
                paths=paths,
                base=repo_root,
                roots=roots,
                recorded=match.group(1),
            )
            if target is None or len(match.group(1)) != 64:
                continue
            actual = digest(target)
            if actual == match.group(1):
                continue
            start, end = match.span(1)
            line = line[:start] + actual + line[end:]
            replacements += 1
        lines[line_number - 1] = line + ("\n" if original.endswith("\n") else "")
    path.write_text("".join(lines), encoding="utf-8")
    return replacements


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
            target = document_target(
                line,
                match,
                paths=paths,
                base=repo_root,
                roots=roots,
                recorded=recorded,
            )
            report_hash(
                source=source,
                line=line_number,
                recorded=recorded,
                target=target,
                counts=counts,
            )


def document_target(
    line: str,
    match: re.Match[str],
    *,
    paths: list[re.Match[str]],
    base: Path,
    roots: list[Path],
    recorded: str | None = None,
) -> Path | None:
    """Resolve only a document hash that names a file explicitly.

    Training prose also contains digests of tensors, data-stream memberships,
    and sequence values.  Those values are evidence fields, not hashes of the
    nearest file path.  The labels below keep those values unresolved instead
    of silently assigning them to an unrelated artifact.
    """

    preceding = [item for item in paths if item.end() <= match.start()]
    previous_hashes = list(SHA_PATTERN.finditer(line[: match.start()]))
    context_start = max(
        line.rfind("|", 0, match.start()) + 1,
        line.rfind(";", 0, match.start()) + 1,
    )
    if previous_hashes:
        context_start = max(context_start, previous_hashes[-1].end())
    cell = line[context_start : match.start()].lower()
    candidates = preceding
    if not candidates:
        return None

    def resolve_document(raw: str) -> Path | None:
        matches = [
            path
            for path in candidate_paths(raw, base=base, roots=roots)
            if path.is_file()
        ]
        if raw.startswith("evals/") and len(matches) != 1:
            if recorded is None:
                return None
            matching = [path for path in matches if digest(path) == recorded]
            return matching[0] if len(matching) == 1 else None
        return matches[0] if matches else None

    def choose(suffix: str) -> Path | None:
        matches = [item for item in candidates if item.group(1).endswith(suffix)]
        if not matches:
            return None
        return resolve_document(matches[-1].group(1))

    if "manifest sha" in cell:
        return choose("manifest.json")
    if "comparison json sha" in cell or "comparison sha" in cell:
        return choose("comparison.json")
    if "analysis script sha" in cell or "comparison script" in cell:
        return choose(".py")
    if "loss-vector" in cell and "sha" in cell:
        return choose("loss-vectors.json")
    if "trace sha" in cell and "driver log" not in cell and "probe log" not in cell:
        matches = [item for item in candidates if "/trace/" in item.group(1)]
        if matches:
            return resolve_document(matches[-1].group(1))
        return None
    if "artifact sha" in cell or "stage sha" in cell:
        matches = [
            item
            for item in candidates
            if item.group(1).startswith(".cache/")
            and "loss-vectors" not in item.group(1)
        ]
        if matches:
            return resolve_document(matches[-1].group(1))
        return None

    nearest = candidates[-1]
    between = line[nearest.end() : match.start()]
    compact_between = between.strip(" `|,;:()")
    if not compact_between or (
        "sha-256" in between.lower() and len(compact_between) <= 12
    ):
        target = resolve_document(nearest.group(1))
        if target is not None:
            return target
    if match.start() - nearest.end() > 180:
        return None
    if not re.search(r"(?i)sha-?256|hash", between):
        return None
    if re.search(
        r"(?i)(?:canonical\s+stream|sequence|pad[- ]mask|tensor|digest|\bstate\b|\bloader\b|\bcondition\b|\bpackage\b|\bvendor\b|driver\s+log|probe\s+log)",
        between,
    ):
        return None
    return resolve_document(nearest.group(1))


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
        "--fix-documents",
        action="store_true",
        help="rewrite mismatched explicit file hashes in the current documents",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="audit documents without walking the launch manifests or parity JSON",
    )
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    search_roots_arg = [path.resolve() for path in args.search_root]
    if not search_roots_arg:
        search_roots_arg = [repo_root.parent]
    roots = unique_paths(
        [repo_root]
        + [
            item
            for root in search_roots_arg
            for item in [root, *sorted(root.glob("ralf-*"))]
        ]
    )
    counts = {"checked": 0, "mismatch": 0, "invalid": 0, "unresolved": 0}

    if args.fix_documents:
        fixed = sum(
            rewrite_document(path, repo_root=repo_root, roots=roots)
            for path in DOC_PATHS
        )
        print(f"FIXED document_hash_mismatches={fixed}")

    for path in DOC_PATHS:
        audit_document(path, repo_root=repo_root, roots=roots, counts=counts)
    if not args.documents_only:
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
