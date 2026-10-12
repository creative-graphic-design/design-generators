"""Merge-base baseline references and policy diagnostics."""

from __future__ import annotations

from pathlib import Path

from devharness.baselines import print_entries, read_entry_baseline
from devharness.git import git_output

from . import object_annotations, raw_annotations, shaped_aliases, weak_cast_types


def baseline_reference_entries(root: Path, baseline_path: Path) -> set[str] | None:
    """Return baseline entries from merge-base with origin/main, if present."""
    merge_base = git_output(root, ["git", "merge-base", "origin/main", "HEAD"])
    if merge_base is None:
        return None

    rel_path = baseline_path.relative_to(root).as_posix()
    content = git_output(root, ["git", "show", f"{merge_base.strip()}:{rel_path}"])
    if content is None:
        return None

    return {line for line in content.splitlines() if line and not line.startswith("#")}


def check_jaxtyping_annotations(root: Path, baseline_path: Path) -> int:
    """Check raw annotations against the shrink-only baseline."""
    current = raw_annotations.current_entries(root)
    baseline = read_entry_baseline(baseline_path)
    reference = baseline_reference_entries(root, baseline_path)
    baseline_additions = sorted(baseline - reference) if reference is not None else []
    unexpected = sorted(current - baseline)
    if not baseline_additions and not unexpected:
        return 0

    print_entries("New jaxtyping baseline entries:", "+", baseline_additions)
    print_entries(
        "New raw tensor/ndarray annotations in package source:", "+", unexpected
    )
    return 1


def check_jaxtyping_aliases(root: Path, baseline_path: Path) -> int:
    """Check shaped aliases against the shrink-only baseline."""
    current = shaped_aliases.current_entries(root)
    baseline = read_entry_baseline(baseline_path)
    reference = baseline_reference_entries(root, baseline_path)
    baseline_additions = sorted(baseline - reference) if reference is not None else []
    unexpected = sorted(current - baseline)
    if not baseline_additions and not unexpected:
        return 0

    print_entries("New jaxtyping alias baseline entries:", "+", baseline_additions)
    print_entries(
        "New jaxtyping shaped-type aliases in package source:", "+", unexpected
    )
    return 1


def check_object_annotations(root: Path, baseline_path: Path) -> int:
    """Check function object annotations against the shrink-only baseline."""
    current = object_annotations.current_entries(root)
    baseline = read_entry_baseline(baseline_path)
    reference = baseline_reference_entries(root, baseline_path)
    baseline_additions = sorted(baseline - reference) if reference is not None else []
    unexpected = sorted(current - baseline)
    if not baseline_additions and not unexpected:
        return 0

    print_entries("New object annotation baseline entries:", "+", baseline_additions)
    print_entries("New object annotations in function signatures:", "+", unexpected)
    return 1


def check_weak_cast_types(root: Path, baseline_path: Path) -> int:
    """Check weak cast target types against the shrink-only baseline."""
    current = weak_cast_types.current_entries(root)
    baseline = read_entry_baseline(baseline_path)
    reference = baseline_reference_entries(root, baseline_path)
    baseline_additions = sorted(baseline - reference) if reference is not None else []
    unexpected = sorted(current - baseline)
    if not baseline_additions and not unexpected:
        return 0

    print_entries("New weak cast type baseline entries:", "+", baseline_additions)
    print_entries(
        "New bare object/Any references in cast target types:", "+", unexpected
    )
    return 1
