"""Run all four jaxtyping annotation policy surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from devharness.baselines import write_entry_baseline
from devharness.checks.model_readmes.constants import find_repo_root

from . import (
    baseline,
    object_annotations,
    raw_annotations,
    shaped_aliases,
    weak_cast_types,
)
from .constants import baseline_paths


def check(root: Path) -> int:
    """Run the four annotation policy surfaces in their established order."""
    raw_path, alias_path, object_path, weak_cast_path = baseline_paths(root)
    raw_status = baseline.check_jaxtyping_annotations(root, raw_path)
    alias_status = baseline.check_jaxtyping_aliases(root, alias_path)
    object_status = baseline.check_object_annotations(root, object_path)
    weak_cast_status = baseline.check_weak_cast_types(root, weak_cast_path)
    return raw_status or alias_status or object_status or weak_cast_status


def main(argv: list[str] | None = None) -> int:
    """Run the jaxtyping annotation checker."""
    parser = argparse.ArgumentParser(prog="devharness check jaxtyping-annotations")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite all four baselines from current annotations",
    )
    args = parser.parse_args([] if argv is None else argv)
    root = find_repo_root(Path.cwd())
    raw_path, alias_path, object_path, weak_cast_path = baseline_paths(root)
    if args.write_baseline:
        write_entry_baseline(raw_path, raw_annotations.current_entries(root))
        write_entry_baseline(alias_path, shaped_aliases.current_entries(root))
        write_entry_baseline(object_path, object_annotations.current_entries(root))
        write_entry_baseline(weak_cast_path, weak_cast_types.current_entries(root))
        return 0

    return check(root)
