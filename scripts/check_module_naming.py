"""Enforce model package module naming conventions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "module_naming_baseline.txt"

HF_CORE_PREFIXES = (
    "configuration",
    "modeling",
    "pipeline",
    "scheduling",
    "processing",
    "tokenization",
    "image_processing",
    "generation",
)

ALLOWLIST_CATEGORIES = {
    "package": {
        "__init__.py",
    },
    "repository-convention": {
        "conversion.py",
        "data.py",
        "datasets.py",
        "model_card.py",
        "postprocessing.py",
        "testing.py",
        "visualization.py",
    },
    "conversion-support": {
        "tf_checkpoint.py",
        "vendor_state.py",
        "vendor_state_dict.py",
    },
    "layout-domain": {
        "bbox.py",
        "constraints.py",
        "data_specs.py",
        "geometry.py",
        "graph_schema.py",
        "hierarchy.py",
        "labels.py",
        "masking.py",
        "relation_schema.py",
        "retrieval.py",
        "serialization.py",
        "tasks.py",
        "types.py",
    },
    "poster-domain": {
        "candidate_generation.py",
        "color.py",
    },
}

PROMPT_AGENT_MODULES = {
    "layout_gpt": {
        "agent.py",
        "enums.py",
        "exemplars.py",
        "parser.py",
        "prompts.py",
        "schema.py",
        "types.py",
    },
    "layoutprompter": {
        "agent.py",
        "arrays.py",
        "data.py",
        "enums.py",
        "parsing.py",
        "records.py",
        "schemas.py",
        "selection.py",
        "serialization.py",
        "similarity.py",
        "vendor_parity.py",
    },
}


@dataclass(frozen=True)
class ModuleRecord:
    """Direct module under a model package."""

    package: str
    path: Path
    root: Path = ROOT

    @property
    def relative_path(self) -> str:
        """Return the repository-relative module path."""
        return self.path.relative_to(self.root).as_posix()


@dataclass(frozen=True)
class NamingViolation:
    """Module naming violation recorded by the checker."""

    path: str
    reason: str

    def as_baseline_entry(self) -> str:
        """Return the stable baseline entry for this violation."""
        return f"{self.path}\t{self.reason}"


def expected_core_names(package: str) -> set[str]:
    """Return allowed HF-style core filenames for a package."""
    return {f"{prefix}_{package}.py" for prefix in HF_CORE_PREFIXES}


def direct_model_modules(root: Path) -> list[ModuleRecord]:
    """Return direct Python modules in every model package source directory."""
    modules: list[ModuleRecord] = []
    for src_dir in sorted((root / "models").glob("*/src")):
        if not src_dir.is_dir():
            continue
        for package_dir in sorted(path for path in src_dir.iterdir() if path.is_dir()):
            modules.extend(
                ModuleRecord(package=package_dir.name, path=path, root=root)
                for path in sorted(package_dir.glob("*.py"))
            )
    return modules


def allowlist_category(name: str) -> str | None:
    """Return the allowlist category for a module filename, if any."""
    for category, names in ALLOWLIST_CATEGORIES.items():
        if name in names:
            return category
    return None


def has_hf_core_prefix(name: str) -> bool:
    """Return whether a filename looks like an HF-style core module."""
    stem = name.removesuffix(".py")
    return any(
        stem == prefix or stem.startswith(f"{prefix}_") for prefix in HF_CORE_PREFIXES
    )


def violation_for_module(module: ModuleRecord) -> NamingViolation | None:
    """Return a naming violation when the module is not explicitly allowed."""
    name = module.path.name
    expected = expected_core_names(module.package)
    if name in expected:
        return None
    if has_hf_core_prefix(name):
        expected_names = ", ".join(sorted(expected))
        return NamingViolation(
            module.relative_path,
            f"HF core module must use the package suffix; expected one of: {expected_names}",
        )
    if allowlist_category(name) is not None:
        return None
    if name in PROMPT_AGENT_MODULES.get(module.package, set()):
        return None
    return NamingViolation(
        module.relative_path,
        "direct model package module is not HF core-named or in an allowlist category",
    )


def current_violations(root: Path) -> list[NamingViolation]:
    """Return current naming violations."""
    return sorted(
        (
            violation
            for module in direct_model_modules(root)
            if (violation := violation_for_module(module)) is not None
        ),
        key=lambda violation: violation.path,
    )


def current_entries(root: Path) -> set[str]:
    """Return current violation entries."""
    return {violation.as_baseline_entry() for violation in current_violations(root)}


def baseline_entries(path: Path) -> set[str]:
    """Return committed shrink-only baseline entries."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def write_baseline(path: Path, entries: Iterable[str]) -> None:
    """Write sorted baseline entries."""
    content = "\n".join(sorted(entries))
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def check_module_naming(root: Path, baseline_path: Path) -> int:
    """Check current violations against the shrink-only baseline."""
    current = current_entries(root)
    baseline = baseline_entries(baseline_path)
    unexpected = sorted(current - baseline)
    stale = sorted(baseline - current)
    if not unexpected and not stale:
        return 0
    if unexpected:
        print("New non-conforming model module filenames:", file=sys.stderr)
        for entry in unexpected:
            print(f"  + {entry}", file=sys.stderr)
    if stale:
        print("Stale module naming baseline entries:", file=sys.stderr)
        for entry in stale:
            print(f"  - {entry}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the model module naming checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the shrink-only baseline from current violations",
    )
    args = parser.parse_args(argv)
    if args.write_baseline:
        write_baseline(BASELINE_PATH, current_entries(ROOT))
        return 0
    return check_module_naming(ROOT, BASELINE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
