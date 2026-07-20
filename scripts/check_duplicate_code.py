"""Run the repository duplicate-code gate."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TARGET_DIRS = ("lib", "models", "scripts")
_DUPLICATE_START_RE = re.compile(r"^.+:\d+:\d+: R0801: Similar lines in \d+ files$")
_MODULE_SPAN_RE = re.compile(r"^(==[^:\n]+):\[\d+:\d+\](.*)$")


def is_python_target(path: str) -> bool:
    """Return whether path is a repository Python file covered by the gate."""
    if not path.endswith(".py"):
        return False
    return path.split("/", 1)[0] in TARGET_DIRS


def python_targets(root: Path) -> list[str]:
    """Return repository Python files covered by the duplicate-code gate."""
    return sorted(
        str(path.relative_to(root))
        for target in TARGET_DIRS
        for path in (root / target).rglob("*.py")
        if (root / target).is_dir()
    )


def changed_python_targets(root: Path) -> list[str]:
    """Return Python files changed by the branch or current local diff."""
    commands = [
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "origin/main...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMR"],
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    changed: set[str] = set()
    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode == 0:
            changed.update(
                path for path in result.stdout.splitlines() if is_python_target(path)
            )
    return sorted(changed)


def module_labels_for_path(path: str) -> set[str]:
    """Return Pylint module label variants that may refer to a file path."""
    without_suffix = path.removesuffix(".py")
    parts = without_suffix.split("/")
    labels = {parts[-1], ".".join(parts)}
    if "src" in parts:
        src_index = parts.index("src")
        labels.add(".".join(parts[src_index + 1 :]))
    if len(parts) >= 3 and parts[0] in {"lib", "models"}:
        labels.add(".".join(parts[2:]))
    return {label for label in labels if label}


def build_command(targets: list[str]) -> list[str]:
    """Build the Pylint duplicate-code command for Python targets."""
    return [
        "pylint",
        *targets,
        "--disable=all",
        "--enable=duplicate-code",
        "--ignore=vendor",
    ]


def parse_duplicate_blocks(output: str) -> list[str]:
    """Parse Pylint R0801 output into duplicate report blocks."""
    blocks: list[str] = []
    current: list[str] = []

    for line in output.splitlines():
        if _DUPLICATE_START_RE.match(line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            if line.startswith("-" * 10) or line.startswith("Your code has been rated"):
                blocks.append("\n".join(current))
                current = []
            else:
                current.append(line)

    if current:
        blocks.append("\n".join(current))
    return blocks


def block_module_labels(block: str) -> set[str]:
    """Return module labels mentioned in a duplicate report block."""
    labels: set[str] = set()
    for line in block.splitlines():
        module_span = _MODULE_SPAN_RE.match(line)
        if module_span:
            labels.add(module_span.group(1).removeprefix("=="))
    return labels


def label_matches(changed_label: str, report_label: str) -> bool:
    """Return whether a changed module label matches a Pylint report label."""
    return (
        changed_label == report_label
        or changed_label.endswith(f".{report_label}")
        or report_label.endswith(f".{changed_label}")
    )


def run_pylint(root: Path, targets: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Pylint duplicate-code detection and capture output."""
    return subprocess.run(
        build_command(targets),
        check=False,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def check_duplicate_code(root: Path, *, print_reports: bool = False) -> int:
    """Run duplicate-code checks and fail when changed modules are duplicated."""
    targets = python_targets(root)
    changed_targets = changed_python_targets(root)
    if not targets or not changed_targets:
        return 0

    result = run_pylint(root, targets)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    blocks = parse_duplicate_blocks(output)
    if print_reports:
        print(output, end="" if output.endswith("\n") else "\n")
        return 0

    changed_labels = {
        label
        for target in changed_targets
        for label in module_labels_for_path(target)
    }
    new_blocks = [
        block
        for block in blocks
        if any(
            label_matches(changed_label, report_label)
            for changed_label in changed_labels
            for report_label in block_module_labels(block)
        )
    ]
    if new_blocks:
        print("New duplicate-code reports were found:", file=sys.stderr)
        print("\n\n".join(new_blocks), file=sys.stderr)
        return 1

    if result.returncode != 0 and not blocks:
        print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
        return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run Pylint duplicate-code checks for repository Python files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-reports",
        action="store_true",
        help="print raw Pylint duplicate-code reports for diagnostics",
    )
    args = parser.parse_args(argv)
    return check_duplicate_code(Path.cwd(), print_reports=args.print_reports)


if __name__ == "__main__":
    raise SystemExit(main())
