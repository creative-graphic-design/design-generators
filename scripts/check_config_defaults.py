"""Reject synthesized default configs in public source code."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SOURCE_DIRS = ("lib", "models")


def is_config_constructor(node: ast.AST) -> bool:
    """Return whether an AST node calls a class/function ending in ``Config``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id.endswith("Config")
    if isinstance(func, ast.Attribute):
        return func.attr.endswith("Config")
    return False


def contains_config_constructor(node: ast.AST) -> bool:
    """Return whether a node contains any synthesized config constructor call."""
    return any(is_config_constructor(child) for child in ast.walk(node))


class ConfigFallbackVisitor(ast.NodeVisitor):
    """Collect config constructor calls hidden behind ``or`` fallbacks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[tuple[int, int, str]] = []

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Flag ``left or SomeConfig(...)`` and nested constructor variants."""
        if isinstance(node.op, ast.Or):
            for value in node.values[1:]:
                if contains_config_constructor(value):
                    source = ast.unparse(node)
                    self.violations.append((node.lineno, node.col_offset + 1, source))
                    break
        self.generic_visit(node)


def python_targets(root: Path) -> list[Path]:
    """Return Python source files covered by the config-default checker."""
    targets: list[Path] = []
    for top_level in SOURCE_DIRS:
        for src_dir in (root / top_level).glob("*/src"):
            targets.extend(src_dir.rglob("*.py"))
    return sorted(targets)


def check_file(root: Path, path: Path) -> list[tuple[Path, int, int, str]]:
    """Return synthesized-config fallback reports for one Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [
            (
                path.relative_to(root),
                exc.lineno or 1,
                exc.offset or 1,
                f"syntax error: {exc.msg}",
            )
        ]
    visitor = ConfigFallbackVisitor(path)
    visitor.visit(tree)
    return [
        (path.relative_to(root), line, column, source)
        for line, column, source in visitor.violations
    ]


def check_config_defaults(root: Path) -> int:
    """Run the repository source-code config-default check."""
    reports = [
        report for path in python_targets(root) for report in check_file(root, path)
    ]
    if not reports:
        return 0
    print("Synthesized default config fallbacks are not allowed:", file=sys.stderr)
    for path, line, column, source in reports:
        print(f"{path}:{line}:{column}: {source}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the config-default checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root to scan",
    )
    args = parser.parse_args(argv)
    return check_config_defaults(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
