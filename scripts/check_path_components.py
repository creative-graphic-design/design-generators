"""Reject Python path strings that combine multiple components."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_CLASS_NAMES = frozenset(
    {
        "Path",
        "PurePath",
        "PosixPath",
        "PurePosixPath",
        "WindowsPath",
        "PureWindowsPath",
    }
)
EXCLUDED_DIRECTORY_NAMES = frozenset({".git", ".venv", "__pycache__", "vendor"})


@dataclass(frozen=True)
class Violation:
    """A path-component style violation in one Python source file."""

    path: Path
    line: int
    column: int
    rule: str
    expression: str


def python_targets(root: Path) -> list[Path]:
    """Return Python files covered by the path-component check."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and not EXCLUDED_DIRECTORY_NAMES.intersection(path.relative_to(root).parts)
    )


def _path_import_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return path-class and pathlib-module names imported in one source file."""
    class_aliases = set(PATH_CLASS_NAMES)
    module_aliases = {"pathlib"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pathlib":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            for alias in node.names:
                if alias.name in PATH_CLASS_NAMES:
                    class_aliases.add(alias.asname or alias.name)

    return class_aliases, module_aliases


def _is_path_constructor(
    node: ast.AST, class_aliases: set[str], module_aliases: set[str]
) -> bool:
    """Return whether a call target names a pathlib path class."""
    if not isinstance(node, ast.Call):
        return False

    function = node.func
    if isinstance(function, ast.Name):
        return function.id in class_aliases

    return (
        isinstance(function, ast.Attribute)
        and function.attr in PATH_CLASS_NAMES
        and isinstance(function.value, ast.Name)
        and function.value.id in module_aliases
    )


def _string_constant(node: ast.AST) -> str | None:
    """Return the value of a string literal expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    return None


def _string_expression_has_path_separator(node: ast.AST) -> bool:
    """Return whether a string literal or f-string segment has a separator."""
    value = _string_constant(node)
    if value is not None:
        return _contains_path_separator(value)

    return isinstance(node, ast.JoinedStr) and any(
        isinstance(part, ast.Constant)
        and isinstance(part.value, str)
        and _contains_path_separator(part.value)
        for part in node.values
    )


def _contains_path_separator(value: str) -> bool:
    """Return whether a string contains a path separator."""
    return "/" in value or "\\" in value


def _file_anchor_parent_count(
    node: ast.AST, class_aliases: set[str], module_aliases: set[str]
) -> int:
    """Return the parent-chain length for a direct ``Path(__file__)`` anchor."""
    count = 0
    current = node
    while isinstance(current, ast.Attribute) and current.attr == "parent":
        count += 1
        current = current.value

    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        current = current.func.value

    if not isinstance(current, ast.Call) or len(current.args) != 1:
        return 0

    if not isinstance(current.args[0], ast.Name) or current.args[0].id != "__file__":
        return 0

    if not _is_path_constructor(current, class_aliases, module_aliases):
        return 0

    return count


def _file_anchored_parent_violations(
    tree: ast.Module,
    relative_path: Path,
    class_aliases: set[str],
    module_aliases: set[str],
) -> list[Violation]:
    """Report maximal repeated ``parent`` chains rooted at ``Path(__file__)``."""
    parent_attributes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "parent"
    ]
    nested_parent_ids = {
        id(node.value)
        for node in parent_attributes
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parent"
    }

    return [
        Violation(
            relative_path,
            node.lineno,
            node.col_offset + 1,
            "repeated-parent-chain",
            ast.unparse(node),
        )
        for node in parent_attributes
        if id(node) not in nested_parent_ids
        and _file_anchor_parent_count(node, class_aliases, module_aliases) >= 2
    ]


def check_file(root: Path, path: Path) -> list[Violation]:
    """Return path-component violations for one Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [
            Violation(
                path.relative_to(root),
                exc.lineno or 1,
                exc.offset or 1,
                "syntax-error",
                exc.msg,
            )
        ]

    class_aliases, module_aliases = _path_import_aliases(tree)
    reports: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_path_constructor(
            node, class_aliases, module_aliases
        ):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            for argument in arguments:
                if _string_expression_has_path_separator(argument):
                    reports.append(
                        Violation(
                            path.relative_to(root),
                            argument.lineno,
                            argument.col_offset + 1,
                            "path-constructor-string",
                            ast.unparse(argument),
                        )
                    )

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if _string_expression_has_path_separator(node.right):
                reports.append(
                    Violation(
                        path.relative_to(root),
                        node.right.lineno,
                        node.right.col_offset + 1,
                        "path-operand-string",
                        ast.unparse(node.right),
                    )
                )

    reports.extend(
        _file_anchored_parent_violations(
            tree, path.relative_to(root), class_aliases, module_aliases
        )
    )
    return sorted(reports, key=lambda violation: (violation.line, violation.column))


def check_path_components(root: Path) -> int:
    """Run the repository Python path-component style check."""
    reports = [
        report for path in python_targets(root) for report in check_file(root, path)
    ]
    if not reports:
        return 0

    print("Python paths must use one component per operand:", file=sys.stderr)
    for report in reports:
        print(
            f"{report.path}:{report.line}:{report.column}: "
            f"{report.rule}: {report.expression}",
            file=sys.stderr,
        )

    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the repository Python path-component style check."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to scan",
    )
    args = parser.parse_args(argv)
    return check_path_components(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
