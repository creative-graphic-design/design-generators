"""Weak object and Any references in cast target types."""

from __future__ import annotations

import ast
from pathlib import Path

from .ast_utils import (
    ImportResolver,
    contains_weak_top_type,
    dotted_name,
    line_for_node,
    normalize_annotation,
    source_files,
)
from .types import AnnotationRecord, WeakCastTypeViolation


class CastTypeVisitor(ast.NodeVisitor):
    """Collect ``typing.cast`` target type expressions."""

    def __init__(self, lines: list[str], resolver: ImportResolver) -> None:
        """Initialize source lines, import resolution, and cast targets."""
        self.lines = lines
        self.resolver = resolver
        self.cast_types: list[AnnotationRecord] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Collect the first argument to resolved ``typing.cast`` calls."""
        name = dotted_name(node.func)
        if name is not None and self.resolver.resolve(name) == "typing.cast":
            if node.args:
                cast_type = node.args[0]
                self.cast_types.append(
                    AnnotationRecord(cast_type, line_for_node(self.lines, cast_type))
                )

        self.generic_visit(node)


def violations(root: Path) -> list[WeakCastTypeViolation]:
    """Return weak top-type references in cast target types."""
    violations: list[WeakCastTypeViolation] = []
    for path in source_files(root):
        rel_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel_path)
        resolver = ImportResolver.from_tree(tree)
        visitor = CastTypeVisitor(text.splitlines(), resolver)
        visitor.visit(tree)
        for record in visitor.cast_types:
            if not contains_weak_top_type(record.node, resolver):
                continue

            violations.append(
                WeakCastTypeViolation(
                    rel_path,
                    normalize_annotation(record.node),
                    record.line,
                )
            )

    return violations


def current_entries(root: Path) -> set[str]:
    """Return current weak cast target type entries."""
    return {violation.as_baseline_entry() for violation in violations(root)}
