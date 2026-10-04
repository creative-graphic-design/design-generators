"""Raw tensor and ndarray annotation detection."""

from __future__ import annotations

import ast
from pathlib import Path

from .ast_utils import (
    ImportResolver,
    contains_raw_annotation,
    is_type_alias_annotation,
    is_type_alias_target,
    line_for_node,
    normalize_annotation,
    source_files,
)
from .types import AnnotationRecord, AnnotationViolation


class AnnotationVisitor(ast.NodeVisitor):
    """Collect annotation expressions from a Python module AST."""

    def __init__(self, lines: list[str]) -> None:
        """Initialize source lines and the collected annotations."""
        self.lines = lines
        self.annotations: list[AnnotationRecord] = []

    def append_annotation(self, node: ast.AST) -> None:
        """Append an annotation with the source line that introduced it."""
        self.annotations.append(AnnotationRecord(node, line_for_node(self.lines, node)))

    def visit_arg(self, node: ast.arg) -> None:
        """Collect function argument annotations."""
        if node.annotation is not None:
            self.append_annotation(node.annotation)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function return annotations."""
        if node.returns is not None:
            self.append_annotation(node.returns)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async function return annotations."""
        if node.returns is not None:
            self.append_annotation(node.returns)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect variable annotations and typed aliases."""
        self.append_annotation(node.annotation)
        if is_type_alias_annotation(node.annotation) and node.value is not None:
            self.annotations.append(
                AnnotationRecord(node.value, line_for_node(self.lines, node))
            )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect unannotated type alias values."""
        if any(is_type_alias_target(target) for target in node.targets):
            self.annotations.append(
                AnnotationRecord(node.value, line_for_node(self.lines, node))
            )

        self.generic_visit(node)


def violations(root: Path) -> list[AnnotationViolation]:
    """Return raw tensor annotation violations under package source roots."""
    violations: list[AnnotationViolation] = []
    for path in source_files(root):
        rel_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel_path)
        resolver = ImportResolver.from_tree(tree)
        visitor = AnnotationVisitor(text.splitlines())
        visitor.visit(tree)
        for record in visitor.annotations:
            if not contains_raw_annotation(record.node, resolver):
                continue

            violations.append(
                AnnotationViolation(
                    rel_path,
                    normalize_annotation(record.node),
                    record.line,
                )
            )

    return violations


def current_entries(root: Path) -> set[str]:
    """Return current raw annotation entries."""
    return {violation.as_baseline_entry() for violation in violations(root)}
