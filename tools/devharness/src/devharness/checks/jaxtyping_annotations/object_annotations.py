"""Function-signature object and Any annotation detection."""

from __future__ import annotations

import ast
from pathlib import Path

from .ast_utils import (
    ImportResolver,
    contains_object_annotation,
    is_type_checking_guard,
    line_for_node,
    normalize_annotation,
    source_files,
)
from .shaped_aliases import type_checking_shaped_alias_names
from .types import FunctionAnnotationRecord, ObjectAnnotationViolation


class FunctionAnnotationVisitor(ast.NodeVisitor):
    """Collect function parameter and return annotation expressions."""

    def __init__(self, lines: list[str]) -> None:
        """Initialize source lines and the collected annotations."""
        self.lines = lines
        self.annotations: list[FunctionAnnotationRecord] = []

    def append_annotation(
        self, node: ast.AST, *, is_keyword_variadic: bool = False
    ) -> None:
        """Append a signature annotation with its source line."""
        self.annotations.append(
            FunctionAnnotationRecord(
                node,
                line_for_node(self.lines, node),
                is_keyword_variadic=is_keyword_variadic,
            )
        )

    def append_arguments(self, args: ast.arguments) -> None:
        """Append annotations from a function argument list."""
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if arg.annotation is not None:
                self.append_annotation(arg.annotation)

        if args.vararg is not None and args.vararg.annotation is not None:
            self.append_annotation(args.vararg.annotation)

        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.append_annotation(args.kwarg.annotation, is_keyword_variadic=True)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function parameter and return annotations."""
        self.append_arguments(node.args)
        if node.returns is not None:
            self.append_annotation(node.returns)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async function parameter and return annotations."""
        self.append_arguments(node.args)
        if node.returns is not None:
            self.append_annotation(node.returns)

        self.generic_visit(node)


def violations(root: Path) -> list[ObjectAnnotationViolation]:
    """Return weak object/Any signature and alias-fallback violations."""
    violations: list[ObjectAnnotationViolation] = []
    for path in source_files(root):
        rel_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text, filename=rel_path)
        resolver = ImportResolver.from_tree(tree)
        visitor = FunctionAnnotationVisitor(lines)
        visitor.visit(tree)
        for record in visitor.annotations:
            if not contains_object_annotation(record.node, resolver):
                continue

            violations.append(
                ObjectAnnotationViolation(
                    rel_path,
                    normalize_annotation(record.node),
                    record.line,
                )
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not is_type_checking_guard(
                node.test, resolver
            ):
                continue

            shaped_aliases = type_checking_shaped_alias_names(node.body, resolver)
            for fallback in node.orelse:
                if not isinstance(
                    fallback, ast.Assign
                ) or not contains_object_annotation(fallback.value, resolver):
                    continue

                for target in fallback.targets:
                    alias = target.id if isinstance(target, ast.Name) else None
                    if alias not in shaped_aliases:
                        continue

                    violations.append(
                        ObjectAnnotationViolation(
                            rel_path,
                            normalize_annotation(fallback.value),
                            line_for_node(lines, fallback),
                        )
                    )

    return violations


def current_entries(root: Path) -> set[str]:
    """Return current function object annotation entries."""
    return {violation.as_baseline_entry() for violation in violations(root)}
