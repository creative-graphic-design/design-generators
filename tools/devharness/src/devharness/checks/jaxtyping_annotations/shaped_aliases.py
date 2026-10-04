"""Jaxtyping shaped-type alias detection."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from .ast_utils import (
    ImportResolver,
    alias_target_name,
    contains_jaxtyping_shaped_subscript,
    is_type_alias_annotation,
    line_for_node,
    normalize_annotation,
    source_files,
)
from .types import AliasViolation


def type_checking_shaped_alias_names(
    body: Iterable[ast.stmt], resolver: ImportResolver
) -> set[str]:
    """Return shaped type aliases declared in a TYPE_CHECKING branch."""
    aliases: set[str] = set()
    for stmt in body:
        if isinstance(stmt, ast.Assign) and contains_jaxtyping_shaped_subscript(
            stmt.value, resolver
        ):
            for target in stmt.targets:
                alias = alias_target_name(target)
                if alias is not None:
                    aliases.add(alias)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and contains_jaxtyping_shaped_subscript(stmt.value, resolver)
        ):
            alias = alias_target_name(stmt.target)
            if alias is not None:
                aliases.add(alias)

    return aliases


def violations(root: Path) -> list[AliasViolation]:
    """Return module-level jaxtyping shaped-type alias violations."""
    violations: list[AliasViolation] = []
    for path in source_files(root):
        rel_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text, filename=rel_path)
        resolver = ImportResolver.from_tree(tree)
        for node in tree.body:
            if isinstance(node, ast.Assign) and contains_jaxtyping_shaped_subscript(
                node.value, resolver
            ):
                for target in node.targets:
                    alias = alias_target_name(target)
                    if alias is None:
                        continue

                    violations.append(
                        AliasViolation(
                            rel_path,
                            alias,
                            normalize_annotation(node.value),
                            line_for_node(lines, node),
                        )
                    )
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and is_type_alias_annotation(node.annotation, resolver)
                and contains_jaxtyping_shaped_subscript(node.value, resolver)
            ):
                alias = alias_target_name(node.target)
                if alias is None:
                    continue

                violations.append(
                    AliasViolation(
                        rel_path,
                        alias,
                        normalize_annotation(node.value),
                        line_for_node(lines, node),
                    )
                )

    return violations


def current_entries(root: Path) -> set[str]:
    """Return current jaxtyping alias entries."""
    return {violation.as_baseline_entry() for violation in violations(root)}
