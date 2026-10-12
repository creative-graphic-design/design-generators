"""AST and import-resolution helpers for annotation checks."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .constants import JAXTYPING_SHAPED_TYPES, SCAN_GLOBS, TORCH_TENSOR_TYPES


@dataclass(frozen=True)
class ImportResolver:
    """Resolve imported aliases in annotation expressions."""

    aliases: dict[str, str]

    @classmethod
    def from_tree(cls, tree: ast.Module) -> ImportResolver:
        """Return import aliases declared in a module."""
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".", 1)[0]
                    aliases[local] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    if alias.name == "*":
                        continue

                    local = alias.asname or alias.name
                    aliases[local] = f"{node.module}.{alias.name}"

        return cls(aliases)

    def resolve(self, name: str) -> str:
        """Resolve the first segment of a dotted name through imports."""
        head, dot, tail = name.partition(".")
        resolved = self.aliases.get(head, head)
        return f"{resolved}{dot}{tail}" if dot else resolved


def source_files(root: Path) -> list[Path]:
    """Return package source files covered by this check."""
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(path for path in root.glob(pattern) if path.is_file())

    return sorted(files)


def dotted_name(node: ast.AST) -> str | None:
    """Return the dotted name for a simple name or attribute expression."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        if prefix is None:
            return None

        return f"{prefix}.{node.attr}"

    return None


def is_jaxtyping_shaped_type(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether a subscript value is a jaxtyping shaped annotation."""
    name = dotted_name(node)
    if name is None:
        return False

    resolved = resolver.resolve(name)
    if resolved.startswith("jaxtyping."):
        return resolved.rsplit(".", 1)[-1] in JAXTYPING_SHAPED_TYPES

    return name.rsplit(".", 1)[-1] in JAXTYPING_SHAPED_TYPES


def parse_string_annotation(node: ast.Constant) -> ast.AST | None:
    """Return an AST expression for a string annotation, when parseable."""
    if not isinstance(node.value, str):
        return None

    try:
        return ast.parse(node.value, mode="eval").body
    except SyntaxError:
        return None


def contains_jaxtyping_shaped_subscript(
    node: ast.AST, resolver: ImportResolver
) -> bool:
    """Return whether an expression contains a jaxtyping shaped subscript."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = parse_string_annotation(node)
        if parsed is None:
            return False

        return contains_jaxtyping_shaped_subscript(parsed, resolver)

    if isinstance(node, ast.Subscript) and is_jaxtyping_shaped_type(
        node.value, resolver
    ):
        return True

    return any(
        contains_jaxtyping_shaped_subscript(child, resolver)
        for child in ast.iter_child_nodes(node)
    )


def is_raw_tensor_type(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether a node is a raw torch tensor or numpy ndarray type."""
    name = dotted_name(node)
    if name is None:
        return False

    resolved = resolver.resolve(name)
    if resolved in {
        "numpy.ndarray",
        "numpy.typing.NDArray",
        "numpy.typing.NDArray[Any]",
    }:
        return True

    return (
        resolved.startswith("torch.")
        and resolved.rsplit(".", 1)[-1] in TORCH_TENSOR_TYPES
    )


def contains_raw_annotation(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether an annotation contains a disallowed raw tensor reference."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = parse_string_annotation(node)
        if parsed is None:
            return False

        return contains_raw_annotation(parsed, resolver)

    if is_raw_tensor_type(node, resolver):
        return True

    if isinstance(node, ast.Subscript):
        if is_jaxtyping_shaped_type(node.value, resolver):
            return False

        return contains_raw_annotation(node.value, resolver) or contains_raw_annotation(
            node.slice, resolver
        )

    return any(
        contains_raw_annotation(child, resolver) for child in ast.iter_child_nodes(node)
    )


def is_object_type(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether a node is a bare object type annotation."""
    name = dotted_name(node)
    if name is None:
        return False

    resolved = resolver.resolve(name)
    return name == "object" or resolved == "builtins.object"


def is_any_type(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether a node is a bare ``Any`` type reference."""
    name = dotted_name(node)
    if name is None:
        return False

    resolved = resolver.resolve(name)
    return name == "Any" or resolved == "typing.Any"


def contains_weak_top_type(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether a type expression contains bare ``object`` or ``Any``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = parse_string_annotation(node)
        if parsed is None:
            return False

        return contains_weak_top_type(parsed, resolver)

    if is_object_type(node, resolver) or is_any_type(node, resolver):
        return True

    return any(
        contains_weak_top_type(child, resolver) for child in ast.iter_child_nodes(node)
    )


def contains_object_annotation(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether an annotation contains disallowed ``object`` or ``Any``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = parse_string_annotation(node)
        if parsed is None:
            return False

        return contains_object_annotation(parsed, resolver)

    if is_object_type(node, resolver) or is_any_type(node, resolver):
        return True

    return any(
        contains_object_annotation(child, resolver)
        for child in ast.iter_child_nodes(node)
    )


def rendered_annotation(node: ast.AST) -> ast.AST:
    """Return a parseable annotation expression for display and matching."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = parse_string_annotation(node)
        if parsed is not None:
            return parsed

    return node


def line_for_node(lines: list[str], node: ast.AST) -> str:
    """Return the normalized physical source line for an AST node."""
    lineno = getattr(node, "lineno", None)
    if lineno is None or not 1 <= lineno <= len(lines):
        return normalize_annotation(node)

    return " ".join(lines[lineno - 1].strip().split())


def is_type_alias_annotation(
    node: ast.AST | None, resolver: ImportResolver | None = None
) -> bool:
    """Return whether an annotation declares a TypeAlias assignment."""
    if node is None:
        return False

    name = dotted_name(node)
    if name is None:
        return False

    if resolver is not None:
        name = resolver.resolve(name)

    return name in {"TypeAlias", "typing.TypeAlias"}


def alias_target_name(node: ast.AST) -> str | None:
    """Return a module-level alias target name when the target is simple."""
    if isinstance(node, ast.Name):
        return node.id

    return None


def is_type_alias_target(node: ast.AST) -> bool:
    """Return whether an assignment target is likely a module-level type alias."""
    if isinstance(node, ast.Name):
        return node.id[:1].isupper() or node.id.endswith(
            ("Tensor", "Array", "Input", "Output", "Payload", "Value")
        )

    return dotted_name(node) == "TypeAlias"


def is_type_checking_guard(node: ast.AST, resolver: ImportResolver) -> bool:
    """Return whether an expression is a ``typing.TYPE_CHECKING`` guard."""
    name = dotted_name(node)
    if name is None:
        return False

    return name == "TYPE_CHECKING" or resolver.resolve(name) == "typing.TYPE_CHECKING"


def normalize_annotation(node: ast.AST) -> str:
    """Return a stable one-line representation for an annotation."""
    return " ".join(ast.unparse(rendered_annotation(node)).strip().split())
