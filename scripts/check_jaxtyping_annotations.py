"""Reject new raw tensor and ndarray annotations in package source."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "jaxtyping_baseline.txt"
SCAN_GLOBS = ("models/*/src/**/*.py", "lib/*/src/**/*.py")

JAXTYPING_SHAPED_TYPES = {
    "Bool",
    "Complex",
    "Complex64",
    "Complex128",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Inexact",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "Num",
    "Real",
    "Shaped",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
}

TORCH_TENSOR_TYPES = {
    "Tensor",
    "BoolTensor",
    "ByteTensor",
    "CharTensor",
    "DoubleTensor",
    "FloatTensor",
    "HalfTensor",
    "IntTensor",
    "LongTensor",
    "ShortTensor",
}


@dataclass(frozen=True)
class AnnotationViolation:
    """A raw tensor annotation violation."""

    path: str
    annotation: str
    line: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.annotation}\t{self.line}"


@dataclass(frozen=True)
class AnnotationRecord:
    """An annotation expression and its source line."""

    node: ast.AST
    line: str


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
    if (
        resolved.startswith("torch.")
        and resolved.rsplit(".", 1)[-1] in TORCH_TENSOR_TYPES
    ):
        return True
    return False


def parse_string_annotation(node: ast.Constant) -> ast.AST | None:
    """Return an AST expression for a string annotation, when parseable."""
    if not isinstance(node.value, str):
        return None
    try:
        return ast.parse(node.value, mode="eval").body
    except SyntaxError:
        return None


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


def is_type_alias_annotation(node: ast.AST | None) -> bool:
    """Return whether an annotation declares a TypeAlias assignment."""
    return node is not None and dotted_name(node) in {"TypeAlias", "typing.TypeAlias"}


def is_type_alias_target(node: ast.AST) -> bool:
    """Return whether an assignment target is a conventional type alias name."""
    if isinstance(node, ast.Name):
        return node.id.endswith("Tensor") or node.id.endswith("Array")
    return dotted_name(node) == "TypeAlias"


def normalize_annotation(node: ast.AST) -> str:
    """Return a stable one-line representation for an annotation."""
    return " ".join(ast.unparse(rendered_annotation(node)).strip().split())


class AnnotationVisitor(ast.NodeVisitor):
    """Collect annotation expressions from a Python module AST."""

    def __init__(self, lines: list[str]) -> None:
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


def raw_annotation_violations(root: Path) -> list[AnnotationViolation]:
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
    return {
        violation.as_baseline_entry() for violation in raw_annotation_violations(root)
    }


def baseline_entries(path: Path) -> set[str]:
    """Return committed baseline entries."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def git_output(root: Path, command: list[str]) -> str | None:
    """Return stdout for a best-effort git command."""
    result = subprocess.run(
        command,
        check=False,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def baseline_reference_entries(root: Path, baseline_path: Path) -> set[str] | None:
    """Return baseline entries from the merge-base with origin/main, if present."""
    merge_base = git_output(root, ["git", "merge-base", "origin/main", "HEAD"])
    if merge_base is None:
        return None
    rel_path = baseline_path.relative_to(root).as_posix()
    content = git_output(root, ["git", "show", f"{merge_base.strip()}:{rel_path}"])
    if content is None:
        return None
    return {line for line in content.splitlines() if line and not line.startswith("#")}


def write_baseline(path: Path, entries: Iterable[str]) -> None:
    """Write sorted baseline entries."""
    content = "\n".join(sorted(entries))
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def check_jaxtyping_annotations(root: Path, baseline_path: Path) -> int:
    """Check current raw annotations against the shrink-only baseline."""
    current = current_entries(root)
    baseline = baseline_entries(baseline_path)
    reference = baseline_reference_entries(root, baseline_path)
    baseline_additions = sorted(baseline - reference) if reference is not None else []
    unexpected = sorted(current - baseline)
    if not baseline_additions and not unexpected:
        return 0
    if baseline_additions:
        print("New jaxtyping baseline entries:", file=sys.stderr)
        for entry in baseline_additions:
            print(f"  + {entry}", file=sys.stderr)
    if unexpected:
        print("New raw tensor/ndarray annotations in package source:", file=sys.stderr)
        for entry in unexpected:
            print(f"  + {entry}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the jaxtyping annotation checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline from current raw annotations",
    )
    args = parser.parse_args(argv)
    if args.write_baseline:
        write_baseline(BASELINE_PATH, current_entries(ROOT))
        return 0
    return check_jaxtyping_annotations(ROOT, BASELINE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
