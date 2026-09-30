"""Reject direct generator-consuming Torch sampling in package source."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_GLOBS = ("lib/*/src/**/*.py", "models/*/src/**/*.py")
HELPER_PATH = Path("lib/laygen/src/laygen/common/randomness.py")
TORCH_DRAW_NAMES = frozenset(
    {
        "bernoulli",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "randint",
        "randint_like",
        "rand_like",
        "randn",
        "randn_like",
        "randperm",
    }
)
IN_PLACE_DRAW_NAMES = frozenset(
    {
        "bernoulli_",
        "cauchy_",
        "exponential_",
        "geometric_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "log_normal_",
        "normal_",
        "orthogonal_",
        "poisson_",
        "random_",
        "sparse_",
        "trunc_normal_",
        "uniform_",
        "xavier_normal_",
        "xavier_uniform_",
    }
)
METHOD_DRAW_NAMES = frozenset({"bernoulli", "multinomial"})


@dataclass(frozen=True)
class SamplingViolation:
    """One forbidden generator construction or sampling call."""

    path: Path
    line: int
    message: str


def source_files(root: Path) -> list[Path]:
    """Return package source files covered by this check."""
    files = {
        path for pattern in SCAN_GLOBS for path in root.glob(pattern) if path.is_file()
    }
    return sorted(files)


def dotted_name(node: ast.AST) -> str | None:
    """Return the dotted name represented by a simple AST expression."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if not isinstance(node, ast.Name):
        return None

    parts.append(node.id)
    return ".".join(reversed(parts))


def import_aliases(tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    """Return import aliases and fully qualified imported module names."""
    aliases: dict[str, str] = {}
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".", 1)[0]
                aliases[local] = imported.name
                module_names.add(imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                if imported.name == "*":
                    continue

                local = imported.asname or imported.name
                aliases[local] = f"{node.module}.{imported.name}"
                if node.module == "laygen.common" and imported.name == "randomness":
                    module_names.add(f"{node.module}.{imported.name}")

    return aliases, module_names


def resolve_name(name: str | None, aliases: dict[str, str]) -> str | None:
    """Resolve the imported head of a dotted name."""
    if name is None:
        return None

    head, separator, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}{separator}{tail}" if separator else resolved


def has_generator_keyword(call: ast.Call) -> bool:
    """Return whether a call supplies or may supply a generator."""
    return any(keyword.arg in {None, "generator"} for keyword in call.keywords)


def is_cpu_literal(node: ast.AST) -> bool:
    """Return whether a device expression is the literal CPU string."""
    return isinstance(node, ast.Constant) and node.value == "cpu"


def check_tree(path: Path, root: Path, tree: ast.Module) -> list[SamplingViolation]:
    """Return forbidden sampling operations in one parsed source tree."""
    aliases, module_names = import_aliases(tree)
    violations: list[SamplingViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = resolve_name(dotted_name(node.func), aliases)
        if name == "torch.Generator":
            if (
                node.args
                or any(keyword.arg is None for keyword in node.keywords)
                or any(keyword.arg != "device" for keyword in node.keywords)
            ):
                violations.append(
                    SamplingViolation(
                        path.relative_to(root),
                        node.lineno,
                        "torch.Generator construction must use an implicit or literal CPU device",
                    )
                )
                continue

            for keyword in node.keywords:
                if keyword.arg != "device":
                    continue

                if not is_cpu_literal(keyword.value):
                    violations.append(
                        SamplingViolation(
                            path.relative_to(root),
                            node.lineno,
                            "torch.Generator(device=...) must use the literal CPU",
                        )
                    )

                break

            continue

        draw_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else name.rsplit(".", 1)[-1]
            if name is not None
            else None
        )
        receiver_name = (
            resolve_name(dotted_name(node.func.value), aliases)
            if isinstance(node.func, ast.Attribute)
            else None
        )
        is_torch_draw = name in {f"torch.{draw_name}" for draw_name in TORCH_DRAW_NAMES}
        is_in_place_draw = draw_name in IN_PLACE_DRAW_NAMES
        is_tensor_method_draw = (
            isinstance(node.func, ast.Attribute)
            and draw_name in METHOD_DRAW_NAMES
            and (
                receiver_name is None
                or (
                    receiver_name not in module_names
                    and receiver_name != "torch"
                    and not receiver_name.startswith("torch.")
                )
            )
        )
        if not (
            is_torch_draw or is_in_place_draw or is_tensor_method_draw
        ) or not has_generator_keyword(node):
            continue

        operation = name or draw_name or "sampling"
        violations.append(
            SamplingViolation(
                path.relative_to(root),
                node.lineno,
                f"direct generator-consuming {operation} call",
            )
        )

    return violations


def check_file(path: Path, root: Path) -> list[SamplingViolation]:
    """Return forbidden sampling operations in one source file."""
    if path.relative_to(root) == HELPER_PATH:
        return []

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        line = exc.lineno or 1 if isinstance(exc, SyntaxError) else 1
        return [
            SamplingViolation(
                path.relative_to(root), line, f"could not parse source: {exc}"
            )
        ]

    return check_tree(path, root, tree)


def check_generator_sampling(root: Path = ROOT) -> int:
    """Run the generator-consuming Torch sampling check."""
    violations = [
        violation for path in source_files(root) for violation in check_file(path, root)
    ]
    if not violations:
        return 0

    print("Forbidden generator-consuming Torch sampling:", file=sys.stderr)
    for violation in violations:
        print(
            f"{violation.path}:{violation.line}: {violation.message}",
            file=sys.stderr,
        )

    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the generator sampling checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    return check_generator_sampling(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
