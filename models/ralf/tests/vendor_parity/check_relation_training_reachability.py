#!/usr/bin/env python3
"""Check that the relation inference changes do not enter the training call graph."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = ROOT / "models" / "ralf" / "src" / "ralf"
CHANGED_MODULES = {
    "ralf.modeling_ralf",
    "ralf.pipeline_ralf",
    "ralf.relation_restriction",
}
SHARED_MODEL_MODULES = {"ralf.modeling_ralf"}
INFERENCE_ONLY_MODEL_SYMBOLS = {
    "_apply_decode_space_restriction",
    "_generate_sequences",
    "_restrict_only_category",
    "_restrict_reliable_label_or_size",
    "multinomial",
}


def module_path(module: str) -> Path:
    return PACKAGE_ROOT.joinpath(*module.removeprefix("ralf.").split(".")).with_suffix(
        ".py"
    )


def resolve_import(current: str, level: int, name: str | None) -> str | None:
    package_parts = current.split(".")[:-level]
    if name:
        package_parts.extend(name.split("."))
    candidate = ".".join(package_parts)
    path = module_path(candidate)
    if path.is_file():
        return candidate
    init_path = path.with_suffix("") / "__init__.py"
    return candidate if init_path.is_file() else None


def top_level_imports(module: str) -> set[str]:
    tree = ast.parse(module_path(module).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name == "ralf" or alias.name.startswith("ralf.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imported = resolve_import(module, node.level, node.module)
                if imported is not None:
                    imports.add(imported)
            elif node.module and (
                node.module == "ralf" or node.module.startswith("ralf.")
            ):
                imports.add(node.module)
    return imports


def import_closure(start: str) -> set[str]:
    seen: set[str] = set()
    pending = [start]
    while pending:
        module = pending.pop()
        if module in seen or not module_path(module).is_file():
            continue
        seen.add(module)
        pending.extend(top_level_imports(module) - seen)
    return seen


def training_method_calls() -> dict[str, set[str]]:
    module = "ralf.training.lightning_module"
    tree = ast.parse(module_path(module).read_text(encoding="utf-8"))
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = node

    calls: dict[str, set[str]] = {}
    for name, node in methods.items():
        calls[name] = {
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        }
    reachable: set[str] = set()
    pending = ["training_step", "validation_step"]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(calls.get(name, set()) - reachable)
    return {root: reachable for root in ("training_step", "validation_step")}


def modeling_inference_calls() -> set[str]:
    """Return inference-only modeling symbols reachable from model.forward."""
    module = "ralf.modeling_ralf"
    tree = ast.parse(module_path(module).read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RalfForConditionalLayoutGeneration"
    )
    model_methods = {
        node.name
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    reachable: set[str] = set()
    pending = ["forward"]
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        node = methods.get(name)
        if node is None:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ):
                target = call.func.attr
            elif isinstance(call.func, ast.Name):
                target = call.func.id
            else:
                continue
            if target in INFERENCE_ONLY_MODEL_SYMBOLS:
                found.add(target)
            if target in model_methods or target in methods:
                pending.append(target)
    return found


def main() -> int:
    closure = import_closure("ralf.training.lightning_module")
    calls = training_method_calls()
    imported_changed = sorted((closure & CHANGED_MODULES) - SHARED_MODEL_MODULES)
    shared_model_modules = sorted(closure & SHARED_MODEL_MODULES)
    modeling_changed = sorted(modeling_inference_calls())
    called_changed: list[str] = []
    print(f"training import closure changed modules: {imported_changed}")
    print(f"training import closure shared modeling modules: {shared_model_modules}")
    for root, reachable in calls.items():
        print(f"{root} reachable local methods: {sorted(reachable)}")
        print(f"{root} reachable changed symbols: {called_changed}")
    print(f"training model forward reachable inference symbols: {modeling_changed}")
    if imported_changed or called_changed or modeling_changed:
        return 1
    print("result: PASS, relation preparation and decoder are inference-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
