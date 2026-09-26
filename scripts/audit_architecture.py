"""Report repository architecture facts without enforcing policy.

Member stats cover only ``src/`` and ``tests/``, while hotspots scan Python
files under ``lib/``, ``models/``, and ``scripts/`` except paths containing
``tests`` or ``vendor``.

The original report payload keys remain stable. The ``import_audit`` payload
key is an additive schema entry for direct imports in ``scripts/*.py`` and
``tests/*.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Final, TypeAlias, cast

WORKSPACE_KINDS: Final[dict[str, str]] = {
    "lib": "library",
    "models": "model",
}
PYTHON_TARGET_DIRS: Final[tuple[str, ...]] = ("lib", "models", "scripts")
_REQUIREMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
IMPORT_DISTRIBUTION_OVERRIDES: Final[dict[str, str]] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "pytorch_fid": "pytorch-fid",
    "yaml": "PyYAML",
}

TomlValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | time
    | list["TomlValue"]
    | dict[str, "TomlValue"]
)
TomlTable: TypeAlias = dict[str, TomlValue]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class DependencyEdge:
    """One dependency edge between uv workspace members."""

    source: str
    target: str
    scope: str


@dataclass(frozen=True)
class MemberStats:
    """Architecture facts for one uv workspace member."""

    name: str
    path: str
    kind: str
    framework: str | None
    task: str | None
    source_files: int
    source_lines: int
    test_files: int
    test_lines: int
    largest_source_file: str | None
    largest_source_lines: int


@dataclass(frozen=True)
class SourceHotspot:
    """One large Python source file."""

    path: str
    lines: int


@dataclass(frozen=True)
class ImportRecord:
    """One direct import found in a root tooling file."""

    path: str
    module: str
    category: str
    distribution: str | None
    declared_scopes: tuple[str, ...]


@dataclass(frozen=True)
class ImportAudit:
    """Direct-import classifications and root dependency ownership results."""

    records: tuple[ImportRecord, ...] = ()
    undeclared_distributions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureReport:
    """Deterministic architecture report for the repository."""

    members: tuple[MemberStats, ...]
    dependency_edges: tuple[DependencyEdge, ...]
    hotspots: tuple[SourceHotspot, ...]
    warnings: tuple[str, ...]
    import_audit: ImportAudit = field(default_factory=ImportAudit)

    @property
    def library_count(self) -> int:
        """Return the number of shared-library workspace members."""
        return sum(member.kind == "library" for member in self.members)

    @property
    def model_count(self) -> int:
        """Return the number of model workspace members."""
        return sum(member.kind == "model" for member in self.members)

    @property
    def root_count(self) -> int:
        """Return the number of root-project workspace members."""
        return sum(member.kind == "root" for member in self.members)


def normalize_requirement_name(requirement: str) -> str:
    """Return a normalized distribution name from a PEP 508-like requirement."""
    match = _REQUIREMENT_NAME_RE.match(requirement.strip())
    if match is None:
        raise ValueError(f"cannot parse requirement name: {requirement!r}")

    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _load_toml(path: Path) -> TomlTable:
    with path.open("rb") as file:
        config = tomllib.load(file)

    return cast(TomlTable, config)


def _dict_value(value: TomlValue | None) -> TomlTable:
    if not isinstance(value, dict):
        return {}

    return cast(TomlTable, value)


def _string_list(value: TomlValue | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()

    return tuple(item for item in value if isinstance(item, str))


def _string_value(value: TomlValue | None) -> str | None:
    return value if isinstance(value, str) else None


def _project_table(config: TomlTable) -> TomlTable:
    return _dict_value(config.get("project"))


def _workspace_patterns(root_config: TomlTable) -> tuple[str, ...]:
    tool = _dict_value(root_config.get("tool"))
    uv = _dict_value(tool.get("uv"))
    workspace = _dict_value(uv.get("workspace"))
    return _string_list(workspace.get("members"))


def discover_member_paths(root: Path) -> tuple[Path, ...]:
    """Return uv workspace member directories in deterministic order."""
    root_config = _load_toml(root / "pyproject.toml")
    root_project_name = _string_value(_project_table(root_config).get("name"))
    members: set[Path] = {root} if root_project_name is not None else set()

    for pattern in _workspace_patterns(root_config):
        for candidate in root.glob(pattern):
            if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
                members.add(candidate)

    return tuple(
        sorted(
            members,
            key=lambda path: (
                path != root,
                path.relative_to(root).as_posix(),
            ),
        )
    )


def _python_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()

    return tuple(sorted(path for path in directory.rglob("*.py") if path.is_file()))


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _member_metadata(config: TomlTable) -> tuple[str | None, str | None]:
    tool = _dict_value(config.get("tool"))
    metadata = _dict_value(tool.get("design-generators"))
    return _string_value(metadata.get("framework")), _string_value(metadata.get("task"))


def _member_kind(root: Path, member_path: Path) -> str:
    if member_path == root:
        return "root"

    relative = member_path.relative_to(root)
    return WORKSPACE_KINDS.get(relative.parts[0], "other")


def _member_stats(root: Path, member_path: Path) -> MemberStats:
    config = _load_toml(member_path / "pyproject.toml")
    project = _project_table(config)
    name = _string_value(project.get("name"))
    if name is None:
        raise ValueError(f"{member_path / 'pyproject.toml'} is missing project.name")

    relative = member_path.relative_to(root)
    framework, task = _member_metadata(config)
    source_files = _python_files(member_path / "src")
    test_files = _python_files(member_path / "tests")
    source_line_counts = tuple((path, _line_count(path)) for path in source_files)
    largest = max(source_line_counts, key=lambda item: item[1], default=None)

    return MemberStats(
        name=name,
        path="." if member_path == root else relative.as_posix(),
        kind=_member_kind(root, member_path),
        framework=framework,
        task=task,
        source_files=len(source_files),
        source_lines=sum(lines for _, lines in source_line_counts),
        test_files=len(test_files),
        test_lines=sum(_line_count(path) for path in test_files),
        largest_source_file=(
            largest[0].relative_to(root).as_posix() if largest is not None else None
        ),
        largest_source_lines=largest[1] if largest is not None else 0,
    )


def _dependency_specs(config: TomlTable) -> tuple[tuple[str, str], ...]:
    project = _project_table(config)
    specs = [
        ("core", requirement)
        for requirement in _string_list(project.get("dependencies"))
    ]
    optional = _dict_value(project.get("optional-dependencies"))

    for extra_name in sorted(optional):
        specs.extend(
            (f"extra:{extra_name}", requirement)
            for requirement in _string_list(optional[extra_name])
        )

    groups = _dict_value(config.get("dependency-groups"))
    for group_name in sorted(groups):
        specs.extend(
            (f"group:{group_name}", requirement)
            for requirement in _string_list(groups[group_name])
        )

    return tuple(specs)


def _root_dependency_scopes(root: Path) -> dict[str, tuple[str, ...]]:
    """Return root dependency declarations keyed by normalized distribution."""
    scopes: dict[str, set[str]] = {}
    for scope, requirement in _dependency_specs(_load_toml(root / "pyproject.toml")):
        try:
            distribution = normalize_requirement_name(requirement)
        except ValueError:
            continue

        scopes.setdefault(distribution, set()).add(scope)

    return {
        distribution: tuple(sorted(declared_scopes))
        for distribution, declared_scopes in sorted(scopes.items())
    }


def _workspace_import_names(member_paths: tuple[Path, ...]) -> set[str]:
    names: set[str] = set()
    for member_path in member_paths:
        config = _load_toml(member_path / "pyproject.toml")
        project_name = _string_value(_project_table(config).get("name"))
        if project_name is not None:
            names.add(project_name.replace("-", "_").lower())

        source = member_path / "src"
        if not source.is_dir():
            continue

        for candidate in source.iterdir():
            if candidate.name.startswith("."):
                continue

            if candidate.is_dir() and (candidate / "__init__.py").is_file():
                names.add(candidate.name.lower())
            elif candidate.suffix == ".py":
                names.add(candidate.stem.lower())

    return names


def _import_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module.split(".", 1)[0])

    return tuple(sorted(modules))


def _import_audit_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for directory in (root / "scripts", root / "tests")
            for path in directory.glob("*.py")
            if path.is_file()
        )
    )


def audit_imports(
    root: Path, *, member_paths: tuple[Path, ...] | None = None
) -> ImportAudit:
    """Classify root tooling imports and find undeclared distributions."""
    members = discover_member_paths(root) if member_paths is None else member_paths
    script_modules = {
        path.stem.lower() for path in (root / "scripts").glob("*.py") if path.is_file()
    }
    workspace_modules = _workspace_import_names(members)
    dependency_scopes = _root_dependency_scopes(root)
    records: list[ImportRecord] = []

    for path in _import_audit_files(root):
        relative_path = path.relative_to(root).as_posix()
        for module in _import_modules(path):
            normalized_module = module.lower().replace("-", "_")
            if module in sys.stdlib_module_names:
                category = "stdlib"
                distribution = None
            elif normalized_module in script_modules or normalized_module == "scripts":
                category = "local-root-script"
                distribution = None
            elif normalized_module in workspace_modules:
                category = "workspace-package"
                distribution = None
            else:
                category = "third-party-distribution"
                distribution = IMPORT_DISTRIBUTION_OVERRIDES.get(
                    module, module.replace("_", "-")
                )

            declared_scopes = (
                dependency_scopes.get(normalize_requirement_name(distribution), ())
                if distribution is not None
                else ()
            )
            records.append(
                ImportRecord(
                    path=relative_path,
                    module=module,
                    category=category,
                    distribution=distribution,
                    declared_scopes=declared_scopes,
                )
            )

    sorted_records = tuple(
        sorted(records, key=lambda record: (record.path, record.module))
    )
    undeclared = tuple(
        sorted(
            {
                record.distribution
                for record in sorted_records
                if record.category == "third-party-distribution"
                and not record.declared_scopes
                and record.distribution is not None
            },
            key=lambda distribution: (distribution.lower(), distribution),
        )
    )
    return ImportAudit(records=sorted_records, undeclared_distributions=undeclared)


def _dependency_edges(
    member_paths: tuple[Path, ...],
    member_names: dict[str, str],
) -> tuple[tuple[DependencyEdge, ...], tuple[str, ...]]:
    edges: set[DependencyEdge] = set()
    warnings: list[str] = []
    for member_path in member_paths:
        config = _load_toml(member_path / "pyproject.toml")
        source = _string_value(_project_table(config).get("name"))
        if source is None:
            continue

        for scope, requirement in _dependency_specs(config):
            try:
                normalized_target = normalize_requirement_name(requirement)
            except ValueError:
                warnings.append(f"{source} ({scope}): {requirement}")
                continue

            target = member_names.get(normalized_target)
            if target is not None:
                edges.add(DependencyEdge(source=source, target=target, scope=scope))

    return (
        tuple(sorted(edges, key=lambda edge: (edge.source, edge.target, edge.scope))),
        tuple(warnings),
    )


def _hotspots(root: Path, *, limit: int) -> tuple[SourceHotspot, ...]:
    files = tuple(
        path
        for target in PYTHON_TARGET_DIRS
        for path in _python_files(root / target)
        if not {"tests", "vendor"}.intersection(path.relative_to(root).parts)
    )
    ranked = sorted(
        (
            SourceHotspot(
                path=path.relative_to(root).as_posix(),
                lines=_line_count(path),
            )
            for path in files
        ),
        key=lambda item: (-item.lines, item.path),
    )
    return tuple(ranked[:limit])


def build_report(root: Path, *, hotspot_limit: int = 20) -> ArchitectureReport:
    """Build the deterministic repository architecture report."""
    member_paths = discover_member_paths(root)
    members = tuple(_member_stats(root, path) for path in member_paths)
    member_names = {
        normalize_requirement_name(member.name): member.name for member in members
    }
    dependency_edges, warnings = _dependency_edges(member_paths, member_names)
    return ArchitectureReport(
        members=members,
        dependency_edges=dependency_edges,
        hotspots=_hotspots(root, limit=hotspot_limit),
        warnings=warnings,
        import_audit=audit_imports(root, member_paths=member_paths),
    )


def report_payload(report: ArchitectureReport) -> JsonObject:
    """Return a JSON-serializable report payload."""
    return {
        "summary": {
            "workspace_members": len(report.members),
            "roots": report.root_count,
            "libraries": report.library_count,
            "models": report.model_count,
            "dependency_edges": len(report.dependency_edges),
        },
        "warnings": list(report.warnings),
        "members": [cast(JsonObject, asdict(member)) for member in report.members],
        "dependency_edges": [
            cast(JsonObject, asdict(edge)) for edge in report.dependency_edges
        ],
        "hotspots": [cast(JsonObject, asdict(hotspot)) for hotspot in report.hotspots],
        "import_audit": {
            "records": [
                {
                    "category": record.category,
                    "declared_scopes": list(record.declared_scopes),
                    "distribution": record.distribution,
                    "module": record.module,
                    "path": record.path,
                }
                for record in report.import_audit.records
            ],
            "undeclared_distributions": list(
                report.import_audit.undeclared_distributions
            ),
        },
    }


def render_text(report: ArchitectureReport) -> str:
    """Render a compact human-readable architecture report."""
    lines = [
        (
            f"Workspace: {len(report.members)} members "
            f"({report.root_count} root, {report.library_count} libraries, "
            f"{report.model_count} models)"
        ),
        "Member stats cover only src/ and tests/; hotspots scan Python files under "
        "lib/, models/, and scripts/ except paths containing tests or vendor.",
        f"Workspace dependency edges: {len(report.dependency_edges)}",
        f"Warnings: {len(report.warnings)}",
        "",
        "Members:",
    ]
    for member in report.members:
        metadata = ", ".join(
            value
            for value in (
                member.framework,
                member.task,
            )
            if value is not None
        )
        suffix = f" [{metadata}]" if metadata else ""
        lines.append(
            f"- {member.name} ({member.kind}): "
            f"src={member.source_files} files/{member.source_lines} lines, "
            f"tests={member.test_files} files/{member.test_lines} lines{suffix}"
        )

    lines.extend(("", "Workspace dependencies:"))
    lines.extend(
        f"- {edge.source} -> {edge.target} ({edge.scope})"
        for edge in report.dependency_edges
    )
    lines.extend(("", "Warnings:"))
    lines.extend(f"- {warning}" for warning in report.warnings)
    lines.extend(("", "Python hotspots:"))
    lines.extend(f"- {item.lines:>5} {item.path}" for item in report.hotspots)
    lines.extend(("", "Direct imports (scripts/*.py and tests/*.py):"))
    for record in report.import_audit.records:
        details = record.category
        if record.distribution is not None:
            details += f", {record.distribution}"
            if record.declared_scopes:
                details += f", declared in {', '.join(record.declared_scopes)}"
            else:
                details += ", undeclared"

        lines.append(f"- {record.path}: {record.module} ({details})")

    lines.extend(("", "Undeclared root-tooling distributions:"))
    lines.extend(
        f"- {distribution}"
        for distribution in report.import_audit.undeclared_distributions
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the architecture audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Report uv workspace dependencies, root-tooling imports, and Python "
            "source hotspots."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report output format",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of Python source hotspots to include",
    )
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be non-negative")

    report = build_report(args.root.resolve(), hotspot_limit=args.top)
    if args.format == "json":
        print(json.dumps(report_payload(report), indent=2, sort_keys=True))
    else:
        print(render_text(report), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
