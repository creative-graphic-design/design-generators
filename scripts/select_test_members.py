#!/usr/bin/env python3
"""List or select workspace members whose tests should run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from typing import NotRequired, Sequence, TypedDict, cast


DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class WorkspaceSource(TypedDict, total=False):
    workspace: bool
    path: str


SourceSpec = WorkspaceSource | list[WorkspaceSource]
Project = TypedDict(
    "Project",
    {
        "name": NotRequired[str],
        "dependencies": NotRequired[list[str]],
        "optional-dependencies": NotRequired[dict[str, list[str]]],
    },
    total=False,
)


class Workspace(TypedDict):
    members: list[str]


class UvConfig(TypedDict, total=False):
    workspace: Workspace
    sources: dict[str, SourceSpec]


class ToolConfig(TypedDict, total=False):
    uv: UvConfig


Pyproject = TypedDict(
    "Pyproject",
    {
        "project": NotRequired[Project],
        "dependency-groups": NotRequired[dict[str, list[str]]],
        "tool": NotRequired[ToolConfig],
    },
    total=False,
)


@dataclass(frozen=True)
class Member:
    directory: str
    package: str
    dependencies: frozenset[str]
    has_tests: bool


def normalize_package_name(name: str) -> str:
    """Normalize a distribution name for dependency matching."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _workspace_source(
    source: SourceSpec,
    source_directory: Path,
    workspace_directories: set[Path],
) -> bool:
    if isinstance(source, list):
        return any(
            _workspace_source(item, source_directory, workspace_directories)
            for item in source
        )

    if not isinstance(source, dict):
        return False

    if source.get("workspace") is True:
        return True

    source_path = source.get("path")
    return (
        isinstance(source_path, str)
        and (source_directory / source_path).resolve() in workspace_directories
    )


def _requirements(
    project: Project, dependency_groups: dict[str, list[str]]
) -> list[str]:
    requirements = list(project.get("dependencies", []))
    for optional_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(optional_requirements)

    for group_requirements in dependency_groups.values():
        requirements.extend(group_requirements)

    return requirements


def _read_pyproject(path: Path) -> Pyproject:
    return cast(Pyproject, tomllib.loads(path.read_text()))


def discover_members(repository_root: Path) -> list[Member]:
    """Read workspace packages, their declared dependencies, and test directories."""
    root_project = _read_pyproject(repository_root / "pyproject.toml")
    workspace = root_project["tool"]["uv"]["workspace"]
    root_sources = root_project.get("tool", {}).get("uv", {}).get("sources", {})
    pyproject_files = sorted(
        {
            pyproject_file
            for pattern in workspace["members"]
            for pyproject_file in repository_root.glob(f"{pattern}/pyproject.toml")
        }
    )
    records = [
        (pyproject_file.parent, _read_pyproject(pyproject_file))
        for pyproject_file in pyproject_files
    ]
    package_keys = {
        normalize_package_name(project["project"]["name"]) for _, project in records
    }
    workspace_directories = {
        member_directory.resolve() for member_directory, _ in records
    }
    members: list[Member] = []

    for member_directory, project_data in records:
        project = project_data["project"]
        local_sources = project_data.get("tool", {}).get("uv", {}).get("sources", {})
        source_specs: dict[str, tuple[SourceSpec, Path]] = {
            normalize_package_name(name): (source, repository_root)
            for name, source in root_sources.items()
        }
        source_specs.update(
            {
                normalize_package_name(name): (source, member_directory)
                for name, source in local_sources.items()
            }
        )
        dependencies: set[str] = set()

        for requirement in _requirements(
            project, project_data.get("dependency-groups", {})
        ):
            match = DEPENDENCY_NAME_RE.match(requirement)
            if match is None:
                raise ValueError(f"invalid dependency requirement: {requirement!r}")

            dependency = normalize_package_name(match.group(1))
            source = source_specs.get(dependency)
            if dependency in package_keys and source is not None:
                source_value, source_directory = source
                if _workspace_source(
                    source_value, source_directory, workspace_directories
                ):
                    dependencies.add(dependency)

        members.append(
            Member(
                directory=member_directory.relative_to(repository_root).as_posix(),
                package=project["name"],
                dependencies=frozenset(dependencies),
                has_tests=(member_directory / "tests").is_dir(),
            )
        )

    return members


def select_members(
    repository_root: Path, changed_paths: Sequence[str] | None
) -> list[Member]:
    """Select test-bearing members from changed files and the workspace graph."""
    members = discover_members(repository_root)
    test_members = [member for member in members if member.has_tests]
    if changed_paths is None:
        return test_members

    non_markdown_paths = [path for path in changed_paths if not path.endswith(".md")]
    if not non_markdown_paths:
        return []

    package_keys = {normalize_package_name(member.package) for member in members}
    dependents: dict[str, set[str]] = {package: set() for package in package_keys}
    member_by_directory = {member.directory: member for member in members}
    changed_packages: set[str] = set()

    for path in non_markdown_paths:
        owner_directory = next(
            (
                directory
                for directory in member_by_directory
                if path == directory or path.startswith(f"{directory}/")
            ),
            None,
        )
        if owner_directory is None:
            return test_members

        changed_packages.add(
            normalize_package_name(member_by_directory[owner_directory].package)
        )

    for member in members:
        member_key = normalize_package_name(member.package)
        for dependency in member.dependencies:
            dependents[dependency].add(member_key)

    # shortcut: undeclared test-only imports are not modeled; declare the edge before relying on affected selection.
    # Root tooling tests inspect repository-wide files, so run them for every non-Markdown change.
    affected_packages = changed_packages | {"devharness"}
    pending = list(changed_packages)
    while pending:
        package = pending.pop()
        for dependent in dependents[package] - affected_packages:
            affected_packages.add(dependent)
            pending.append(dependent)

    return [
        member
        for member in test_members
        if normalize_package_name(member.package) in affected_packages
    ]


def _changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit for changed-file selection")
    parser.add_argument("--head", help="head commit for changed-file selection")
    args = parser.parse_args(argv)
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be provided together")

    changed_paths = None if args.base is None else _changed_paths(args.base, args.head)
    selected = select_members(REPOSITORY_ROOT, changed_paths)
    print(
        json.dumps(
            [
                {"dir": member.directory, "package": member.package}
                for member in selected
            ],
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
