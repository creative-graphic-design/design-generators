"""Repository metadata and install-command contracts."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from .constants import (
    GIT_REPO_URL,
    LIB_MEMBER_DIRS,
    MODEL_MEMBER_DIRS,
    REPO_ROOT,
    ROOT_REPO_BLOB_URL,
    PyprojectValue,
)


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        return ""

    rest = text[match.end() :]
    next_heading = re.search(r"\n## ", rest)
    return rest[: next_heading.start()] if next_heading else rest


def _bash_fences(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)


def _project_metadata(member_dir: Path) -> dict[str, PyprojectValue]:
    return tomllib.loads((member_dir / "pyproject.toml").read_text(encoding="utf-8"))


def _project_name(member_dir: Path) -> str:
    project = _project_metadata(member_dir)["project"]
    if not isinstance(project, dict):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: missing [project]")

    name = project.get("name")
    if not isinstance(name, str):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: project.name missing")

    return name


def _normalize_root_repo_link(link: str) -> str:
    return link.removeprefix(ROOT_REPO_BLOB_URL)


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()


def _dependency_direct_name(requirement: str) -> str:
    return re.split(r"[<>=!~;]", requirement, maxsplit=1)[0].strip()


def _project_dependencies(member_dir: Path) -> list[str]:
    project = _project_metadata(member_dir)["project"]
    if not isinstance(project, dict):
        raise AssertionError(f"{member_dir / 'pyproject.toml'}: missing [project]")

    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise AssertionError(
            f"{member_dir / 'pyproject.toml'}: project.dependencies must be a list"
        )

    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def _workspace_package_subdirs() -> dict[str, str]:
    subdirs: dict[str, str] = {}
    for member_dir in [*LIB_MEMBER_DIRS, *MODEL_MEMBER_DIRS]:
        subdirs[_project_name(member_dir)] = str(member_dir.relative_to(REPO_ROOT))

    return subdirs


def _workspace_source_names(member_dir: Path) -> set[str]:
    tool = _project_metadata(member_dir).get("tool", {})
    if not isinstance(tool, dict):
        return set()

    uv = tool.get("uv", {})
    if not isinstance(uv, dict):
        return set()

    sources = uv.get("sources", {})
    if not isinstance(sources, dict):
        return set()

    return {
        name
        for name, source in sources.items()
        if isinstance(name, str)
        and isinstance(source, dict)
        and source.get("workspace") is True
    }


def _direct_requirement(package_name: str, subdirectory: str) -> str:
    return f"{package_name} @ git+{GIT_REPO_URL}#subdirectory={subdirectory}"


def _model_install_requirements(member_dir: Path) -> list[tuple[str, str]]:
    workspace_dependencies: list[tuple[str, str]] = []
    workspace_package_subdirs = _workspace_package_subdirs()
    workspace_source_names = _workspace_source_names(member_dir)
    own_package = _project_name(member_dir)
    for requirement in _project_dependencies(member_dir):
        dependency = _dependency_name(requirement)
        if dependency == own_package or dependency not in workspace_source_names:
            continue

        subdirectory = workspace_package_subdirs.get(dependency)
        if subdirectory is None:
            raise AssertionError(
                f"{member_dir / 'pyproject.toml'}: workspace dependency "
                f"{dependency!r} is not a workspace member"
            )

        workspace_dependencies.append(
            (_dependency_direct_name(requirement), subdirectory)
        )

    slug = member_dir.name
    return [*workspace_dependencies, (own_package, f"models/{slug}")]


def _pip_install_snippet(requirements: list[tuple[str, str]]) -> str:
    direct_requirements = [
        _direct_requirement(package_name, subdirectory)
        for package_name, subdirectory in requirements
    ]
    if len(direct_requirements) == 1:
        return f'pip install "{direct_requirements[0]}"'

    lines = ["pip install \\"]
    for index, requirement in enumerate(direct_requirements):
        suffix = " \\" if index < len(direct_requirements) - 1 else ""
        lines.append(f'  "{requirement}"{suffix}')

    return "\n".join(lines)


def _assert_pip_install_snippet(
    path: Path,
    section: str,
    requirements: list[tuple[str, str]],
    section_label: str,
) -> None:
    expected_requirements = [
        _direct_requirement(package_name, subdirectory)
        for package_name, subdirectory in requirements
    ]
    for fence in _bash_fences(section):
        if "pip install" in fence and all(
            requirement in fence for requirement in expected_requirements
        ):
            return

    expected_example = f"```bash\n{_pip_install_snippet(requirements)}\n```"
    missing = [
        requirement
        for requirement in expected_requirements
        if requirement not in section
    ]
    raise AssertionError(
        f"{path}: {section_label} must include a pip install snippet with "
        f"the package direct URL and required shared package direct URLs. "
        f"Missing: {missing}. Expected example:\n{expected_example}"
    )


def _assert_model_pip_install_snippet(path: Path, text: str) -> None:
    section = _section(text, "## How to Get Started with the Model")
    _assert_pip_install_snippet(
        path,
        section,
        _model_install_requirements(path.parent),
        "How to Get Started",
    )


def _assert_library_pip_install_snippet(path: Path, text: str) -> None:
    section = _section(text, "## Install")
    member_dir = path.parent
    _assert_pip_install_snippet(
        path,
        section,
        [(_project_name(member_dir), f"lib/{member_dir.name}")],
        "Install",
    )
