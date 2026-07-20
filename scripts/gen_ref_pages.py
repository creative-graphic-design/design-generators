"""Generate mkdocstrings API pages for workspace members."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]
GENERATED_API_DIR = ROOT / "docs" / "api"
MEMBER_PARENTS = ("lib", "models")
GROUP_TITLES = {
    "lib": "Libraries",
    "models": "Models",
}
PUBLIC_MODEL_MODULE_PREFIXES = (
    "configuration",
    "modeling",
    "model_card",
    "pipeline",
    "processing",
    "processor",
    "scheduling",
    "scheduler",
    "tokenization",
    "tokenizer",
)


@dataclass(frozen=True)
class ApiPage:
    """Generated API page metadata."""

    module: str
    page_path: Path
    source_path: Path


@dataclass(frozen=True)
class ApiPackage:
    """Generated API package metadata."""

    group: str
    project_name: str
    package_name: str
    member_dir: Path
    package_root: Path
    index_path: Path
    pages: tuple[ApiPage, ...]


def write_text_file(path: Path, content: str) -> None:
    """Write generated Markdown content under ``docs/api``."""
    target = ROOT / "docs" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def iter_member_dirs() -> list[Path]:
    """Return workspace member directories that contain a pyproject file."""
    member_dirs: list[Path] = []
    for parent_name in MEMBER_PARENTS:
        parent = ROOT / parent_name
        if not parent.is_dir():
            continue
        member_dirs.extend(
            path.parent for path in sorted(parent.glob("*/pyproject.toml"))
        )
    return member_dirs


def read_project_name(pyproject_path: Path) -> str:
    """Read the project name from a workspace member pyproject file."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(data["project"]["name"])


def group_for_member(member_dir: Path) -> str:
    """Return the top-level API group for a workspace member."""
    return GROUP_TITLES[member_dir.parent.name]


def iter_package_roots(member_dir: Path) -> list[Path]:
    """Return import package roots under a member's src directory."""
    src_dir = member_dir / "src"
    if not src_dir.is_dir():
        return []
    return [
        path.parent
        for path in sorted(src_dir.glob("*/__init__.py"))
        if path.parent.is_dir()
    ]


def module_name(package_root: Path, source_path: Path) -> str:
    """Build a dotted module name for a package or module source file."""
    relative = source_path.relative_to(package_root.parent)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def package_slug(project_name: str) -> str:
    """Return a stable URL path component for a workspace package."""
    return project_name.replace("_", "-")


def page_path_for(
    group: str,
    project_name: str,
    package_name: str,
    module: str,
    source_path: Path,
) -> Path:
    """Return the generated markdown path for a module."""
    group_slug = group.lower()
    project_slug = package_slug(project_name)
    if module == package_name:
        return Path("api", group_slug, project_slug, "package.md")
    if module.startswith(f"{package_name}."):
        module_slug = module.removeprefix(f"{package_name}.").replace(".", "/")
    else:
        module_slug = module.replace(".", "/")
    if source_path.name == "__init__.py":
        return Path("api", group_slug, project_slug, module_slug, "index.md")
    return Path("api", group_slug, project_slug, f"{module_slug}.md")


def imported_public_modules(init_file: Path) -> set[str]:
    """Return sibling module stems imported by a package ``__init__`` file."""
    imported_modules: set[str] = set()
    module = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            imported_modules.add(node.module.split(".", maxsplit=1)[0])
    return imported_modules


def should_document_source(
    source_path: Path,
    package_root: Path,
    group: str,
    imported_modules: set[str],
) -> bool:
    """Return whether a source file belongs in the sidebar API reference."""
    if source_path.name == "__init__.py":
        return True
    relative = source_path.relative_to(package_root)
    if source_path.stem.startswith("_") or any(
        part.startswith("_") for part in relative.parts[:-1]
    ):
        return False
    if group == "Libraries":
        return True
    return len(relative.parts) == 1 and (
        source_path.stem in imported_modules
        or source_path.stem.startswith(PUBLIC_MODEL_MODULE_PREFIXES)
    )


def iter_source_files(
    package_root: Path,
    group: str,
    imported_modules: set[str],
) -> list[Path]:
    """Return public package and module source files for API docs."""
    source_files = [package_root / "__init__.py"]
    source_files.extend(
        path
        for path in sorted(package_root.rglob("*.py"))
        if path.name != "__init__.py"
        and should_document_source(path, package_root, group, imported_modules)
    )
    return source_files


def discover_api_packages() -> list[ApiPackage]:
    """Discover grouped mkdocstrings pages from current workspace members."""
    packages: list[ApiPackage] = []
    for member_dir in iter_member_dirs():
        group = group_for_member(member_dir)
        project_name = read_project_name(member_dir / "pyproject.toml")
        for package_root in iter_package_roots(member_dir):
            package_name = package_root.name
            imported_modules = imported_public_modules(package_root / "__init__.py")
            pages: list[ApiPage] = []
            for source_path in iter_source_files(package_root, group, imported_modules):
                module = module_name(package_root, source_path)
                pages.append(
                    ApiPage(
                        module=module,
                        page_path=page_path_for(
                            group,
                            project_name,
                            package_name,
                            module,
                            source_path,
                        ),
                        source_path=source_path,
                    )
                )
            packages.append(
                ApiPackage(
                    group=group,
                    project_name=project_name,
                    package_name=package_name,
                    member_dir=member_dir,
                    package_root=package_root,
                    index_path=Path(
                        "api",
                        group.lower(),
                        package_slug(project_name),
                        "index.md",
                    ),
                    pages=tuple(sorted(pages, key=lambda page: page.module)),
                )
            )
    return sorted(packages, key=lambda package: (package.group, package.project_name))


def write_api_index(packages: list[ApiPackage]) -> None:
    """Write the API reference landing page."""
    lines = ["# API Reference", ""]
    if not packages:
        lines.extend(
            [
                "No workspace packages were discovered. Packages under `lib/*` "
                "and `models/*` will appear here automatically when they are added.",
                "",
            ]
        )
        write_text_file(Path("api/index.md"), "\n".join(lines))
        return
    lines.extend(
        [
            "The pages in this section are generated from Python packages under "
            "`lib/*/src` and `models/*/src`.",
        ]
    )
    for group in GROUP_TITLES.values():
        group_packages = [package for package in packages if package.group == group]
        if not group_packages:
            continue
        lines.extend(["", f"## {group}", ""])
        for package in group_packages:
            lines.append(
                f"- [{package.project_name}]({package.index_path.relative_to('api')})"
            )
    lines.append("")
    write_text_file(Path("api/index.md"), "\n".join(lines))


def write_package_indexes(packages: list[ApiPackage]) -> None:
    """Write package landing pages with README content and module links."""
    for package in packages:
        readme_path = package.member_dir / "README.md"
        lines = []
        if readme_path.is_file():
            lines.extend([readme_path.read_text(encoding="utf-8").rstrip(), ""])
        else:
            lines.extend([f"# {package.project_name}", ""])
        lines.extend(["## API Modules", ""])
        for page in package.pages:
            relative_page_path = page.page_path.relative_to(package.index_path.parent)
            lines.append(f"- [`{page.module}`]({relative_page_path})")
        lines.append("")
        write_text_file(package.index_path, "\n".join(lines))


def write_api_pages(packages: list[ApiPackage]) -> None:
    """Write one mkdocstrings page per discovered module."""
    for package in packages:
        for page in package.pages:
            write_text_file(
                page.page_path,
                f"# `{page.module}`\n\n::: {page.module}\n",
            )


def clean_generated_api_dir() -> None:
    """Remove stale generated API pages before writing the current tree."""
    if GENERATED_API_DIR.is_dir():
        shutil.rmtree(GENERATED_API_DIR)


def main() -> None:
    """Generate all API reference files."""
    clean_generated_api_dir()
    packages = discover_api_packages()
    write_api_index(packages)
    write_package_indexes(packages)
    write_api_pages(packages)


if __name__ == "__main__":
    main()
