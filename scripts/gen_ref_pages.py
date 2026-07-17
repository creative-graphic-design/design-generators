"""Generate mkdocstrings API pages for workspace members."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
MEMBER_PARENTS = ("lib", "models")


@dataclass(frozen=True)
class ApiPage:
    """Generated API page metadata."""

    module: str
    page_path: Path
    source_path: Path


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


def page_path_for(module: str, source_path: Path) -> Path:
    """Return the generated markdown path for a module."""
    module_parts = module.split(".")
    if source_path.name == "__init__.py":
        return Path("api", *module_parts, "index.md")
    return Path("api", *module_parts[:-1], f"{module_parts[-1]}.md")


def iter_source_files(package_root: Path) -> list[Path]:
    """Return public package and module source files for API docs."""
    source_files = [package_root / "__init__.py"]
    source_files.extend(
        path
        for path in sorted(package_root.rglob("*.py"))
        if path.name != "__init__.py"
        and not path.stem.startswith("_")
        and not any(
            part.startswith("_") for part in path.relative_to(package_root).parts[:-1]
        )
    )
    return source_files


def discover_api_pages() -> list[ApiPage]:
    """Discover all mkdocstrings pages from current workspace members."""
    pages: list[ApiPage] = []
    for member_dir in iter_member_dirs():
        read_project_name(member_dir / "pyproject.toml")
        for package_root in iter_package_roots(member_dir):
            for source_path in iter_source_files(package_root):
                module = module_name(package_root, source_path)
                pages.append(
                    ApiPage(
                        module=module,
                        page_path=page_path_for(module, source_path),
                        source_path=source_path,
                    )
                )
    return sorted(pages, key=lambda page: page.module)


def write_api_index(pages: list[ApiPage]) -> None:
    """Write the API reference landing page."""
    with mkdocs_gen_files.open("api/index.md", "w") as nav_file:
        nav_file.write("# API Reference\n\n")
        if not pages:
            nav_file.write(
                "No workspace packages were discovered. Packages under `lib/*` "
                "and `models/*` will appear here automatically when they are added.\n"
            )
            return
        nav_file.write(
            "The pages in this section are generated from Python packages under "
            "`lib/*/src` and `models/*/src`.\n"
        )


def write_api_pages(pages: list[ApiPage]) -> None:
    """Write one mkdocstrings page per discovered module."""
    for page in pages:
        with mkdocs_gen_files.open(page.page_path, "w") as reference_file:
            reference_file.write(f"# `{page.module}`\n\n")
            reference_file.write(f"::: {page.module}\n")
        mkdocs_gen_files.set_edit_path(
            page.page_path,
            page.source_path.relative_to(ROOT),
        )


def write_literate_nav(pages: list[ApiPage]) -> None:
    """Write literate navigation for the generated API section."""
    with mkdocs_gen_files.open("api/SUMMARY.md", "w") as summary_file:
        summary_file.write("# API Reference\n\n")
        summary_file.write("- [Overview](index.md)\n")
        for page in pages:
            summary_file.write(
                f"- [`{page.module}`]({page.page_path.relative_to('api')})\n"
            )


def main() -> None:
    """Generate all API reference files."""
    pages = discover_api_pages()
    write_api_index(pages)
    write_api_pages(pages)
    write_literate_nav(pages)


main()
