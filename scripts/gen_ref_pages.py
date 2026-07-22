"""Generate mkdocstrings API pages for workspace members."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tomllib
from typing import cast
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
GENERATED_API_DIR = ROOT / "docs" / "api"
GENERATED_MKDOCS_CONFIG = ROOT / "mkdocs.generated.yml"
GITHUB_BLOB_BASE_URL = (
    "https://github.com/creative-graphic-design/design-generators/blob/main"
)
MEMBER_PARENTS = ("lib", "models")
GROUP_TITLES = {
    "lib": "Libraries",
    "models": "Models",
}
PUBLIC_MODEL_MODULE_PREFIXES = (
    "conversion",
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
OVERVIEW_FRONTMATTER = """---
icon: lucide/layout-template
tags:
  - Overview
  - Documentation
---
"""
DESIGN_METADATA_TOOL_KEY = "design-generators"
DESIGN_METADATA_REQUIRED_KEYS = ("framework", "task", "conditions", "datasets")
DESIGN_METADATA_EXAMPLE = (
    '[tool.design-generators] framework = "transformers" '
    'task = "content-agnostic-layout-generation" '
    'conditions = ["unconditional"] datasets = ["rico25"]'
)
MODEL_OVERVIEW_BADGE_COLORS = {
    "framework": "blue",
    "task": "purple",
    "conditions": "green",
    "datasets": "orange",
}
MODEL_OVERVIEW_FRAMEWORK_LOGOS = {
    "diffusers": "huggingface",
    "pydantic-ai": "pydantic",
    "transformers": "huggingface",
}
MODEL_OVERVIEW_HF_DATASETS = frozenset(
    {
        "crello",
        "magazine",
        "publaynet",
        "rico13",
        "rico25",
    }
)
FRAMEWORK_TAGS = frozenset({"transformers", "diffusers", "pydantic-ai"})
TASK_TAGS = frozenset(
    {
        "content-agnostic-layout-generation",
        "content-aware-layout-generation",
    }
)


@dataclass(frozen=True)
class ApiPage:
    """Generated API page metadata."""

    module: str
    page_path: Path
    source_path: Path


@dataclass(frozen=True)
class ModelDesignMetadata:
    """Documentation tag metadata declared by a model workspace member."""

    framework: str
    tasks: tuple[str, ...]
    conditions: tuple[str, ...]
    datasets: tuple[str, ...]

    @property
    def tags(self) -> tuple[str, ...]:
        """Return the flat tag list consumed by the docs site."""
        return (
            self.framework,
            *self.tasks,
            *self.conditions,
            *self.datasets,
        )


@dataclass(frozen=True)
class ApiPackage:
    """Generated API package metadata."""

    group: str
    project_name: str
    display_name: str
    package_name: str
    member_dir: Path
    package_root: Path
    index_path: Path
    reproducing_path: Path | None
    pages: tuple[ApiPage, ...]
    design_metadata: ModelDesignMetadata | None


def write_text_file(path: Path, content: str) -> None:
    """Write generated Markdown content under ``docs/api``."""
    target = ROOT / "docs" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def write_generated_file(path: Path, content: str) -> None:
    """Write generated content outside the tracked documentation tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def docs_route(path: Path) -> str:
    """Return a documentation URL route for a generated markdown page."""
    if path.name == "index.md":
        return f"{path.parent.as_posix()}/"
    if path.suffix == ".md":
        return f"{path.with_suffix('').as_posix()}/"
    return path.as_posix()


def relative_docs_route(target: Path, *, source: Path) -> str:
    """Return a relative documentation route between generated pages."""
    relative = target.relative_to(source.parent)
    return docs_route(relative)


def site_page_for_repo_link(link: str) -> str:
    """Return the documentation-site target for a repository-relative link."""
    target = link.removeprefix("./")
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return link
    if target.startswith("models/") and target.endswith("/README.md"):
        parts = target.split("/")
        if len(parts) == 3:
            project_name = read_project_name(
                ROOT / "models" / parts[1] / "pyproject.toml"
            )
            return docs_route(
                Path("api", "models", package_slug(project_name), "index.md")
            )
    if target.startswith("models/") and target.endswith("/REPRODUCING.md"):
        parts = target.split("/")
        if len(parts) == 3:
            member_dir = ROOT / "models" / parts[1]
            project_name = read_project_name(member_dir / "pyproject.toml")
            reproducing_path = reproducing_page_path_for(
                "Models",
                project_name,
                member_dir,
            )
            if reproducing_path is not None:
                return docs_route(reproducing_path)
    if target.startswith("lib/") and target.endswith("/README.md"):
        parts = target.split("/")
        if len(parts) == 3:
            project_name = read_project_name(ROOT / "lib" / parts[1] / "pyproject.toml")
            return docs_route(
                Path("api", "libraries", package_slug(project_name), "index.md")
            )
    return f"{GITHUB_BLOB_BASE_URL}/{target}"


def rewrite_repo_relative_links(markdown: str) -> str:
    """Rewrite README repository links for the generated documentation site."""

    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        link = match.group("link")
        return f"[{label}]({site_page_for_repo_link(link)})"

    return re.sub(
        r"(?<!!)\[(?P<label>[^\]]+)\]\((?P<link>[^):#][^)]+)\)",
        replace,
        markdown,
    )


def write_overview_page() -> None:
    """Generate the documentation Overview page from the repository README."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").rstrip()
    write_text_file(
        Path("index.md"),
        f"{OVERVIEW_FRONTMATTER}\n{rewrite_repo_relative_links(readme)}\n",
    )


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


def read_pyproject(member_dir: Path) -> dict[str, object]:
    """Read a workspace member ``pyproject.toml`` file."""
    return tomllib.loads((member_dir / "pyproject.toml").read_text(encoding="utf-8"))


def read_str_enum_values(source_path: Path, class_name: str) -> frozenset[str]:
    """Return string values declared by a ``StrEnum`` source class."""
    module = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        values: set[str] = set()
        for item in node.body:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(item.value, ast.Constant) and isinstance(
                item.value.value, str
            ):
                values.add(item.value.value)
            elif isinstance(item.value, ast.Call) and isinstance(
                item.value.func, ast.Name
            ):
                if item.value.func.id == "auto":
                    values.add(target.id)
        return frozenset(values)
    raise ValueError(f"Could not find enum class {class_name} in {source_path}")


def allowed_condition_tags() -> frozenset[str]:
    """Return canonical condition tags from the shared condition enum."""
    return read_str_enum_values(
        ROOT / "lib" / "laygen" / "src" / "laygen" / "common" / "conditions.py",
        "ConditionType",
    )


def allowed_dataset_tags() -> frozenset[str]:
    """Return canonical dataset tags from shared layout and poster enums."""
    return read_str_enum_values(
        ROOT / "lib" / "laygen" / "src" / "laygen" / "common" / "labels.py",
        "DatasetName",
    ) | read_str_enum_values(
        ROOT / "lib" / "posgen" / "src" / "posgen" / "common" / "labels.py",
        "DatasetName",
    )


def model_metadata_error(member_dir: Path, key: str, detail: str) -> str:
    """Return an actionable model metadata validation error."""
    required = ", ".join(DESIGN_METADATA_REQUIRED_KEYS)
    return (
        f"{member_dir.relative_to(ROOT)} [tool.{DESIGN_METADATA_TOOL_KEY}] "
        f"{key}: {detail}. Required keys: {required}. Example: {DESIGN_METADATA_EXAMPLE}"
    )


def require_str_sequence(
    value: object, *, key: str, member_dir: Path
) -> tuple[str, ...]:
    """Return a tuple of strings from a model metadata list."""
    if not isinstance(value, list):
        raise TypeError(
            model_metadata_error(member_dir, key, "must be a non-empty list of strings")
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(
                model_metadata_error(
                    member_dir, key, "must be a non-empty list of strings"
                )
            )
        items.append(item.strip())
    if not items or any(not item for item in items):
        raise ValueError(
            model_metadata_error(member_dir, key, "must be a non-empty list of strings")
        )
    return tuple(items)


def require_str_or_sequence(
    value: object, *, key: str, member_dir: Path
) -> tuple[str, ...]:
    """Return a tuple of strings from a scalar or list metadata field."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(model_metadata_error(member_dir, key, "must not be empty"))
        return (value.strip(),)
    return require_str_sequence(value, key=key, member_dir=member_dir)


def validate_values(
    values: tuple[str, ...],
    *,
    allowed: frozenset[str],
    key: str,
    member_dir: Path,
) -> None:
    """Validate model metadata values against a closed vocabulary."""
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            model_metadata_error(member_dir, key, f"has unknown values: {unknown}")
        )


def required_metadata_value(
    table: Mapping[str, object], *, key: str, member_dir: Path
) -> object:
    """Return a required metadata value or fail with an actionable error."""
    if key not in table:
        raise KeyError(model_metadata_error(member_dir, key, "is required"))
    return table[key]


def read_model_design_metadata(member_dir: Path) -> ModelDesignMetadata:
    """Read and validate model docs tag metadata from ``pyproject.toml``."""
    data = read_pyproject(member_dir)
    tool = data.get("tool", {})
    if not isinstance(tool, dict):
        tool = {}
    table = tool.get(DESIGN_METADATA_TOOL_KEY)
    if not isinstance(table, dict):
        raise KeyError(model_metadata_error(member_dir, "table", "is required"))
    metadata_table = cast(Mapping[str, object], table)
    framework = required_metadata_value(
        metadata_table, key="framework", member_dir=member_dir
    )
    tasks = require_str_or_sequence(
        required_metadata_value(metadata_table, key="task", member_dir=member_dir),
        key="task",
        member_dir=member_dir,
    )
    if not isinstance(framework, str) or not framework.strip():
        raise TypeError(
            model_metadata_error(member_dir, "framework", "must be a non-empty string")
        )
    framework = framework.strip()
    if framework not in FRAMEWORK_TAGS:
        raise ValueError(
            model_metadata_error(member_dir, "framework", f"is unknown: {framework}")
        )
    validate_values(tasks, allowed=TASK_TAGS, key="task", member_dir=member_dir)
    conditions = require_str_sequence(
        required_metadata_value(
            metadata_table, key="conditions", member_dir=member_dir
        ),
        key="conditions",
        member_dir=member_dir,
    )
    datasets = require_str_sequence(
        required_metadata_value(metadata_table, key="datasets", member_dir=member_dir),
        key="datasets",
        member_dir=member_dir,
    )
    validate_values(
        conditions,
        allowed=allowed_condition_tags(),
        key="conditions",
        member_dir=member_dir,
    )
    validate_values(
        datasets,
        allowed=allowed_dataset_tags(),
        key="datasets",
        member_dir=member_dir,
    )
    return ModelDesignMetadata(
        framework=framework,
        tasks=tasks,
        conditions=conditions,
        datasets=datasets,
    )


def read_model_display_name(member_dir: Path, project_name: str) -> str:
    """Read the human-facing model name from a model README."""
    readme_path = member_dir / "README.md"
    if not readme_path.is_file():
        return project_name
    readme = readme_path.read_text(encoding="utf-8")
    frontmatter_match = re.match(r"---\n(?P<frontmatter>.*?)\n---\n", readme, re.S)
    if frontmatter_match is not None:
        name_match = re.search(
            r"(?m)^\s*-\s+name:\s+[\"']?(?P<name>[^\"'\n]+)[\"']?\s*$",
            frontmatter_match.group("frontmatter"),
        )
        if name_match is not None:
            return name_match.group("name")
    heading_match = re.search(r"(?m)^# Model Card for (?P<name>.+?)\s*$", readme)
    if heading_match is not None:
        return heading_match.group("name")
    return project_name


def display_name_for_member(group: str, member_dir: Path, project_name: str) -> str:
    """Return the human-facing label for docs navigation."""
    if group == "Models":
        return read_model_display_name(member_dir, project_name)
    return project_name


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


def reproducing_page_path_for(
    group: str,
    project_name: str,
    member_dir: Path,
) -> Path | None:
    """Return the generated reproducing guide path for a workspace member."""
    reproducing_file = member_dir / "REPRODUCING.md"
    if group == "Models" and not reproducing_file.is_file():
        msg = (
            f"Model package {member_dir.relative_to(ROOT)} must include REPRODUCING.md"
        )
        raise FileNotFoundError(msg)
    if not reproducing_file.is_file():
        return None
    return Path("api", group.lower(), package_slug(project_name), "reproducing.md")


def imported_public_modules(init_file: Path) -> set[str]:
    """Return sibling module stems imported by a package ``__init__`` file."""
    imported_modules: set[str] = set()
    package_name = init_file.parent.name
    module = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                imported_modules.add(node.module.split(".", maxsplit=1)[0])
            elif node.level == 0 and node.module:
                parts = node.module.split(".")
                if len(parts) > 1 and parts[0] == package_name:
                    imported_modules.add(parts[1])
                elif node.module == package_name:
                    imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) > 1 and parts[0] == package_name:
                    imported_modules.add(parts[1])
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
        display_name = display_name_for_member(group, member_dir, project_name)
        design_metadata = (
            read_model_design_metadata(member_dir) if group == "Models" else None
        )
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
                    display_name=display_name,
                    package_name=package_name,
                    member_dir=member_dir,
                    package_root=package_root,
                    index_path=Path(
                        "api",
                        group.lower(),
                        package_slug(project_name),
                        "index.md",
                    ),
                    reproducing_path=reproducing_page_path_for(
                        group,
                        project_name,
                        member_dir,
                    ),
                    pages=tuple(sorted(pages, key=lambda page: page.module)),
                    design_metadata=design_metadata,
                )
            )
    return sorted(packages, key=lambda package: (package.group, package.display_name))


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
                f"- [{package.display_name}]({relative_docs_route(package.index_path, source=Path('api/index.md'))})"
            )
    lines.append("")
    write_text_file(Path("api/index.md"), "\n".join(lines))


def write_group_indexes(packages: list[ApiPackage]) -> None:
    """Write API group landing pages for explicit site navigation."""
    for group in GROUP_TITLES.values():
        group_packages = [package for package in packages if package.group == group]
        if not group_packages:
            continue
        lines = [
            f"# {group}",
            "",
            f"{group} generated from workspace members.",
            "",
        ]
        for package in group_packages:
            lines.append(
                f"- [{package.display_name}]({relative_docs_route(package.index_path, source=Path('api', group.lower(), 'index.md'))})"
            )
        lines.append("")
        write_text_file(Path("api", group.lower(), "index.md"), "\n".join(lines))


def strip_markdown_frontmatter(markdown: str) -> str:
    """Remove YAML front matter from embedded README content."""
    frontmatter_match = re.match(r"---\n.*?\n---\n", markdown, re.S)
    if frontmatter_match is None:
        return markdown
    return markdown[frontmatter_match.end() :].lstrip("\n")


def render_tags_frontmatter(tags: tuple[str, ...]) -> list[str]:
    """Render YAML front matter for docs page tags."""
    if not tags:
        return []
    lines = ["---", "tags:"]
    lines.extend(f"  - {tag}" for tag in tags)
    lines.extend(["---", ""])
    return lines


def model_overview_badge_logo(value: str, *, axis: str) -> str | None:
    """Return a Simple Icons slug for a model overview badge when applicable."""
    if axis == "framework":
        return MODEL_OVERVIEW_FRAMEWORK_LOGOS.get(value)
    if axis == "datasets" and value in MODEL_OVERVIEW_HF_DATASETS:
        return "huggingface"
    return None


def render_model_overview_badge(value: str, *, axis: str) -> str:
    """Render a shields.io static badge for the model overview table."""
    color = MODEL_OVERVIEW_BADGE_COLORS[axis]
    params = [
        ("label", axis.removesuffix("s")),
        ("message", value),
        ("color", color),
        ("style", "flat-square"),
    ]
    logo = model_overview_badge_logo(value, axis=axis)
    if logo is not None:
        params.extend([("logo", logo), ("logoColor", "white")])
    url = f"https://img.shields.io/static/v1?{urlencode(params)}"
    return f"![{axis.removesuffix('s')}: {value}]({url})"


def render_model_overview_badges(values: tuple[str, ...], *, axis: str) -> str:
    """Render badges for one metadata axis in the model overview table."""
    return " ".join(render_model_overview_badge(value, axis=axis) for value in values)


def write_package_indexes(packages: list[ApiPackage]) -> None:
    """Write package landing pages with README content and module links."""
    for package in packages:
        readme_path = package.member_dir / "README.md"
        metadata = package.design_metadata
        lines = render_tags_frontmatter(metadata.tags if metadata is not None else ())
        if readme_path.is_file():
            readme = strip_markdown_frontmatter(
                readme_path.read_text(encoding="utf-8")
            ).rstrip()
            lines.extend([readme, ""])
        else:
            lines.extend([f"# {package.display_name}", ""])
        if package.reproducing_path is not None:
            lines.extend(
                [
                    "**Reproducing parity:** "
                    "[Open the model reproducing guide]"
                    f"({relative_docs_route(package.reproducing_path, source=package.index_path)}).",
                    "",
                ]
            )
        lines.extend(["## API Modules", ""])
        for page in package.pages:
            relative_page_path = relative_docs_route(
                page.page_path, source=package.index_path
            )
            lines.append(f"- [`{page.module}`]({relative_page_path})")
        lines.append("")
        write_text_file(package.index_path, "\n".join(lines))


def write_models_overview(packages: list[ApiPackage]) -> None:
    """Write the generated model comparison table."""
    model_packages = [package for package in packages if package.group == "Models"]
    lines = [
        "---",
        "tags:",
        "  - Models",
        "  - Documentation",
        "---",
        "",
        "# Models",
        "",
        "This generated table summarizes model package metadata declared in each "
        "`models/*/pyproject.toml`.",
        "",
        "| Package | Framework | Task | Conditions | Datasets |",
        "| --- | --- | --- | --- | --- |",
    ]
    for package in model_packages:
        metadata = package.design_metadata
        if metadata is None:
            raise ValueError(f"Missing model metadata for {package.project_name}")
        package_link = f"[{package.display_name}]({docs_route(package.index_path)})"
        lines.append(
            " | ".join(
                [
                    f"| {package_link}",
                    render_model_overview_badge(metadata.framework, axis="framework"),
                    render_model_overview_badges(metadata.tasks, axis="task"),
                    render_model_overview_badges(
                        metadata.conditions, axis="conditions"
                    ),
                    render_model_overview_badges(metadata.datasets, axis="datasets")
                    + " |",
                ]
            )
        )
    lines.append("")
    write_text_file(Path("models.md"), "\n".join(lines))


def render_generated_nav(packages: list[ApiPackage]) -> list[str]:
    """Render explicit MkDocs nav lines with model-level API entries."""
    lines = [
        "nav:",
        "  - Overview: index.md",
        "  - Getting Started: getting-started.md",
        "  - Models: models.md",
        "  - Conventions: conventions.md",
        "  - API Reference:",
        "      - Overview: api/index.md",
    ]
    for group in GROUP_TITLES.values():
        group_packages = [package for package in packages if package.group == group]
        if not group_packages:
            continue
        lines.extend(
            [
                f"      - {group}:",
                f"          - Overview: api/{group.lower()}/index.md",
            ]
        )
        for package in group_packages:
            lines.append(f"          - {package.display_name}: {package.index_path}")
    return lines


def write_generated_mkdocs_config(packages: list[ApiPackage]) -> None:
    """Write a generated config whose sidebar lists API pages by package."""
    source = (ROOT / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
    nav_start = next(index for index, line in enumerate(source) if line == "nav:")
    nav_end = nav_start + 1
    while nav_end < len(source):
        line = source[nav_end]
        if line and not line.startswith((" ", "-")):
            break
        nav_end += 1
    generated = [
        *source[:nav_start],
        *render_generated_nav(packages),
        *source[nav_end:],
    ]
    write_generated_file(GENERATED_MKDOCS_CONFIG, "\n".join(generated) + "\n")


def write_reproducing_pages(packages: list[ApiPackage]) -> None:
    """Write package reproducing guides when the member ships one."""
    for package in packages:
        if package.reproducing_path is None:
            continue
        reproducing_path = package.member_dir / "REPRODUCING.md"
        write_text_file(
            package.reproducing_path,
            f"{reproducing_path.read_text(encoding='utf-8').rstrip()}\n",
        )


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
    write_overview_page()
    write_api_index(packages)
    write_group_indexes(packages)
    write_package_indexes(packages)
    write_models_overview(packages)
    write_reproducing_pages(packages)
    write_api_pages(packages)
    write_generated_mkdocs_config(packages)


if __name__ == "__main__":
    main()
