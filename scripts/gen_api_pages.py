"""Write mkdocstrings pages and a literate-nav summary under ``docs/api``."""

from pathlib import Path
import shutil
import tomllib


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"lib": "libraries", "models": "models"}
EXCLUDED_MODULES = {"testing", "vendor_parity", "vendor_state", "vendor_state_dict"}
PAGE_HEADER = "---\nicon: lucide/file-code\ntags:\n  - API Reference\n---\n"


ModulePage = tuple[str, str, str, Path]


def discover_modules(root: Path = ROOT) -> list[ModulePage]:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pages: list[ModulePage] = []
    for pattern in data["tool"]["uv"]["workspace"]["members"]:
        for member in sorted(root.glob(pattern)):
            project_file = member / "pyproject.toml"
            if not project_file.is_file():
                continue

            project = str(
                tomllib.loads(project_file.read_text(encoding="utf-8"))["project"][
                    "name"
                ]
            ).replace("_", "-")
            group = GROUPS.get(member.relative_to(root).parts[0], member.parent.name)
            for source in sorted((member / "src").rglob("*.py")):
                parts = list(source.relative_to(member / "src").with_suffix("").parts)
                if (
                    not parts
                    or parts[-1] == "__main__"
                    or parts[-1] in EXCLUDED_MODULES
                    or any(
                        part.startswith("_") and part != "__init__" for part in parts
                    )
                ):
                    continue

                is_package = parts[-1] == "__init__"
                parts = parts[:-1] if is_package else parts
                tail = parts[1:]
                filename = "index.md" if is_package else f"{parts[-1]}.md"
                if not is_package:
                    tail = tail[:-1]

                pages.append(
                    (
                        ".".join(parts),
                        group,
                        project,
                        Path(group, project, *tail, filename),
                    )
                )

    return sorted(pages, key=lambda page: (page[1], page[2], page[0]))


def generate(root: Path = ROOT) -> list[ModulePage]:
    api_root = root / "docs" / "api"
    if api_root.is_dir():
        shutil.rmtree(api_root)

    pages = discover_modules(root)
    api_root.mkdir(parents=True)
    summary = ["* [Overview](index.md)"]
    seen: set[tuple[str, ...]] = set()
    for page in pages:
        target = api_root / page[3]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{PAGE_HEADER}\n::: {page[0]}\n", encoding="utf-8")

        module_parts = page[0].split(".")
        keys = (page[2], *module_parts[1:])
        for index, key in enumerate(keys):
            prefix = keys[: index + 1]
            if prefix in seen:
                continue

            seen.add(prefix)
            indent = "  " * index
            summary.append(
                f"{indent}- [{key}]({page[3]})"
                if index == len(keys) - 1
                else f"{indent}- {key}"
            )

    (api_root / "index.md").write_text(
        f"{PAGE_HEADER}\n# API Reference\n", encoding="utf-8"
    )
    (api_root / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return pages


if __name__ == "__main__":
    generate()
