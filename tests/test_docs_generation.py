"""Tests for documentation-page frontmatter and API coverage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
import re
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.S)


def _nav_files(
    value: object, parents: tuple[str, ...] = ()
) -> Iterator[tuple[str, tuple[str, ...]]]:
    if isinstance(value, str):
        yield value, parents
    elif isinstance(value, list):
        for child in value:
            yield from _nav_files(child, parents)
    elif isinstance(value, dict):
        for title, child in value.items():
            if not isinstance(title, str):
                continue
            yield from _nav_files(child, (*parents, title))


def test_docs_markdown_pages_have_icon_and_tags_frontmatter() -> None:
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(REPO_ROOT / "docs")
        if "plans" in relative.parts:
            continue

        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        assert match is not None, f"{path}: missing YAML frontmatter"
        body = match.group("body")
        assert re.search(r"(?m)^icon:\s+lucide/", body), path
        assert re.search(r"(?m)^tags:\n(?:  - .+\n?)+", body), path


def test_api_stubs_match_workspace_members_and_nav() -> None:
    workspace = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["uv"]["workspace"]["members"]
    members = {}
    for pattern in workspace:
        for member in REPO_ROOT.glob(pattern):
            if (member / "pyproject.toml").is_file():
                group = "libraries" if member.parts[-2] == "lib" else "models"
                members[(group, member.name)] = tomllib.loads(
                    (member / "pyproject.toml").read_text(encoding="utf-8")
                )["project"]["name"].replace("-", "_")
    nav = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))["nav"]
    api_nav = next(item["API Reference"] for item in nav if "API Reference" in item)
    nav_files = list(_nav_files(api_nav, ("API Reference",)))
    counts = Counter(path for path, _ in nav_files)
    for path, _ in nav_files:
        assert (REPO_ROOT / "docs" / path).is_file(), (
            f"{Path(path).stem}: missing nav target (nav)"
        )
    for stub in (REPO_ROOT / "docs/api").rglob("*.md"):
        if stub.name == "index.md":
            continue
        assert counts[stub.relative_to(REPO_ROOT / "docs").as_posix()] == 1, (
            f"{stub.stem}: orphan nav entry (orphan)"
        )
        group = stub.parent.name
        assert (group, stub.stem) in members, f"{stub.stem}: orphan API stub (orphan)"
    for (group, slug), import_name in members.items():
        stub = REPO_ROOT / "docs/api" / group / f"{slug}.md"
        assert stub.is_file(), f"{slug}: missing API stub (stub)"
        assert f"::: {import_name}" in stub.read_text(encoding="utf-8"), (
            f"{slug}: missing directive (directive)"
        )
        expected = f"api/{group}/{slug}.md"
        matches = [parents for path, parents in nav_files if path == expected]
        assert len(matches) == 1, f"{slug}: missing or duplicate nav entry (nav)"
        assert matches[0][-2] == group.title(), f"{slug}: wrong API nav group (nav)"


def test_data_source_registry_matches_package_metadata() -> None:
    metadata_datasets: set[str] = set()
    for pyproject in sorted((REPO_ROOT / "models").glob("*/pyproject.toml")):
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        datasets = (
            metadata.get("tool", {}).get("design-generators", {}).get("datasets", [])
        )
        assert isinstance(datasets, list), f"{pyproject}: datasets must be a list"
        metadata_datasets.update(
            dataset for dataset in datasets if isinstance(dataset, str)
        )

    text = (REPO_ROOT / "docs" / "data-sources.md").read_text(encoding="utf-8")
    registry = re.search(r"(?ms)^## Dataset registry\s*\n(.*?)(?=^## |\Z)", text)
    assert registry is not None, (
        "docs/data-sources.md: missing Dataset registry section"
    )
    documented_datasets = {
        fields[0].strip().strip("`")
        for line in registry.group(1).splitlines()
        if line.startswith("|")
        and not line.startswith("| ---")
        and not line.startswith("| Metadata key")
        for fields in [line.strip("|").split("|")]
    }
    missing = sorted(metadata_datasets - documented_datasets)
    extra = sorted(documented_datasets - metadata_datasets)
    assert not missing and not extra, (
        "docs/data-sources.md dataset registry differs from [tool.design-generators].datasets: "
        f"missing={missing}; extra={extra}"
    )
