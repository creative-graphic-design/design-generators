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
    for root_name in ("lib", "models"):
        for pyproject in sorted((REPO_ROOT / root_name).glob("*/pyproject.toml")):
            metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            datasets = (
                metadata.get("tool", {})
                .get("design-generators", {})
                .get("datasets", [])
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
        fields[0].strip().strip("`"): fields[2].strip()
        for line in registry.group(1).splitlines()
        if line.startswith("|")
        and not line.startswith("| ---")
        and not line.startswith("| Metadata key")
        for fields in [line.strip("|").split("|")]
    }
    documented_dataset_keys = set(documented_datasets)
    missing = sorted(metadata_datasets - documented_dataset_keys)
    extra = sorted(documented_dataset_keys - metadata_datasets)
    assert not missing and not extra, (
        "docs/data-sources.md dataset registry differs from [tool.design-generators].datasets: "
        f"missing={missing}; extra={extra}"
    )
    invalid_ids = sorted(
        dataset
        for dataset, dataset_id in documented_datasets.items()
        if dataset_id != "unverified"
        and not re.fullmatch(
            r"\[[^]]+\]\(https://huggingface\.co/datasets/[^)]+\)", dataset_id
        )
    )
    assert not invalid_ids, (
        f"docs/data-sources.md has non-Hugging Face dataset IDs for: {invalid_ids}"
    )


def test_roadmap_links_implemented_packages_and_issue_targets() -> None:
    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("|")
        and not line.startswith("| ---")
        and not line.startswith("| Target")
    ]
    packages: set[str] = set()
    for row in rows:
        fields = [field.strip() for field in row.strip("|").split("|")]
        package = fields[2]
        issue = fields[4]
        if package != "—":
            match = re.fullmatch(
                r"\[:octicons-package-16:models/([^]]+)\]\(api/models/([^`]+)\.md\)",
                package,
            )
            assert match is not None, f"invalid roadmap package link: {package}"
            directory, target = match.groups()
            assert directory == target, f"package/link mismatch: {package}"
            assert (REPO_ROOT / "docs/api/models" / f"{target}.md").is_file(), (
                f"missing API target for {package}"
            )
            packages.add(directory)
        if issue != "—":
            match = re.fullmatch(
                r"\[:octicons-issue-opened-16:issue #(\d+)\]\(https://github\.com/creative-graphic-design/design-generators/issues/(\d+)\)",
                issue,
            )
            assert match is not None, f"invalid roadmap issue link: {issue}"
            assert match.group(1) == match.group(2), (
                f"issue text/link mismatch: {issue}"
            )

    model_directories = {
        path.name for path in (REPO_ROOT / "models").iterdir() if path.is_dir()
    }
    assert packages == model_directories, (
        "roadmap package registry differs from models/: "
        f"missing={sorted(model_directories - packages)}; "
        f"extra={sorted(packages - model_directories)}"
    )

    for page in (REPO_ROOT / "docs/roadmap.md", REPO_ROOT / "docs/data-sources.md"):
        raw_urls = re.findall(r"(?<!\]\()https?://[^)\s|]+", page.read_text())
        assert not raw_urls, f"{page}: raw URLs outside Markdown links: {raw_urls}"
