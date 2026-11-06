#!/usr/bin/env python3
"""Publish the Collage custom pipeline and model card to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import hub_publishing  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = (
    PACKAGE_ROOT / "src" / "collage_diffusion" / "pipeline_collage_diffusion.py"
)
README_SOURCE = PACKAGE_ROOT / "README.md"
HUB_REPOSITORY_ID = "creative-graphic-design/collage-diffusion"
UPLOAD_ALLOW_PATTERNS = ("README.md", "pipeline.py")
SOURCE_ASSET_PREFIX = "source-assets/"
PLANNED_HUB_BADGE = (
    "![hub](https://img.shields.io/static/v1?label=hub&message=planned-code-only"
    "&color=orange&style=flat-square&logo=huggingface&logoColor=white)"
)
PUBLISHED_HUB_BADGE = (
    "![hub](https://img.shields.io/static/v1?label=hub&message=published-code-only"
    "&color=brightgreen&style=flat-square&logo=huggingface&logoColor=white)"
)
ARCHIVE_URL_MARKER = "/archive/main.zip"
ARCHIVE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def prepare_upload_files(
    pipeline_source: bytes,
    readme_text: str,
    revision: str,
) -> dict[str, bytes]:
    """Build the exact Hub files from the canonical pipeline and package README."""
    if not ARCHIVE_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")

    if not readme_text.startswith("---\n"):
        raise ValueError("package README must begin with YAML frontmatter")

    frontmatter_end = readme_text.find("\n---\n", 4)
    if frontmatter_end < 0:
        raise ValueError("package README has no closing YAML frontmatter delimiter")

    frontmatter = readme_text[4:frontmatter_end]
    metadata_lines = frontmatter.splitlines()
    license_line_indexes = [
        index
        for index, line in enumerate(metadata_lines)
        if line.startswith("license:")
    ]
    if len(license_line_indexes) != 1:
        raise ValueError("package README must have exactly one frontmatter license")

    del metadata_lines[license_line_indexes[0]]
    card_text = "---\n" + "\n".join(metadata_lines) + "\n---\n"
    card_text += readme_text[frontmatter_end + 5 :]

    if card_text.count(ARCHIVE_URL_MARKER) != 2:
        raise ValueError("package README must document both archive installs once")

    card_text = card_text.replace(ARCHIVE_URL_MARKER, f"/archive/{revision}.zip")
    card_text = card_text.replace(PLANNED_HUB_BADGE, PUBLISHED_HUB_BADGE, 1)
    return {"README.md": card_text.encode("utf-8"), "pipeline.py": pipeline_source}


def _read_source_files(revision: str) -> dict[str, bytes]:
    return prepare_upload_files(
        PIPELINE_SOURCE.read_bytes(),
        README_SOURCE.read_text(encoding="utf-8"),
        revision,
    )


def _source_asset_files(api: HfApi, *, revision: str) -> set[str]:
    return {
        path
        for path in api.list_repo_files(
            HUB_REPOSITORY_ID,
            repo_type="model",
            revision=revision,
        )
        if path.startswith(SOURCE_ASSET_PREFIX)
    }


def publish(*, token: str, revision: str) -> None:
    """Upload the two declared files and verify them without touching other paths."""
    from huggingface_hub import HfApi

    files = _read_source_files(revision)
    api = HfApi(token=token)
    hub_publishing.verify_hub_write_access(api, repo_id=HUB_REPOSITORY_ID)
    parent_commit = hub_publishing.get_parent_commit(api, repo_id=HUB_REPOSITORY_ID)

    source_assets_before = _source_asset_files(api, revision=parent_commit)
    if not source_assets_before:
        raise RuntimeError("the existing Hub repository has no source-assets files")

    def read_hub_file(filename: str, published_revision: str) -> bytes:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HUB_REPOSITORY_ID,
            filename=filename,
            repo_type="model",
            revision=published_revision,
            token=token,
        )
        return Path(downloaded_path).read_bytes()

    published_revision = hub_publishing.upload_and_verify(
        api,
        repo_id=HUB_REPOSITORY_ID,
        files=files,
        allow_patterns=UPLOAD_ALLOW_PATTERNS,
        parent_commit=parent_commit,
        commit_message=f"Publish Collage Diffusion pipeline at {revision[:12]}",
        staging_prefix="collage-hub-publish-",
        read_file=read_hub_file,
    )
    source_assets_after = _source_asset_files(api, revision=published_revision)
    if source_assets_after != source_assets_before:
        raise RuntimeError(
            "Hub publication changed the existing source-assets file set"
        )

    print(
        f"Published and verified {', '.join(UPLOAD_ALLOW_PATTERNS)} at "
        f"{published_revision}; preserved {len(source_assets_after)} source-assets files."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=os.environ.get("GITHUB_SHA"),
        help="full commit SHA used to pin Hub code and package archives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and list the exact upload files without contacting the Hub",
    )
    args = parser.parse_args(argv)
    if not args.revision:
        parser.error("--revision or GITHUB_SHA is required")
    if not ARCHIVE_REVISION_PATTERN.fullmatch(args.revision):
        parser.error("--revision must be a full lowercase 40-character commit SHA")

    hub_publishing.verify_source_revision(REPOSITORY_ROOT, args.revision)

    if args.dry_run:
        files = _read_source_files(args.revision)
        for filename, contents in sorted(files.items()):
            print(f"{filename}: {len(contents)} bytes")
        print(f"Hub target: {HUB_REPOSITORY_ID}")
        print("Hub write: skipped (--dry-run)")
        return 0

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        parser.error("HF_TOKEN is required for Hub publication")

    publish(token=token, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
