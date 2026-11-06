#!/usr/bin/env python3
"""Publish the Structure Diffusion custom pipeline and model card to the Hub."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import hub_publishing  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = (
    PACKAGE_ROOT / "src" / "structure_diffusion" / "pipeline_structure_diffusion.py"
)
MODELING_SOURCE = (
    PACKAGE_ROOT / "src" / "structure_diffusion" / "modeling_structure_diffusion.py"
)
PROCESSING_SOURCE = (
    PACKAGE_ROOT / "src" / "structure_diffusion" / "processing_structure_diffusion.py"
)
README_SOURCE = PACKAGE_ROOT / "README.md"
HUB_REPOSITORY_ID = "creative-graphic-design/structure-diffusion"
UPLOAD_ALLOW_PATTERNS = (
    "README.md",
    "pipeline.py",
    "modeling_structure_diffusion.py",
    "processing_structure_diffusion.py",
)
ARCHIVE_URL_MARKER = "/archive/main.zip"
ARCHIVE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def prepare_upload_files(
    pipeline_source: bytes,
    modeling_source: bytes,
    processing_source: bytes,
    readme_text: str,
    revision: str,
) -> dict[str, bytes]:
    """Build the exact Hub files from canonical package sources."""
    if not ARCHIVE_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")

    if readme_text.count(ARCHIVE_URL_MARKER) != 1:
        raise ValueError(
            "package README must document its archive install exactly once"
        )

    card_text = readme_text.replace(ARCHIVE_URL_MARKER, f"/archive/{revision}.zip")
    return {
        "README.md": card_text.encode("utf-8"),
        "pipeline.py": pipeline_source,
        "modeling_structure_diffusion.py": modeling_source,
        "processing_structure_diffusion.py": processing_source,
    }


def _read_source_files(revision: str) -> dict[str, bytes]:
    return prepare_upload_files(
        PIPELINE_SOURCE.read_bytes(),
        MODELING_SOURCE.read_bytes(),
        PROCESSING_SOURCE.read_bytes(),
        README_SOURCE.read_text(encoding="utf-8"),
        revision,
    )


def publish(*, token: str, revision: str) -> None:
    """Upload the four declared files and verify their exact bytes on the Hub."""
    from huggingface_hub import HfApi

    files = _read_source_files(revision)
    api = HfApi(token=token)
    hub_publishing.verify_hub_write_access(api, repo_id=HUB_REPOSITORY_ID)
    parent_commit = hub_publishing.get_parent_commit(api, repo_id=HUB_REPOSITORY_ID)
    published_revision = hub_publishing.upload_and_verify(
        api,
        repo_id=HUB_REPOSITORY_ID,
        files=files,
        allow_patterns=UPLOAD_ALLOW_PATTERNS,
        parent_commit=parent_commit,
        commit_message=f"Publish Structure Diffusion pipeline at {revision[:12]}",
        staging_prefix="structure-diffusion-hub-publish-",
    )

    print(
        f"Published and verified {', '.join(UPLOAD_ALLOW_PATTERNS)} at "
        f"{published_revision}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=os.environ.get("GITHUB_SHA"),
        help="full commit SHA used to pin the Hub card's package archive",
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
