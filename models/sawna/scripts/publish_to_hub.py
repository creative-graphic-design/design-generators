#!/usr/bin/env python3
"""Publish the code-only SAWNA pipeline and model card to the Hub."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import hub_publishing  # noqa: E402

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
README_SOURCE: Final[Path] = PACKAGE_ROOT / "README.md"
HUB_REPOSITORY_ID: Final[str] = "creative-graphic-design/sawna"
UPLOAD_ALLOW_PATTERNS: Final[tuple[str, ...]] = ("README.md", "pipeline.py")
ARCHIVE_URL_MARKER: Final[str] = "/archive/main.zip"
ARCHIVE_REVISION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
HUB_PIPELINE_SOURCE: Final[bytes] = b"from sawna import SAWNAHubPipeline\n"


def prepare_upload_files(readme_text: str, revision: str) -> dict[str, bytes]:
    """Build the exact Hub files from package sources and the README."""
    if not ARCHIVE_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")

    if readme_text.count(ARCHIVE_URL_MARKER) != 2:
        raise ValueError("package README must pin both workspace archive installs")

    card_text = readme_text.replace(ARCHIVE_URL_MARKER, f"/archive/{revision}.zip")
    return {"README.md": card_text.encode("utf-8"), "pipeline.py": HUB_PIPELINE_SOURCE}


def _read_source_files(revision: str) -> dict[str, bytes]:
    return prepare_upload_files(README_SOURCE.read_text(encoding="utf-8"), revision)


def publish(*, token: str, revision: str) -> None:
    """Upload the declared files and verify their exact bytes on the Hub."""
    from huggingface_hub import HfApi, hf_hub_download

    files = _read_source_files(revision)
    api = HfApi(token=token)
    hub_publishing.verify_hub_write_access(api, repo_id=HUB_REPOSITORY_ID)
    parent_commit = hub_publishing.get_parent_commit(api, repo_id=HUB_REPOSITORY_ID)

    def read_hub_file(filename: str, published_revision: str) -> bytes:
        downloaded_path = hf_hub_download(
            repo_id=HUB_REPOSITORY_ID,
            filename=filename,
            repo_type="model",
            revision=published_revision,
            token=token,
        )
        return Path(downloaded_path).read_bytes()

    def upload_staging_directory(
        request: hub_publishing.HubUploadRequest,
    ) -> hub_publishing.HubCommitInfo:
        return api.upload_folder(
            repo_id=request.repo_id,
            repo_type="model",
            folder_path=request.folder_path,
            allow_patterns=list(request.allow_patterns),
            commit_message=request.commit_message,
            parent_commit=request.parent_commit,
            run_as_future=False,
        )

    published_revision = hub_publishing.upload_and_verify(
        api,
        repo_id=HUB_REPOSITORY_ID,
        files=files,
        allow_patterns=UPLOAD_ALLOW_PATTERNS,
        parent_commit=parent_commit,
        commit_message=f"Publish SAWNA pipeline at {revision[:12]}",
        staging_prefix="sawna-hub-publish-",
        upload_folder=upload_staging_directory,
        read_file=read_hub_file,
    )
    print(
        f"Published and verified {', '.join(UPLOAD_ALLOW_PATTERNS)} at {published_revision}."
    )


def main(argv: list[str] | None = None) -> int:
    """Validate the source revision, then publish or preview the Hub files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=os.environ.get("GITHUB_SHA"),
        help="full commit SHA used to pin both package archives in the Hub card",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the exact upload files without contacting the Hub",
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
