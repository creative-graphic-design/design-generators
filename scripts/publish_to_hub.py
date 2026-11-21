"""Publish one model package using its declarative Hub configuration."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import hub_publishing  # noqa: E402

ARCHIVE_URL_MARKER = "/archive/main.zip"
ARCHIVE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PINNED_ARCHIVE_URL_PATTERN = re.compile(rb"/archive/[0-9a-f]{40}\.zip")


FileContent = TypedDict("FileContent", {"content": str})


PublisherConfig = TypedDict(
    "PublisherConfig",
    {
        "repo-id": str,
        "archive-url-count": int,
        "files": dict[str, str | FileContent],
        "remove-readme-license": NotRequired[bool],
        "preserve-existing-path-prefixes": NotRequired[list[str]],
    },
)


def load_config(package_directory: Path) -> PublisherConfig:
    config = tomllib.loads((package_directory / "pyproject.toml").read_text())
    return config["tool"]["design-generators"]["hub-publisher"]


def prepare_upload_files(
    package_directory: Path,
    config: PublisherConfig,
    revision: str,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for destination, spec in config["files"].items():
        if destination == "README.md":
            files[destination] = prepare_readme(
                (package_directory / spec).read_text(encoding="utf-8"), config, revision
            )
        elif isinstance(spec, str):
            files[destination] = (package_directory / spec).read_bytes()
        else:
            files[destination] = spec["content"].encode("utf-8")

    return files


def prepare_readme(readme_text: str, config: PublisherConfig, revision: str) -> bytes:
    if config.get("remove-readme-license", False):
        if not readme_text.startswith("---\n"):
            raise ValueError("package README must begin with YAML frontmatter")
        frontmatter, delimiter, body = readme_text[4:].partition("\n---\n")
        if not delimiter:
            raise ValueError("package README has no closing YAML frontmatter delimiter")
        metadata_lines = frontmatter.splitlines()
        license_indexes = [
            i for i, line in enumerate(metadata_lines) if line.startswith("license:")
        ]
        if len(license_indexes) != 1:
            raise ValueError("package README must have exactly one frontmatter license")
        del metadata_lines[license_indexes[0]]
        readme_text = "---\n" + "\n".join(metadata_lines) + "\n---\n" + body

    if readme_text.count(ARCHIVE_URL_MARKER) != config["archive-url-count"]:
        raise ValueError(
            "package README archive URL count does not match its configuration"
        )

    return readme_text.replace(
        ARCHIVE_URL_MARKER,
        f"/archive/{revision}.zip",
    ).encode("utf-8")


def files_match_published_revision(
    repository_files: set[str],
    files: dict[str, bytes],
    read_file: Callable[[str], bytes],
) -> bool:
    return set(files).issubset(repository_files) and all(
        _comparison_bytes(name, read_file(name)) == _comparison_bytes(name, contents)
        for name, contents in files.items()
    )


def _comparison_bytes(filename: str, contents: bytes) -> bytes:
    if filename == "README.md":
        return PINNED_ARCHIVE_URL_PATTERN.sub(b"/archive/{revision}.zip", contents)

    return contents


def publish(
    *,
    config: PublisherConfig,
    files: dict[str, bytes],
    token: str,
    revision: str,
) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = config["repo-id"]
    api = HfApi(token=token)
    hub_publishing.verify_hub_write_access(api, repo_id=repo_id)
    parent_commit = hub_publishing.get_parent_commit(api, repo_id=repo_id)
    repository_files = set(
        api.list_repo_files(repo_id, repo_type="model", revision=parent_commit)
    )
    prefixes = config.get("preserve-existing-path-prefixes", [])
    preserved_files = {
        prefix: {name for name in repository_files if name.startswith(prefix)}
        for prefix in prefixes
    }
    if any(not names for names in preserved_files.values()):
        raise RuntimeError("a configured Hub path prefix has no existing files")

    def read_hub_file(filename: str, file_revision: str) -> bytes:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=file_revision,
            token=token,
        )
        return Path(path).read_bytes()

    if files_match_published_revision(
        repository_files,
        files,
        lambda name: read_hub_file(name, parent_commit),
    ):
        print(f"Hub files already match {parent_commit}; upload skipped.")
        return

    def upload_folder(
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
        repo_id=repo_id,
        files=files,
        allow_patterns=tuple(files),
        parent_commit=parent_commit,
        commit_message=f"Publish {repo_id} at {revision[:12]}",
        staging_prefix="hub-publish-",
        upload_folder=upload_folder,
        read_file=read_hub_file,
    )
    updated_files = set(
        api.list_repo_files(repo_id, repo_type="model", revision=published_revision)
    )
    for prefix, expected in preserved_files.items():
        actual = {name for name in updated_files if name.startswith(prefix)}
        if actual != expected:
            raise RuntimeError(
                f"Hub publication changed the existing {prefix} file set"
            )
    print(f"Published and verified {', '.join(files)} at {published_revision}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-directory", type=Path, required=True)
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.revision:
        parser.error("--revision or GITHUB_SHA is required")
    if not ARCHIVE_REVISION_PATTERN.fullmatch(args.revision):
        parser.error("--revision must be a full lowercase 40-character commit SHA")
    package_directory = REPOSITORY_ROOT / args.package_directory

    hub_publishing.verify_source_revision(REPOSITORY_ROOT, args.revision)
    config = load_config(package_directory)
    files = prepare_upload_files(package_directory, config, args.revision)
    if args.dry_run:
        for filename, contents in sorted(files.items()):
            print(f"{filename}: {len(contents)} bytes")
        print(f"Hub target: {config['repo-id']}")
        print("Hub write: skipped (--dry-run)")
        return 0

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        parser.error("HF_TOKEN is required for Hub publication")
    publish(
        config=config,
        files=files,
        token=token,
        revision=args.revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
