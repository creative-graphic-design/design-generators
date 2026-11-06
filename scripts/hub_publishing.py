"""Shared Hugging Face Hub upload and read-back helpers."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def verify_source_revision(repository_root: Path, revision: str) -> None:
    """Require the publisher checkout to match the requested source revision."""
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    current_revision = result.stdout.strip()
    if current_revision != revision:
        raise RuntimeError(
            "The publisher checkout is not at the requested source revision: "
            f"expected {revision}, found {current_revision}."
        )


def verify_hub_write_access(api: HfApi, *, repo_id: str) -> None:
    """Fail before reading or uploading Hub content when the token lacks write access."""
    try:
        api.auth_check(repo_id=repo_id, repo_type="model", write=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not verify HF_TOKEN content-write access to {repo_id}. "
            "Add this repository to the Hugging Face fine-grained token with write permission."
        ) from error


def get_parent_commit(api: HfApi, *, repo_id: str) -> str:
    """Read the current Hub main revision to use as the upload parent."""
    parent_commit = api.model_info(repo_id, revision="main").sha
    if parent_commit is None:
        raise RuntimeError(f"Hub repository {repo_id} main revision has no commit SHA")

    return parent_commit


def write_upload_staging(
    staging_directory: Path,
    files: dict[str, bytes],
    *,
    allow_patterns: tuple[str, ...],
) -> None:
    """Write the exact allowlisted files into an empty upload directory."""
    allowed_files = set(allow_patterns)
    if set(files) != allowed_files:
        raise ValueError("Hub staging input must contain only the declared Hub files")

    if any(staging_directory.iterdir()):
        raise ValueError("Hub staging directory must be empty")

    for filename, contents in files.items():
        (staging_directory / filename).write_bytes(contents)

    staged_files = {path.name for path in staging_directory.iterdir()}
    if staged_files != allowed_files:
        raise RuntimeError(f"unexpected files in Hub staging directory: {staged_files}")


def verify_uploaded_files(
    api: HfApi,
    repo_id: str,
    files: dict[str, bytes],
    *,
    revision: str,
) -> None:
    """Read each uploaded file from the Hub and compare its exact bytes."""
    from huggingface_hub import hf_hub_download

    repository_files = set(
        api.list_repo_files(repo_id, repo_type="model", revision=revision)
    )
    missing_files = set(files) - repository_files
    if missing_files:
        raise RuntimeError(f"Hub publication is missing files: {sorted(missing_files)}")

    for filename, expected_data in files.items():
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            token=api.token,
        )
        downloaded_data = Path(downloaded_path).read_bytes()
        if downloaded_data != expected_data:
            raise RuntimeError(f"Hub read-back differs from the staged {filename}")


def upload_and_verify(
    api: HfApi,
    *,
    repo_id: str,
    files: dict[str, bytes],
    allow_patterns: tuple[str, ...],
    parent_commit: str,
    commit_message: str,
    staging_prefix: str,
) -> str:
    """Upload the exact allowlisted files on a known parent and verify their bytes."""
    with tempfile.TemporaryDirectory(prefix=staging_prefix) as temp_directory:
        staging_directory = Path(temp_directory)
        write_upload_staging(
            staging_directory,
            files,
            allow_patterns=allow_patterns,
        )
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging_directory),
            allow_patterns=list(allow_patterns),
            commit_message=commit_message,
            parent_commit=parent_commit,
        )

    published_revision = commit_info.oid
    verify_uploaded_files(
        api,
        repo_id,
        files,
        revision=published_revision,
    )
    return published_revision
