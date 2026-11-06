"""Shared Hugging Face Hub upload and read-back helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Protocol


class HubModelInfo(Protocol):
    """Fields needed from a Hugging Face model repository revision."""

    sha: str | None


class HubCommitInfo(Protocol):
    """Fields needed from a Hugging Face upload commit."""

    oid: str


class HubApi(Protocol):
    """Hugging Face API operations used by the shared publisher."""

    def auth_check(
        self,
        *,
        repo_id: str,
        repo_type: Literal["model"],
        write: bool,
    ) -> None: ...

    def model_info(self, repo_id: str, *, revision: str) -> HubModelInfo: ...

    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: Literal["model"],
        revision: str,
    ) -> list[str]: ...


@dataclass(frozen=True)
class HubUploadRequest:
    """Exact Hub folder-upload inputs passed to the client adapter."""

    repo_id: str
    folder_path: str
    allow_patterns: tuple[str, ...]
    commit_message: str
    parent_commit: str


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


def verify_hub_write_access(api: HubApi, *, repo_id: str) -> None:
    """Fail before reading or uploading Hub content when the token lacks write access."""
    try:
        api.auth_check(repo_id=repo_id, repo_type="model", write=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not verify HF_TOKEN content-write access to {repo_id}. "
            "Add this repository to the Hugging Face fine-grained token with write permission."
        ) from error


def get_parent_commit(api: HubApi, *, repo_id: str) -> str:
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
    api: HubApi,
    repo_id: str,
    files: dict[str, bytes],
    *,
    revision: str,
    read_file: Callable[[str, str], bytes],
) -> None:
    """Read each uploaded file from the Hub and compare its exact bytes."""
    repository_files = set(
        api.list_repo_files(repo_id, repo_type="model", revision=revision)
    )
    missing_files = set(files) - repository_files
    if missing_files:
        raise RuntimeError(f"Hub publication is missing files: {sorted(missing_files)}")

    for filename, expected_data in files.items():
        downloaded_data = read_file(filename, revision)
        if downloaded_data != expected_data:
            raise RuntimeError(f"Hub read-back differs from the staged {filename}")


def upload_and_verify(
    api: HubApi,
    *,
    repo_id: str,
    files: dict[str, bytes],
    allow_patterns: tuple[str, ...],
    parent_commit: str,
    commit_message: str,
    staging_prefix: str,
    upload_folder: Callable[[HubUploadRequest], HubCommitInfo],
    read_file: Callable[[str, str], bytes],
) -> str:
    """Upload the exact allowlisted files on a known parent and verify their bytes."""
    with tempfile.TemporaryDirectory(prefix=staging_prefix) as temp_directory:
        staging_directory = Path(temp_directory)
        write_upload_staging(
            staging_directory,
            files,
            allow_patterns=allow_patterns,
        )
        commit_info = upload_folder(
            HubUploadRequest(
                repo_id=repo_id,
                folder_path=str(staging_directory),
                allow_patterns=allow_patterns,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
        )

    published_revision = commit_info.oid
    verify_uploaded_files(
        api,
        repo_id,
        files,
        revision=published_revision,
        read_file=read_file,
    )
    return published_revision
