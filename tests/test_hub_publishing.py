from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest

HUB_PUBLISHING_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "hub_publishing.py"
)
LOADER = SourceFileLoader("design_generators_hub_publishing", str(HUB_PUBLISHING_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
hub_publishing = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = hub_publishing
LOADER.exec_module(hub_publishing)


REPOSITORY_ID = "creative-graphic-design/structure-diffusion"
ALLOW_PATTERNS = ("README.md", "pipeline.py")
PARENT_COMMIT = "b" * 40
PUBLISHED_COMMIT = "c" * 40
FILES = {"README.md": b"card", "pipeline.py": b"pipeline"}


def test_source_revision_requires_the_checkout_to_match_the_requested_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_revision = "a" * 40
    command_arguments: list[str] = []

    def run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        command_arguments.extend(arguments)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        return SimpleNamespace(stdout=f"{expected_revision}\n")

    monkeypatch.setattr(hub_publishing.subprocess, "run", run)

    hub_publishing.verify_source_revision(tmp_path, expected_revision)

    assert command_arguments == [
        "git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "HEAD",
    ]


def test_source_revision_rejects_a_stale_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        hub_publishing.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{'b' * 40}\n"),
    )

    with pytest.raises(RuntimeError, match="not at the requested source revision"):
        hub_publishing.verify_source_revision(tmp_path, "a" * 40)


def test_staging_writes_exactly_the_allowlisted_files(tmp_path: Path) -> None:
    hub_publishing.write_upload_staging(
        tmp_path,
        FILES,
        allow_patterns=ALLOW_PATTERNS,
    )

    assert {path.name for path in tmp_path.iterdir()} == set(ALLOW_PATTERNS)
    assert (tmp_path / "README.md").read_bytes() == b"card"
    assert (tmp_path / "pipeline.py").read_bytes() == b"pipeline"


def test_staging_rejects_files_outside_the_allowlist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the declared Hub files"):
        hub_publishing.write_upload_staging(
            tmp_path,
            {**FILES, "source-assets/data.bin": b"asset"},
            allow_patterns=ALLOW_PATTERNS,
        )


def test_staging_rejects_a_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        hub_publishing.write_upload_staging(
            tmp_path,
            FILES,
            allow_patterns=ALLOW_PATTERNS,
        )


def test_write_access_failure_names_the_target_repository() -> None:
    class UnscopedTokenApi:
        def auth_check(self, **kwargs: str | bool) -> None:
            assert kwargs == {
                "repo_id": REPOSITORY_ID,
                "repo_type": "model",
                "write": True,
            }
            raise RuntimeError("repository is outside token scope")

    with pytest.raises(RuntimeError, match=REPOSITORY_ID):
        hub_publishing.verify_hub_write_access(
            UnscopedTokenApi(),
            repo_id=REPOSITORY_ID,
        )


def test_parent_commit_requires_a_hub_main_sha() -> None:
    class MissingMainCommitApi:
        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == REPOSITORY_ID
            assert revision == "main"
            return SimpleNamespace(sha=None)

    with pytest.raises(RuntimeError, match="main revision has no commit SHA"):
        hub_publishing.get_parent_commit(
            MissingMainCommitApi(),
            repo_id=REPOSITORY_ID,
        )


def test_upload_uses_parent_commit_allowlist_and_reads_back_bytes() -> None:
    upload_arguments: dict[str, object] = {}
    uploaded_files: dict[str, bytes] = {}
    readback_files: list[str] = []

    class ScopedTokenApi:
        token = "fine-grained-token"

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            upload_arguments.update(kwargs)
            staging_directory = Path(str(kwargs["folder_path"]))
            uploaded_files.update(
                {path.name: path.read_bytes() for path in staging_directory.iterdir()}
            )
            return SimpleNamespace(oid=PUBLISHED_COMMIT)

        def list_repo_files(
            self,
            repo_id: str,
            *,
            repo_type: str,
            revision: str,
        ) -> list[str]:
            assert repo_id == REPOSITORY_ID
            assert repo_type == "model"
            assert revision == PUBLISHED_COMMIT
            return list(uploaded_files)

    api = ScopedTokenApi()

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

    def read_hub_file(filename: str, revision: str) -> bytes:
        assert revision == PUBLISHED_COMMIT
        readback_files.append(filename)
        return uploaded_files[filename]

    published_commit = hub_publishing.upload_and_verify(
        api,
        repo_id=REPOSITORY_ID,
        files=FILES,
        allow_patterns=ALLOW_PATTERNS,
        parent_commit=PARENT_COMMIT,
        commit_message="Publish Structure Diffusion pipeline",
        staging_prefix="structure-diffusion-hub-publish-",
        upload_folder=upload_staging_directory,
        read_file=read_hub_file,
    )

    assert published_commit == PUBLISHED_COMMIT
    assert upload_arguments["repo_id"] == REPOSITORY_ID
    assert upload_arguments["repo_type"] == "model"
    assert upload_arguments["folder_path"]
    assert upload_arguments["commit_message"] == "Publish Structure Diffusion pipeline"
    assert upload_arguments["parent_commit"] == PARENT_COMMIT
    assert upload_arguments["allow_patterns"] == list(ALLOW_PATTERNS)
    assert upload_arguments["run_as_future"] is False
    assert uploaded_files == FILES
    assert readback_files == list(FILES)


def test_uploaded_file_readback_rejects_different_bytes() -> None:
    class UploadedFilesApi:
        def list_repo_files(
            self,
            repo_id: str,
            *,
            repo_type: str,
            revision: str,
        ) -> list[str]:
            assert repo_id == REPOSITORY_ID
            assert repo_type == "model"
            assert revision == PUBLISHED_COMMIT
            return list(FILES)

    with pytest.raises(
        RuntimeError, match="read-back differs from the staged README.md"
    ):
        hub_publishing.verify_uploaded_files(
            UploadedFilesApi(),
            REPOSITORY_ID,
            FILES,
            revision=PUBLISHED_COMMIT,
            read_file=lambda filename, revision: b"different bytes",
        )
