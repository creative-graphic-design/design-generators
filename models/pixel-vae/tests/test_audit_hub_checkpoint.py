from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from types import FunctionType, SimpleNamespace
from typing import Callable, Literal, TypedDict, cast

import pytest


class AuditReport(TypedDict):
    checkpoint: str
    requested_revision: str
    pinned_revision: str | None
    metadata_revision: str | None
    download_revision: str | None
    metadata_sha256: str | None
    checkpoint_sha256: str | None
    metadata_size: int | None
    checkpoint_size: int | None
    checkpoint_epoch: int | None
    global_step: int | None
    expected_sha256: str | None
    minimum_step_exclusive: int
    cuda_available: bool | None
    verified_at_utc: str
    status: Literal["verified", "failed"]
    errors: list[str]


AuditMain = Callable[[list[str] | None], int]


def test_audit_pins_metadata_and_download_and_records_sha_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "audit_hub_checkpoint.py"
    namespace = runpy.run_path(str(script_path))
    main_function = cast(FunctionType, namespace["main"])
    main = cast(AuditMain, main_function)
    pinned_revision = "a" * 40
    checkpoint = "checkpoints/last.ckpt"
    content = b"tiny checkpoint bytes"
    digest = hashlib.sha256(content).hexdigest()
    metadata_digest = "0" * 64
    downloaded = tmp_path / "last.ckpt"
    downloaded.write_bytes(content)
    revisions: list[str] = []

    class FakeRepoFile:
        def __init__(self, *, path: str, size: int, lfs: SimpleNamespace) -> None:
            self.path = path
            self.size = size
            self.lfs = lfs

    class FakeApi:
        def __init__(self, token: bool) -> None:
            assert token is True

        def model_info(
            self, *, repo_id: str, revision: str, token: bool
        ) -> SimpleNamespace:
            assert repo_id == "owner/repo"
            assert revision == "main"
            assert token is True
            return SimpleNamespace(sha=pinned_revision)

        def get_paths_info(
            self,
            *,
            repo_id: str,
            paths: list[str],
            repo_type: str,
            revision: str,
            token: bool,
        ) -> list[FakeRepoFile]:
            assert repo_id == "owner/repo"
            assert paths == [checkpoint]
            assert repo_type == "model"
            assert token is True
            revisions.append(revision)
            return [
                FakeRepoFile(
                    path=checkpoint,
                    size=len(content),
                    lfs=SimpleNamespace(sha256=metadata_digest),
                )
            ]

    def download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        force_download: bool,
        token: bool,
    ) -> str:
        assert repo_id == "owner/repo"
        assert filename == checkpoint
        assert repo_type == "model"
        assert force_download is True
        assert token is True
        revisions.append(revision)
        return str(downloaded)

    monkeypatch.setitem(main_function.__globals__, "HfApi", FakeApi)
    monkeypatch.setitem(main_function.__globals__, "RepoFile", FakeRepoFile)
    monkeypatch.setitem(main_function.__globals__, "hf_hub_download", download)
    monkeypatch.setitem(
        main_function.__globals__,
        "torch",
        SimpleNamespace(load=lambda *_args, **_kwargs: {"epoch": 4, "global_step": 14}),
    )

    output = tmp_path / "audit.json"
    exit_code = main(
        [
            "--repo",
            "owner/repo",
            "--checkpoint",
            checkpoint,
            "--revision",
            "main",
            "--expected-sha256",
            "0" * 64,
            "--minimum-step",
            "13",
            "--output",
            str(output),
        ]
    )
    report = cast(AuditReport, json.loads(output.read_text()))

    assert revisions == [pinned_revision, pinned_revision]
    assert report["metadata_revision"] == report["download_revision"] == pinned_revision
    assert report["status"] == "failed"
    assert report["global_step"] == 14
    assert exit_code == 1
    assert report["checkpoint_sha256"] == digest
    assert report["metadata_sha256"] == metadata_digest
    assert (
        "downloaded checkpoint SHA-256 differs from pinned metadata" in report["errors"]
    )
    assert (
        "downloaded checkpoint SHA-256 differs from expected SHA-256"
        in report["errors"]
    )
