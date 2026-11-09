"""Audit a Hub checkpoint using metadata and file bytes from one pinned revision."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Literal, TypedDict

import torch
from huggingface_hub import HfApi, RepoFile, hf_hub_download


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


def sha256_file(path: Path) -> str:
    """Hash a file without loading the full checkpoint into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_checkpoint(
    repo_id: str,
    checkpoint: str,
    revision: str,
    expected_sha256: str | None,
    minimum_step: int,
) -> AuditReport:
    """Compare Hub metadata, downloaded bytes, and checkpoint training state."""
    pinned_revision: str | None = None
    metadata_revision: str | None = None
    download_revision: str | None = None
    metadata_sha256: str | None = None
    checkpoint_sha256: str | None = None
    metadata_size: int | None = None
    checkpoint_size: int | None = None
    checkpoint_epoch: int | None = None
    global_step: int | None = None
    cuda_available: bool | None = None
    errors: list[str] = []

    try:
        api = HfApi(token=True)
        resolved = api.model_info(repo_id=repo_id, revision=revision, token=True)
        pinned_revision = resolved.sha
        if pinned_revision is None or not re.fullmatch(
            r"[0-9a-f]{40}", pinned_revision
        ):
            raise ValueError(
                "Hub did not resolve the requested revision to a commit SHA"
            )

        metadata_revision = pinned_revision
        entries = api.get_paths_info(
            repo_id=repo_id,
            paths=[checkpoint],
            repo_type="model",
            revision=pinned_revision,
            token=True,
        )
        entry = next(
            (
                item
                for item in entries
                if isinstance(item, RepoFile) and item.path == checkpoint
            ),
            None,
        )
        if entry is None:
            raise FileNotFoundError("checkpoint is missing from pinned Hub metadata")
        metadata_size = entry.size
        if entry.lfs is None:
            raise ValueError("checkpoint metadata does not include an LFS SHA-256")
        metadata_sha256 = entry.lfs.sha256

        download_revision = pinned_revision
        downloaded_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=checkpoint,
                repo_type="model",
                revision=pinned_revision,
                force_download=True,
                token=True,
            )
        )
        checkpoint_size = downloaded_path.stat().st_size
        checkpoint_sha256 = sha256_file(downloaded_path)
        checkpoint_data = torch.load(
            downloaded_path, map_location="cpu", weights_only=False
        )
        checkpoint_epoch = int(checkpoint_data["epoch"])
        global_step = int(checkpoint_data["global_step"])
        cuda_available = torch.cuda.is_available()

        if metadata_size != checkpoint_size:
            errors.append("downloaded checkpoint size differs from pinned metadata")
        if metadata_sha256 != checkpoint_sha256:
            errors.append("downloaded checkpoint SHA-256 differs from pinned metadata")
        if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
            errors.append("downloaded checkpoint SHA-256 differs from expected SHA-256")
        if global_step <= minimum_step:
            errors.append("checkpoint global_step did not exceed the previous step")
        if cuda_available:
            errors.append("checkpoint audit must run without CUDA")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    return {
        "checkpoint": checkpoint,
        "requested_revision": revision,
        "pinned_revision": pinned_revision,
        "metadata_revision": metadata_revision,
        "download_revision": download_revision,
        "metadata_sha256": metadata_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_size": metadata_size,
        "checkpoint_size": checkpoint_size,
        "checkpoint_epoch": checkpoint_epoch,
        "global_step": global_step,
        "expected_sha256": expected_sha256,
        "minimum_step_exclusive": minimum_step,
        "cuda_available": cuda_available,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if errors else "verified",
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write an audit record and return nonzero for any failed check."""
    parser = argparse.ArgumentParser(
        description="Audit a Hub checkpoint at one immutable revision."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--minimum-step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_sha256
    ):
        parser.error("--expected-sha256 must be a lowercase SHA-256 digest")
    if args.minimum_step < 0:
        parser.error("--minimum-step must be non-negative")

    report = audit_checkpoint(
        repo_id=args.repo,
        checkpoint=args.checkpoint,
        revision=args.revision,
        expected_sha256=args.expected_sha256,
        minimum_step=args.minimum_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
