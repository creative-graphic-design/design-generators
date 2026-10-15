"""Verify the pinned Colab environment and the installed NCCL RECORD entry."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path, PurePosixPath


NCCL_LIBRARY = PurePosixPath("nvidia/nccl/lib/libnccl.so.2")
NCCL_RECORD_ROOT = PurePosixPath("nvidia/nccl")


def normalize_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def verify_nvidia_nccl_distribution_names(names: Iterable[str]) -> None:
    nccl_distributions = sorted(
        {
            normalized
            for name in names
            if (normalized := normalize_distribution_name(name)).startswith(
                "nvidia-nccl-"
            )
        }
    )
    if nccl_distributions != ["nvidia-nccl-cu13"]:
        raise SystemExit(
            "expected only nvidia-nccl-cu13 in the helper environment; found "
            f"{nccl_distributions}"
        )


def verify_unique_nvidia_nccl_records(
    distributions: Iterable[tuple[str, Iterable[PurePosixPath]]],
) -> None:
    owners: dict[PurePosixPath, set[str]] = {}
    for distribution_name, files in distributions:
        for path in files:
            if path.is_relative_to(NCCL_RECORD_ROOT):
                owners.setdefault(path, set()).add(distribution_name)

    duplicates = {path: names for path, names in owners.items() if len(names) > 1}
    if duplicates:
        details = "; ".join(
            f"{path} owned by {', '.join(sorted(names))}"
            for path, names in sorted(duplicates.items())
        )
        raise SystemExit(f"overlapping nvidia/nccl RECORD ownership: {details}")


def verify_recorded_sha256(path: Path, mode: str | None, value: str | None) -> None:
    if mode != "sha256" or not value:
        raise SystemExit(f"NCCL RECORD has no SHA-256 for {NCCL_LIBRARY}")

    try:
        expected = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except ValueError as error:
        raise SystemExit("NCCL RECORD SHA-256 is malformed") from error
    if len(expected) != hashlib.sha256().digest_size:
        raise SystemExit("NCCL RECORD SHA-256 has an invalid length")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.digest()
    if actual != expected:
        raise SystemExit(
            "installed libnccl.so.2 SHA-256 differs from nvidia-nccl-cu13 RECORD"
        )


def verify_environment(repo_dir: Path, expected_environment: Path) -> None:
    repository_environment = repo_dir / ".venv"
    if repository_environment.exists() or repository_environment.is_symlink():
        raise SystemExit("repository .venv exists; refusing environment drift")

    configured_environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured_environment != str(expected_environment):
        raise SystemExit("UV_PROJECT_ENVIRONMENT differs from the helper environment")
    if Path(sys.prefix).resolve() != expected_environment.resolve():
        raise SystemExit("Python is running outside the helper environment")

    distributions = tuple(metadata.distributions())
    verify_nvidia_nccl_distribution_names(
        distribution.name for distribution in distributions
    )
    verify_unique_nvidia_nccl_records(
        (
            distribution.name,
            (PurePosixPath(str(entry)) for entry in distribution.files or ()),
        )
        for distribution in distributions
    )
    nccl_cu13 = next(
        distribution
        for distribution in distributions
        if normalize_distribution_name(distribution.name) == "nvidia-nccl-cu13"
    )

    record_entry = next(
        (
            entry
            for entry in nccl_cu13.files or ()
            if PurePosixPath(str(entry)) == NCCL_LIBRARY
        ),
        None,
    )
    if record_entry is None:
        raise SystemExit("nvidia-nccl-cu13 RECORD has no libnccl.so.2 entry")
    library_path = Path(record_entry.locate())
    if not library_path.is_file():
        raise SystemExit("nvidia-nccl-cu13 libnccl.so.2 is missing")
    recorded_hash = record_entry.hash
    verify_recorded_sha256(
        library_path,
        recorded_hash.mode if recorded_hash is not None else None,
        recorded_hash.value if recorded_hash is not None else None,
    )
    print("colab_environment_passed nccl_record=nvidia-nccl-cu13")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--expected-environment", required=True, type=Path)
    arguments = parser.parse_args()
    verify_environment(arguments.repo_dir, arguments.expected_environment)


if __name__ == "__main__":
    main()
