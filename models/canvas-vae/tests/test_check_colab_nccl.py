import base64
import hashlib
import importlib.util
from collections.abc import Iterable
from pathlib import Path
from pathlib import PurePosixPath

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_colab_nccl.py"
SPEC = importlib.util.spec_from_file_location("check_colab_nccl", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_nccl_record_sha256_accepts_matching_library(tmp_path: Path) -> None:
    library = tmp_path / "libnccl.so.2"
    content = b"NCCL test fixture"
    library.write_bytes(content)
    record_hash = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    encoded_hash = record_hash.decode().rstrip("=")

    CHECKER.verify_recorded_sha256(library, "sha256", encoded_hash)


def test_nccl_record_sha256_rejects_different_library(tmp_path: Path) -> None:
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"installed cu12 file")
    record_hash = base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")

    with pytest.raises(SystemExit, match="differs from nvidia-nccl-cu13 RECORD"):
        CHECKER.verify_recorded_sha256(library, "sha256", record_hash)


def test_nccl_environment_requires_only_cu13() -> None:
    CHECKER.verify_nvidia_nccl_distribution_names(
        ["nvidia-nccl-cu13", "some-other-package"]
    )

    with pytest.raises(SystemExit, match="expected only nvidia-nccl-cu13"):
        CHECKER.verify_nvidia_nccl_distribution_names(
            ["nvidia-nccl-cu12", "nvidia-nccl-cu13"]
        )


def test_nccl_record_paths_have_one_distribution_owner() -> None:
    library = PurePosixPath("nvidia/nccl/lib/libnccl.so.2")
    CHECKER.verify_unique_nvidia_nccl_records([("nvidia-nccl-cu13", [library])])

    distributions: Iterable[tuple[str, Iterable[PurePosixPath]]] = [
        ("nvidia-nccl-cu12", [library]),
        ("nvidia-nccl-cu13", [library]),
    ]
    with pytest.raises(SystemExit, match="overlapping nvidia/nccl RECORD ownership"):
        CHECKER.verify_unique_nvidia_nccl_records(distributions)
