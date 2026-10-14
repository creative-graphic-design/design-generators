import base64
import hashlib
import importlib.util
from pathlib import Path

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
