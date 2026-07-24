from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_original_assets_are_explicitly_selected() -> None:
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 and PosterLlama asset paths")
    required = [
        "POSTERLLAMA_VENDOR_ROOT",
        "POSTERLLAMA_CHECKPOINT_PATH",
        "POSTERLLAMA_BASE_LLM_PATH",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise AssertionError(f"Missing PosterLlama parity env vars: {missing}")
    for name in required:
        assert Path(os.environ[name]).exists()
