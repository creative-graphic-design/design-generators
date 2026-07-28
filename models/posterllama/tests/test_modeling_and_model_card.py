from __future__ import annotations

from pathlib import Path

import pytest

from posterllama import PosterLlamaConfig
from posterllama.model_card import model_card_metadata
from posterllama.modeling_posterllama import PosterLlamaRuntime


def test_runtime_save_load_round_trip(tmp_path: Path) -> None:
    runtime = PosterLlamaRuntime("<svg></svg>")
    runtime.save_pretrained(tmp_path)

    loaded = PosterLlamaRuntime.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.generate_texts(["a", "b"]) == ["<svg></svg>", "<svg></svg>"]


def test_runtime_missing_assets_error() -> None:
    runtime = PosterLlamaRuntime()

    with pytest.raises(RuntimeError, match="runtime assets are missing"):
        runtime.generate_texts(["prompt"])


def test_runtime_skips_non_main_process_save(tmp_path: Path) -> None:
    PosterLlamaRuntime("<svg></svg>").save_pretrained(tmp_path, is_main_process=False)

    assert not (tmp_path / "runtime_config.json").exists()


def test_model_card_metadata() -> None:
    metadata = model_card_metadata(PosterLlamaConfig(), hub_id="org/posterllama-cgl")

    assert metadata["hub_id"] == "org/posterllama-cgl"
    assert metadata["library_name"] == "transformers"
