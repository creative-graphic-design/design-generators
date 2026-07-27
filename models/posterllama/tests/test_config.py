from __future__ import annotations

from pathlib import Path
from typing import cast

from posterllama import PosterLlamaConfig


def test_config_round_trip(tmp_path: Path) -> None:
    config = PosterLlamaConfig(canvas_size=(360, 504))
    config.save_pretrained(tmp_path)

    loaded = PosterLlamaConfig.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.canvas_size == (360, 504)
    assert loaded.checkpoint_license_status == "unverified"
    id2label = cast(dict[int, str], loaded.id2label)
    assert id2label[0] == "logo"
    assert id2label[1] == "text"


def test_config_contains_no_absolute_runtime_paths() -> None:
    config = PosterLlamaConfig()

    for value in config.to_dict().values():
        if isinstance(value, str):
            assert not value.startswith("/")
