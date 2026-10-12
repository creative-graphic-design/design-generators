import json
from typing import cast
import zipfile
from pathlib import Path

import pytest

from canvas_vae import CanvasVAEConfig
from canvas_vae.processing_canvas_vae import RicoNode

VOCABULARIES = {
    "component": ["[UNK]", "", "Text", "Icon"],
    "icon": ["[UNK]", "", "arrow"],
    "text_button": ["[UNK]", ""],
}


def tiny_config(**overrides: float) -> CanvasVAEConfig:
    config = CanvasVAEConfig(
        vocabularies=VOCABULARIES, max_length=6, num_bins=8, latent_dim=16, num_heads=2
    )
    for key, value in overrides.items():
        setattr(config, key, value)

    return config


@pytest.fixture
def config() -> CanvasVAEConfig:
    return tiny_config()


@pytest.fixture
def make_config():
    return tiny_config


@pytest.fixture
def vocabularies():
    return VOCABULARIES


def node(bounds, *, children=(), **attributes: str | bool) -> RicoNode:
    return cast(
        RicoNode,
        {
            "bounds": list(bounds),
            "class": "View",
            "children": list(children),
            **attributes,
        },
    )


def write_archive(path: Path, screens: dict[str, RicoNode]) -> Path:
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("semantic_annotations/", "")
        for name, screen in screens.items():
            handle.writestr(f"semantic_annotations/{name}.json", json.dumps(screen))
            handle.writestr(f"semantic_annotations/{name}.png", b"")

    return path


def synthetic_screens(count: int = 40) -> dict[str, RicoNode]:
    screens = {}
    for index in range(count):
        children = [
            node(
                (10 * j, 20 * j, 10 * j + 300, 20 * j + 200),
                componentLabel="Text" if j % 2 else "Icon",
                iconClass="arrow" if j % 3 == 0 else "",
                clickable=bool(j % 2),
            )
            for j in range(1 + index % 4)
        ]
        screens[str(index)] = node((0, 0, 1440, 2560 - index), children=children)

    return screens


@pytest.fixture
def rico_dir(tmp_path: Path) -> Path:
    from canvas_vae.processing_canvas_vae import prepare_rico_cache

    archive = write_archive(tmp_path / "rico.zip", synthetic_screens())
    output = tmp_path / "rico"
    prepare_rico_cache(archive, output)
    return output


@pytest.fixture
def make_archive():
    return write_archive


@pytest.fixture
def make_node():
    return node
