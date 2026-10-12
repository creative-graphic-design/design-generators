"""CPU tests for the Crello full-run embedding cache."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import cast

import numpy as np

from canvas_vae.data import image_id, load_embedding_fixture

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_crello_s5_embeddings.py"
SPEC = importlib.util.spec_from_file_location("prepare_crello_s5_embeddings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def _document(split: str, document_id: str, image_ids: list[str]) -> dict[str, object]:
    return {
        "split": split,
        "document_id": f"crello-v1/{split}/{document_id}",
        "context": {"id": document_id},
        "elements": [{"type": "imageElement", "image_id": key} for key in image_ids],
    }


def _write_test_corpus(root: Path) -> tuple[Path, dict[str, list[str]]]:
    package_data = root / "package"
    image_keys: dict[str, list[str]] = {}
    for split, value in (("train", 10), ("val", 20), ("test", 30)):
        key = image_id(split, f"test-png-{value}".encode())
        image_keys[split] = [key]
        document_images = [key, key] if split == "train" else [key]
        (package_data / f"{split}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (package_data / f"{split}.jsonl").write_text(
            json.dumps(_document(split, str(value), document_images)) + "\n",
            encoding="utf-8",
        )

    return package_data, image_keys


def test_tiny_cpu_cache_encodes_each_unique_crello_image_once(tmp_path: Path) -> None:
    package_data, image_keys = _write_test_corpus(tmp_path)
    seen: list[int] = []
    value_by_key = {
        image_keys[split][0]: value
        for split, value in (("train", 10), ("val", 20), ("test", 30))
    }

    def read_image(_data_dir: Path, key: str) -> np.ndarray:
        value = value_by_key[key]
        return np.full((256, 256, 4), value, dtype=np.uint8)

    def encode(images: np.ndarray) -> np.ndarray:
        seen.extend(int(image[0, 0, 0]) for image in images)
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.repeat(values[:, None], 256, axis=1)

    cache_dir = tmp_path / "embedding-cache"
    manifest = PREPARE.create_embedding_cache(
        package_data,
        cache_dir,
        encode,
        encoder_sha256="1" * 64,
        batch_size=2,
        _read_image=read_image,
    )
    expected = [image_keys[split][0] for split in ("train", "val", "test")]
    embeddings = load_embedding_fixture(cache_dir, expected)

    assert len(manifest["records"]) == len(expected) == len(seen)
    assert len(set(seen)) == len(expected)
    assert [embeddings[key][0].item() for key in expected] == [10, 20, 30]


def test_original_rows_follow_document_order_and_reuse_duplicate_ids() -> None:
    key_a = image_id("train", b"a")
    key_b = image_id("train", b"b")
    document = cast(
        PREPARE.CrelloDocument,
        _document("train", "0", [key_b, key_a, key_b]),
    )
    row_a = np.full(256, 1, dtype=np.float32)
    row_b = np.full(256, 2, dtype=np.float32)

    rows = PREPARE._embedding_rows(document, {key_a: row_a, key_b: row_b})

    assert np.array_equal(rows[0], row_b)
    assert np.array_equal(rows[1], row_a)
    assert np.array_equal(rows[2], row_b)
    assert rows[0] is rows[2]
