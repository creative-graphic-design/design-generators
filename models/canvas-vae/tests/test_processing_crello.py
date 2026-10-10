import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import canvas_vae.data as crello
from canvas_vae.data import (
    CRELLO_IMAGE_TYPES,
    CrelloDocument,
    CrelloElement,
    CrelloProcessor,
    CrelloSplit,
    document_id,
    extract_crello_archive,
    fixture_image_ids,
    image_id,
    load_crello_split,
    load_crello_vocabularies,
    load_embedding_fixture,
    training_image_ids,
    write_embedding_fixture,
)


VOCABULARIES = {
    "length": list(range(1, 51)),
    "group": ["", "group"],
    "format": ["", "poster"],
    "category": ["", "food"],
    "canvas_width": [640],
    "canvas_height": [480],
    "type": ["", "textElement", "coloredBackground", "imageElement", "svgElement"],
}


def element(kind: str, key: str, *, offset: int = 0) -> CrelloElement:
    return {
        "type": kind,
        "left": 0.5,
        "top": 0.25,
        "width": 0.75,
        "height": 0.125,
        "opacity": 0.5,
        "color": [offset, 128, 255],
        "image_id": key,
    }


def document(source_id: str, rows: list[CrelloElement]) -> CrelloDocument:
    return {
        "split": "train",
        "document_id": document_id(CrelloSplit.train, source_id),
        "context": {
            "id": source_id,
            "length": len(rows),
            "group": "group",
            "format": "poster",
            "canvas_width": 640,
            "canvas_height": 480,
            "category": "food",
        },
        "elements": rows,
    }


def test_stable_ids_are_split_scoped_and_hash_exact_bytes():
    png = b"one exact png byte string"
    assert document_id("val", 42) == "crello-v1/val/42"
    assert (
        image_id("train", png) == f"crello-v1/train/{hashlib.sha256(png).hexdigest()}"
    )
    assert image_id("train", png) != image_id("val", png)


def test_fixture_ids_deduplicate_before_canonical_encoding():
    late = document(
        "z", [element("svgElement", "repeat"), element("textElement", "other")]
    )
    early = document(
        "a", [element("otherElement", "other"), element("imageElement", "first")]
    )
    ids = fixture_image_ids([late, early])
    assert ids == ["other", "first", "repeat"]
    assert len(ids) == len(set(ids))
    assert ids.count("other") == 1
    assert training_image_ids([late, early]) == ["first", "repeat"]
    assert CRELLO_IMAGE_TYPES == frozenset(
        ("imageElement", "maskElement", "svgElement")
    )


def test_processor_preserves_fields_builds_masks_and_pads_deterministically():
    first = document(
        "a", [element("textElement", "text"), element("svgElement", "svg", offset=17)]
    )
    second = document("b", [element("coloredBackground", "background")])
    embeddings = {
        key: np.full(256, index + 0.25, dtype=np.float32)
        for index, key in enumerate(("text", "svg", "background"))
    }
    processor = CrelloProcessor(VOCABULARIES, embeddings)
    batch = processor([first, second])
    replay = processor([first, second])

    assert batch["document_id"] == [first["document_id"], second["document_id"]]
    assert batch["num_elements"].tolist() == [2, 1]
    assert batch["length"].tolist() == [[1], [0]]
    assert batch["group"].tolist() == [[1], [1]]
    assert batch["type"].tolist() == [[[1], [4]], [[2], [0]]]
    assert batch["element_mask"].tolist() == [[True, True], [True, False]]
    assert batch["color_mask"].tolist() == [[True, False], [True, False]]
    assert batch["image_embedding_mask"].tolist() == [[False, True], [False, False]]
    assert batch["color"][0, 1].tolist() == [1, 7, 15]
    assert batch["image_embedding"][0, 0, 0].item() == 0.25
    assert batch["image_embedding"][1, 1].count_nonzero().item() == 0
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            assert torch.equal(batch[key], replay[key])
        else:
            assert batch[key] == replay[key]


def test_processor_rejects_missing_or_invalid_embeddings():
    row = document("a", [element("imageElement", "missing")])
    processor = CrelloProcessor(VOCABULARIES, {})
    with pytest.raises(ValueError, match="missing posterior mean"):
        processor([row])

    processor = CrelloProcessor(
        VOCABULARIES, {"missing": np.zeros(255, dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="invalid posterior mean"):
        processor([row])


def test_fixture_round_trip_checks_manifest_array_and_row_hashes(tmp_path: Path):
    ids = [image_id("train", b"first"), image_id("val", b"second")]
    values = np.arange(512, dtype=np.float32).reshape(2, 256)
    write_embedding_fixture(
        tmp_path,
        ids,
        values,
        encoder_state_sha256="a" * 64,
    )
    loaded = load_embedding_fixture(tmp_path, ids)
    assert np.array_equal(loaded[ids[0]], values[0])
    assert np.array_equal(loaded[ids[1]], values[1])
    with pytest.raises(ValueError, match="canonical first-occurrence order"):
        load_embedding_fixture(tmp_path, ids[::-1])

    array_path = tmp_path / "posterior_means.npy"
    original_array = array_path.read_bytes()
    array_path.write_bytes(original_array[:-1] + bytes([original_array[-1] ^ 1]))
    with pytest.raises(ValueError, match="array SHA-256"):
        load_embedding_fixture(tmp_path)
    array_path.write_bytes(original_array)

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["records"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(), encoding="ascii"
    )
    with pytest.raises(ValueError, match="fixture row hash"):
        load_embedding_fixture(tmp_path)
    (tmp_path / "manifest.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_embedding_fixture(tmp_path)


def test_fixture_requires_float32_and_finite_values(tmp_path: Path):
    keys = [image_id("train", b"png")]
    with pytest.raises(ValueError, match="float32 posterior means"):
        write_embedding_fixture(
            tmp_path,
            keys,
            np.zeros((1, 256), dtype=np.float64),
            encoder_state_sha256="b" * 64,
        )
    values = np.zeros((1, 256), dtype=np.float32)
    values[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite float32"):
        write_embedding_fixture(
            tmp_path,
            keys,
            values,
            encoder_state_sha256="b" * 64,
        )


def test_split_loader_enforces_utf8_order_and_unique_ids(tmp_path: Path):
    docs = [
        document("z", [element("imageElement", "a")]),
        document("a", [element("imageElement", "b")]),
    ]
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in docs[::-1]), encoding="utf-8"
    )
    loaded = load_crello_split(tmp_path, "train")
    assert [row["document_id"] for row in loaded] == [
        docs[1]["document_id"],
        docs[0]["document_id"],
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in docs), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical train order"):
        load_crello_split(tmp_path, "train")


def test_vocabularies_keep_source_order_and_insert_string_mask_token(tmp_path: Path):
    source = {
        "group": {"group": 20, "": 2},
        "format": {"poster": 3},
        "category": {"food": 8},
        "type": {"imageElement": 4, "textElement": 1},
        "canvas_width": {"640": 2},
        "canvas_height": {"480": 2},
    }
    (tmp_path / "vocabulary.json").write_text(json.dumps(source), encoding="utf-8")
    vocabularies = load_crello_vocabularies(tmp_path)
    assert vocabularies["group"] == ["", "group"]
    assert vocabularies["format"] == ["", "poster"]
    assert vocabularies["type"] == ["", "imageElement", "textElement"]
    assert vocabularies["canvas_width"] == [640]
    assert vocabularies["length"] == list(range(1, 51))


def test_extraction_rejects_unsafe_archive_members(tmp_path: Path, monkeypatch):
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../outside.txt", "private")
    monkeypatch.setattr(crello, "verify_crello_archive", lambda _: "verified")
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="unsafe path"):
        extract_crello_archive(source, target)
    assert not target.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_extraction_marks_only_verified_archive_content(tmp_path: Path, monkeypatch):
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("train-000.tfrecord", b"private data")
    digest = "c" * 64
    monkeypatch.setattr(crello, "verify_crello_archive", lambda _: digest)
    target = extract_crello_archive(source, tmp_path / "extracted")
    assert (target / "train-000.tfrecord").read_bytes() == b"private data"
    assert (target / ".crello-v1-sha256").read_text("ascii").strip() == digest


def test_download_verifies_size_and_digest_before_rename(tmp_path: Path, monkeypatch):
    content = b"small synthetic archive"

    class Response:
        def __enter__(self):
            from io import BytesIO

            self.stream = BytesIO(content)
            return self

        def __exit__(self, *_):
            self.stream.close()

        def read(self, size: int) -> bytes:
            return self.stream.read(size)

    monkeypatch.setattr(crello, "CRELLO_V1_SIZE", len(content))
    monkeypatch.setattr(crello, "CRELLO_V1_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(crello.urllib.request, "urlopen", lambda _: Response())
    path = crello.download_crello_archive(tmp_path / "pinned.zip")
    assert path.read_bytes() == content
    assert crello.verify_crello_archive(path) == hashlib.sha256(content).hexdigest()
    assert crello.download_crello_archive(path) == path

    monkeypatch.setattr(crello, "CRELLO_V1_SIZE", len(content) + 1)
    with pytest.raises(ValueError, match="archive mismatch"):
        crello.verify_crello_archive(path)


def test_processor_rejects_empty_batches_and_unknown_vocabularies():
    processor = CrelloProcessor(VOCABULARIES, {})
    with pytest.raises(ValueError, match="needs at least one document"):
        processor([])
    row = document("a", [element("not-in-vocabulary", "missing")])
    processor = CrelloProcessor(
        VOCABULARIES, {"missing": np.zeros(256, dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="unknown Crello type"):
        processor([row])
