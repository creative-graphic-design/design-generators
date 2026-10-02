import json
from typing import cast

import numpy as np
import pytest
import torch

from canvas_vae import CanvasVAEProcessor
from canvas_vae.processing_canvas_vae import (
    RicoDocument,
    RicoElement,
    RicoSplit,
    build_vocabularies,
    count_values,
    flatten_rico_hierarchy,
    load_rico_split,
    load_rico_vocabularies,
    read_rico_archive,
    split_for_hash,
    stable_hash,
)


def test_flatten_is_preorder_with_defaults(make_node):
    leaf = make_node(
        (0, 0, 144, 256), componentLabel="Icon", iconClass="arrow", clickable=True
    )
    middle = make_node((0, 0, 720, 1280), children=[leaf], textButtonClass="ok")
    root = make_node((0, 0, 1440, 2560), children=[middle, make_node((1, 2, 3, 4))])
    elements = list(flatten_rico_hierarchy(root))
    assert [e["width"] for e in elements] == [
        1.0,
        0.5,
        0.10000000149011612,
        float(np.float32(2 / 1440)),
    ]
    assert elements[0]["component"] == "" and elements[0]["clickable"] == 0
    assert elements[1]["text_button"] == "ok"
    assert elements[2] == {
        "left": 0.0,
        "top": 0.0,
        "width": float(np.float32(0.1)),
        "height": float(np.float32(0.1)),
        "class": "View",
        "clickable": 1,
        "component": "Icon",
        "icon": "arrow",
        "text_button": "",
    }


def test_split_and_hash_rules():
    assert split_for_hash(10) is RicoSplit.val
    assert split_for_hash(11) is RicoSplit.test
    assert split_for_hash(12) is RicoSplit.train
    assert stable_hash(b"{}") % 10 == 6


def test_read_archive_deduplicates_and_drops_long_screens(
    tmp_path, make_archive, make_node
):
    small = make_node((0, 0, 1440, 2560), children=[make_node((0, 0, 1, 1))])
    long = make_node((0, 0, 1440, 2560), children=[make_node((0, 0, 1, 1))] * 50)
    archive = make_archive(tmp_path / "a.zip", {"7": small, "3": small, "5": long})
    documents = read_rico_archive(archive)
    assert [document["id"] for document in documents] == [7]
    assert len(documents[0]["elements"]) == 2
    assert documents[0]["split"] == split_for_hash(
        int(documents[0]["content_hash"], 16)
    )


def test_counts_order_and_vocabularies():
    elements = [
        cast(RicoElement, {"class": "A", "component": c, "icon": i, "text_button": ""})
        for c, i in [("Text", "b"), ("Icon", "a"), ("Text", "a"), ("", "b")]
    ]
    document: RicoDocument = {
        "id": 0,
        "content_hash": "",
        "split": "train",
        "elements": elements,
    }
    counts = count_values([document])
    assert list(counts["component"].items()) == [("Text", 2), ("", 1), ("Icon", 1)]
    assert list(counts["icon"]) == ["a", "b"]
    tables = build_vocabularies(counts, {"component": 1, "icon": 2, "text_button": 5})
    assert tables == {
        "component": ["[UNK]", "Text", "", "Icon"],
        "icon": ["[UNK]", "a", "b"],
        "text_button": ["[UNK]"],
    }


def test_prepare_and_load_cache(rico_dir):
    sizes = json.loads((rico_dir / "count.json").read_text())
    assert sum(sizes.values()) == 40
    train = load_rico_split(rico_dir, "train")
    assert len(train) == sizes["train"]
    assert load_rico_vocabularies(rico_dir)["component"][0] == "[UNK]"


def test_processor_encodes_and_pads(vocabularies):
    processor = CanvasVAEProcessor(vocabularies, num_bins=8)
    element: RicoElement = {
        "left": 0.5,
        "top": 1.0,
        "width": 0.1,
        "height": 0.0,
        "class": "View",
        "clickable": 1,
        "component": "Icon",
        "icon": "missing",
        "text_button": "",
    }
    encoded = processor([[element, element], [element]])
    assert encoded["num_elements"].tolist() == [2, 1]
    ids = encoded["element_ids"]
    assert ids.dtype == torch.int64 and ids.shape == (2, 2, 8)
    assert ids[0, 0].tolist() == [3, 7, 0, 0, 1, 3, 0, 1]
    assert ids[1, 1].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    boundary = float(processor.bin_boundaries[2])
    assert processor.discretize(
        np.array([boundary, np.nextafter(boundary, 0, dtype=np.float32)])
    ).tolist() == [3, 2]
    with pytest.raises(ValueError):
        processor([[]])


def test_processor_round_trip(tmp_path, vocabularies):
    processor = CanvasVAEProcessor(vocabularies, num_bins=8)
    processor.save_pretrained(tmp_path)
    loaded = CanvasVAEProcessor.from_pretrained(tmp_path)
    assert loaded.vocabularies == vocabularies and loaded.num_bins == 8
