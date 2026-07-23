import json
from typing import Literal, cast

import pytest

from layout_detr.datasets import (
    label2id_for_ad_banner,
    load_ad_banner_dataset,
    normalize_vendor_annotation,
)


def test_normalize_vendor_annotation_ltrb_to_xywh():
    row = normalize_vendor_annotation(
        {
            "xyxy_word_fit": [10, 20, 30, 60],
            "label": "button",
            "str": "Shop",
            "width": 100,
            "height": 200,
        }
    )

    assert row["label"] == label2id_for_ad_banner()["button"]
    assert row["text"] == "Shop"
    assert row["bbox"] == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_dataset_loader_local_vendor_json(tmp_path):
    payload = [{"label": "header", "str": "Sale"}]
    (tmp_path / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = list(load_ad_banner_dataset(tmp_path, split="train"))

    assert rows == [{"path": str(tmp_path / "sample.json"), "elements": payload}]
    with pytest.raises(ValueError):
        list(
            load_ad_banner_dataset(
                tmp_path,
                split="train",
                source=cast(Literal["vendor"], "hf"),
            )
        )


def test_normalize_vendor_annotation_rejects_unknown_label():
    with pytest.raises(ValueError):
        normalize_vendor_annotation(
            {
                "xyxy_word_fit": [0, 0, 1, 1],
                "label": "missing",
                "width": 10,
                "height": 10,
            }
        )
