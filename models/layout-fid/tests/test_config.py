from typing import cast

import pytest

from layout_fid import (
    LayoutFIDArchitecture,
    LayoutFIDConfig,
    LayoutFIDSource,
    LayoutFIDStatsSplit,
    normalize_architecture,
    normalize_source,
    normalize_stats_split,
)


def test_config_round_trip(tmp_path):
    cfg = LayoutFIDConfig(
        dataset_name="rico25",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=25,
        num_label_embeddings=26,
        max_length=20,
    )
    cfg.save_pretrained(tmp_path)
    loaded = LayoutFIDConfig.from_pretrained(tmp_path)
    assert loaded.architecture == "layoutnet"
    assert loaded.source == "layoutflow"
    assert loaded.reference_stats["test"] == "reference_stats/test.npz"
    assert cast(dict[int, str], loaded.id2label)[0] == "Text"


def test_config_normalization_and_validation_errors():
    assert normalize_architecture(LayoutFIDArchitecture.layoutnet) == "layoutnet"
    assert normalize_source(LayoutFIDSource.layoutflow) == "layoutflow"
    assert normalize_stats_split(LayoutFIDStatsSplit.test) == "test"
    for func, value in (
        (normalize_architecture, "bad"),
        (normalize_source, "bad"),
        (normalize_stats_split, "bad"),
    ):
        with pytest.raises(ValueError):
            func(value)
    with pytest.raises(ValueError):
        LayoutFIDConfig(
            dataset_name="publaynet",
            architecture="layoutnet",
            source="layoutflow",
            num_public_labels=0,
            num_label_embeddings=6,
            max_length=20,
        )
