"""Tests for PosterO configuration."""

from postero.config import CGL_ID2LABEL, PKU_ID2LABEL, PosterOConfig
from postero.enums import PosterORankStrategy, PosterOStructure


def test_config_normalizes_dataset_modes_and_label_maps() -> None:
    config = PosterOConfig(dataset_name="pku", structure="plain")
    assert config.dataset_name == "pku_posterlayout"
    assert config.structure is PosterOStructure.plain
    assert config.id2label == PKU_ID2LABEL

    cgl = PosterOConfig(dataset_name="cgl", rank_strategy="random")
    assert cgl.rank_strategy is PosterORankStrategy.random
    assert cgl.id2label == CGL_ID2LABEL


def test_config_preserves_explicit_one_based_label_map() -> None:
    config = PosterOConfig(id2label={10: "headline"})
    assert config.id2label == {10: "headline"}
