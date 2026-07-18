from typing import Literal, cast

import pytest

from layoutdiffusion import (
    LayoutDiffusionConfig,
    LayoutDiffusionProcessor,
    LayoutDiffusionTokenizer,
)


def test_processor_rejects_non_pt_return_tensors() -> None:
    processor = LayoutDiffusionProcessor(
        LayoutDiffusionTokenizer(LayoutDiffusionConfig(dataset_name="publaynet"))
    )
    with pytest.raises(ValueError):
        processor(return_tensors=cast(Literal["pt"], "np"))


def test_processor_returns_empty_when_no_inputs() -> None:
    processor = LayoutDiffusionProcessor(
        LayoutDiffusionTokenizer(LayoutDiffusionConfig(dataset_name="publaynet"))
    )
    assert processor() == {}


def test_processor_converts_num_elements_list() -> None:
    processor = LayoutDiffusionProcessor(
        LayoutDiffusionTokenizer(LayoutDiffusionConfig(dataset_name="publaynet"))
    )
    assert processor(num_elements=[1, 2])["num_elements"].tolist() == [1, 2]
