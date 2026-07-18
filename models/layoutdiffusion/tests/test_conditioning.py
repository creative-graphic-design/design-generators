import pytest
import torch

from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionTokenizer
from layoutdiffusion.conditioning import build_condition
from laygen.common import ConditionType


def _tokenizer() -> LayoutDiffusionTokenizer:
    return LayoutDiffusionTokenizer(LayoutDiffusionConfig(dataset_name="publaynet"))


def test_build_unconditional_condition() -> None:
    condition = build_condition(_tokenizer(), condition_type="unconditional")
    assert condition is not None
    assert condition.type is ConditionType.unconditional


def test_build_label_condition_requires_labels() -> None:
    with pytest.raises(ValueError):
        build_condition(_tokenizer(), condition_type="label")


def test_build_refinement_condition_requires_input_ids() -> None:
    with pytest.raises(ValueError):
        build_condition(_tokenizer(), condition_type="refinement")


def test_build_refinement_condition_uses_input_ids() -> None:
    input_ids = torch.zeros(1, 121, dtype=torch.long)
    condition = build_condition(
        _tokenizer(), condition_type="refine", input_ids=input_ids
    )
    assert condition is not None
    assert condition.input_ids is input_ids


def test_build_unsupported_condition_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_condition(_tokenizer(), condition_type="label_size")
