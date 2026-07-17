"""Tests for the Pydantic AI LayoutPrompter wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic_ai.models.test import TestModel

from laygen.common import BoxFormat, LayoutGenerationOutput
from laygen.common.testing import assert_layout_output_schema
from layoutprompter import LayoutPrompter, LayoutPrompterConfig
from layoutprompter.agent import ConditionType, OutputType, PromptFormat
from layoutprompter.data import LayoutPrompterDataset
from layoutprompter.schemas import LayoutElement, PixelBBox


def test_agent_runs_with_pydantic_ai_test_model_without_network() -> None:
    """The agent accepts a TestModel and parses structured output."""
    train_data = [
        {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[10, 10, 20, 20], [40, 40, 30, 30]]),
            "discrete_gold_bboxes": torch.tensor([[10, 10, 20, 20], [40, 40, 30, 30]]),
        }
    ]
    test_data = {
        "labels": torch.tensor([0, 1]),
        "bboxes": torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]),
        "discrete_gold_bboxes": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
    }
    model = TestModel(
        custom_output_args={
            "elements": [
                {
                    "label": "text",
                    "bbox": {"left": 12, "top": 16, "width": 24, "height": 32},
                }
            ]
        }
    )
    prompter = LayoutPrompter(
        LayoutPrompterConfig(model=model, shuffle=False, num_prompt=1)
    )
    output = prompter.run_sync(train_data, test_data)
    assert_layout_output_schema(output, batch_size=1)
    assert output.labels.tolist() == [[0]]


def test_save_pretrained_round_trip_preserves_prompt_config(tmp_path: Path) -> None:
    """LayoutPrompter persists prompt config without serializing provider state."""
    model = TestModel(
        custom_output_args={
            "elements": [
                {
                    "label": "text",
                    "bbox": {"left": 12, "top": 16, "width": 24, "height": 32},
                }
            ]
        }
    )
    prompter = LayoutPrompter(
        LayoutPrompterConfig(
            dataset="publaynet",
            condition_type="label_size",
            output_format="html",
            model=model,
            shuffle=False,
            seed=7,
        )
    )
    prompter.save_pretrained(tmp_path)
    loaded = LayoutPrompter.from_pretrained(tmp_path, model=model)
    assert loaded.config.dataset == "publaynet"
    assert loaded.config.condition_type == "label_size"
    assert loaded.config.output_format == "html"
    assert loaded.config.shuffle is False
    assert loaded.config.seed == 7
    assert isinstance(loaded.config.dataset, LayoutPrompterDataset)
    assert isinstance(loaded.config.condition_type, ConditionType)
    assert isinstance(loaded.config.output_format, PromptFormat)


def test_call_supports_shared_signature_dict_output_and_enum_inputs() -> None:
    """The public call boundary accepts string-compatible enums."""
    train_data = [
        {
            "labels": torch.tensor([0]),
            "bboxes": torch.tensor([[10, 10, 20, 20]]),
            "discrete_gold_bboxes": torch.tensor([[10, 10, 20, 20]]),
        }
    ]
    test_data = {
        "labels": torch.tensor([0]),
        "bboxes": torch.tensor([[0, 0, 1, 1]]),
        "discrete_gold_bboxes": torch.tensor([[0, 0, 1, 1]]),
    }
    model = TestModel(
        custom_output_args={
            "elements": [
                {
                    "label": "text",
                    "bbox": {"left": 12, "top": 16, "width": 24, "height": 32},
                }
            ]
        }
    )
    prompter = LayoutPrompter(
        LayoutPrompterConfig(
            dataset=LayoutPrompterDataset.PUBLAYNET,
            condition_type=ConditionType.label,
            input_format=PromptFormat.SEQ,
            output_format=PromptFormat.SEQ,
            model=model,
            shuffle=False,
            num_prompt=1,
        )
    )
    output = prompter(
        train_data=train_data,
        test_data=test_data,
        box_format=BoxFormat.xywh,
        output_type=OutputType.DICT,
    )
    dict_output = cast(dict[str, object], output)
    labels = cast(torch.Tensor, dict_output["labels"])
    assert labels.tolist() == [[0]]
    dataclass_output = cast(
        LayoutGenerationOutput, prompter(train_data=train_data, test_data=test_data)
    )
    assert dataclass_output.labels.tolist() == [[0]]


def test_config_and_call_reject_unsupported_modes() -> None:
    """Unsupported public string modes raise explicit ValueError."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        LayoutPrompterConfig(dataset="unknown")
    with pytest.raises(ValueError, match="Unknown condition_type"):
        LayoutPrompterConfig(condition_type="unknown")
    with pytest.raises(ValueError, match="Unsupported prompt format"):
        LayoutPrompterConfig(input_format="json")

    prompter = LayoutPrompter(
        LayoutPrompterConfig(
            model=TestModel(custom_output_args={"elements": []}),
            shuffle=False,
            num_prompt=1,
        )
    )
    with pytest.raises(ValueError, match="Unsupported box_format"):
        prompter(
            train_data=[], test_data={"labels": torch.tensor([])}, box_format="bad"
        )
    with pytest.raises(ValueError, match="Unsupported output_type"):
        prompter(
            train_data=[],
            test_data={"labels": torch.tensor([])},
            output_type="json",
        )


def test_resolve_model_prefers_explicit_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model resolution falls back through public environment variables."""
    model = TestModel(custom_output_args={"elements": []})
    assert LayoutPrompter._resolve_model(model) is model
    monkeypatch.setenv("LAYOUTPROMPTER_MODEL", "test-layout-model")
    assert LayoutPrompter._resolve_model(None) == "test-layout-model"
    monkeypatch.delenv("LAYOUTPROMPTER_MODEL")
    monkeypatch.setenv("PYDANTIC_AI_MODEL", "test-pydantic-model")
    assert LayoutPrompter._resolve_model(None) == "test-pydantic-model"
    monkeypatch.delenv("PYDANTIC_AI_MODEL")
    assert LayoutPrompter._resolve_model(None) == "openai:gpt-4o-mini"


def test_structured_label_normalization_removes_trailing_index() -> None:
    """Structured labels tolerate models copying prompt index tokens."""
    element = LayoutElement(
        label="Text 0",
        bbox=PixelBBox(left=1, top=2, width=3, height=4),
    )
    assert element.label == "text"
