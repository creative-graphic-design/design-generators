"""Tests for the Pydantic AI LayoutPrompter wrapper."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic_ai.models.test import TestModel

from layout_generation_common.testing import assert_layout_output_schema
from layoutprompter import LayoutPrompter, LayoutPrompterConfig


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
