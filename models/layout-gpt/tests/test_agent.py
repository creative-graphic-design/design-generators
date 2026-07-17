"""Tests for the LayoutGPT Pydantic AI runner."""

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest
import torch

from layout_generation_common.testing import assert_layout_output_schema
from layout_gpt import LayoutGPTAgent
from layout_gpt.exemplars import LayoutExample
from layout_gpt.schema import LayoutGPTConfig


def _examples() -> list[LayoutExample]:
    return [
        LayoutExample(
            id=1,
            prompt="one clock is in the picture",
            objects=(("clock", (0.25, 0.25, 0.50, 0.50)),),
            metadata={},
        ),
        LayoutExample(
            id=2,
            prompt="one cat is in the picture",
            objects=(("cat", (0.10, 0.10, 0.20, 0.20)),),
            metadata={},
        ),
    ]


def test_agent_runs_function_model_and_returns_common_schema() -> None:
    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content='{"text":"clock {height: 32px; width: 32px; top: 16px; left: 16px; }"}'
                )
            ]
        )

    runner = LayoutGPTAgent(
        model=FunctionModel(respond),
        config=LayoutGPTConfig(icl_type="fixed-random", k=1, canvas_size=64),
    )

    output = runner.run_sync(
        "there is one clock in the image", train_examples=_examples()
    )

    assert output.items[0].label == "clock"
    assert output.selected_exemplar_ids == [2]
    assert output.prompt_messages is not None
    assert "Prompt: one cat is in the picture" in output.prompt_messages[1]["content"]
    assert_layout_output_schema(output.to_layout_generation_output(), batch_size=1)


def test_public_call_returns_common_output_and_rejects_unsupported_conditions() -> None:
    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content='{"text":"cat {height: 16px; width: 16px; top: 0px; left: 0px; }"}'
                )
            ]
        )

    runner = LayoutGPTAgent(
        model=FunctionModel(respond),
        config=LayoutGPTConfig(icl_type="fixed-random", k=1, canvas_size=64),
    )

    output = runner(
        prompt="there is one cat in the image",
        train_examples=_examples(),
        generator=torch.Generator().manual_seed(42),
        return_intermediates=True,
    )

    assert not isinstance(output, dict)
    assert_layout_output_schema(output, batch_size=1)
    assert output.intermediates is not None
    with pytest.raises(ValueError, match="unsupported condition_type"):
        runner(
            prompt="there is one cat in the image",
            train_examples=_examples(),
            condition_type="relation",
        )
