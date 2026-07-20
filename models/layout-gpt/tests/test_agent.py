"""Tests for the LayoutGPT Pydantic AI runner."""

from pathlib import Path
from typing import cast

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest
import torch

from laygen.common import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.common.testing import assert_layout_output_schema
from layout_gpt import LayoutGPTAgent
from layout_gpt.enums import OutputType
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

    raw_output = runner(
        prompt="there is one cat in the image",
        train_examples=_examples(),
        generator=torch.Generator().manual_seed(42),
        return_intermediates=True,
    )

    assert type(raw_output) is not dict
    output = cast(LayoutGenerationOutput, raw_output)
    assert_layout_output_schema(output, batch_size=1)
    assert output.intermediates is not None
    with pytest.raises(ValueError, match="relation"):
        runner(
            prompt="there is one cat in the image",
            train_examples=_examples(),
            condition_type="relation",
        )


def test_public_call_can_return_dict_and_hide_intermediates() -> None:
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

    output = runner(
        prompt="there is one clock in the image",
        train_examples=_examples(),
        condition_type=ConditionType.unconditional,
        output_type=OutputType.dict,
    )

    assert isinstance(output, dict)
    output_dict = cast(dict[str, object], output)
    assert output_dict["intermediates"] is None
    assert output_dict["id2label"] == {0: "clock"}


def test_save_pretrained_round_trip_preserves_prompt_config(tmp_path: Path) -> None:
    """LayoutGPT persists prompt config without serializing provider state."""
    model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[]))
    runner = LayoutGPTAgent(
        model=model,
        config=LayoutGPTConfig(
            setting="spatial",
            icl_type="fixed-random",
            k=3,
            canvas_size=128,
            chat=False,
            temperature=0.2,
            top_p=0.8,
        ),
    )

    runner.save_pretrained(tmp_path)
    loaded = LayoutGPTAgent.from_pretrained(tmp_path, model=model)

    assert loaded.config.setting == "spatial"
    assert loaded.config.icl_type == "fixed-random"
    assert loaded.config.k == 3
    assert loaded.config.canvas_size == 128
    assert loaded.config.chat is False
    assert loaded.config.temperature == 0.2
    assert loaded.config.top_p == 0.8


def test_public_call_validates_batch_size_and_canvas() -> None:
    runner = LayoutGPTAgent(
        model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[])),
        config=LayoutGPTConfig(icl_type="fixed-random", k=1, canvas_size=64),
    )

    with pytest.raises(ValueError, match="batch_size=1"):
        runner(prompt="x", train_examples=_examples(), batch_size=2)
    with pytest.raises(ValueError, match="square canvas_size"):
        runner(prompt="x", train_examples=_examples(), canvas_size=(128, 128))


def test_k_similar_agent_requires_embeddings_and_can_build_completion_prompt() -> None:
    runner = LayoutGPTAgent(
        model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[])),
        config=LayoutGPTConfig(icl_type="k-similar", k=1, canvas_size=64, chat=False),
    )

    with pytest.raises(ValueError, match="k-similar"):
        runner.build_prompt("query", train_examples=_examples())

    prompt, exemplars = runner.build_prompt(
        "query",
        train_examples=_examples(),
        query_embedding=lambda _query: (0.0, 1.0),
        example_embeddings=[(1.0, 0.0), (0.0, 1.0)],
    )

    assert isinstance(prompt, str)
    assert exemplars[0].id == 2
    assert prompt.endswith("\nPrompt: query\nLayout:")
