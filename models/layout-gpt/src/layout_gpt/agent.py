"""Provider-agnostic Pydantic AI wrapper for LayoutGPT."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Final, assert_never, cast

import torch
from layout_generation_common.outputs import LayoutGenerationOutput
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from layout_gpt.enums import BoxFormat, ConditionType, ICLType, OutputType, coerce_enum
from layout_gpt.exemplars import (
    EmbeddingProvider,
    LayoutExample,
    select_fixed_random,
    select_k_similar,
)
from layout_gpt.parser import parse_layout_text
from layout_gpt.prompts import (
    TokenCounter,
    default_token_counter,
    form_prompt_for_chatgpt,
    form_prompt_for_gpt3,
)
from layout_gpt.schema import LayoutGPTConfig, LayoutGPTOutput, RawLayoutResponse

ModelLike = Model | str | None
DEFAULT_MODEL_ENV_VAR: Final[str] = "LAYOUT_GPT_MODEL"


def build_agent(model: ModelLike = None) -> Agent[None]:
    """Build a Pydantic AI agent with provider selected by argument or env."""
    selected_model = model or os.getenv(DEFAULT_MODEL_ENV_VAR)
    return Agent(
        selected_model,
        output_type=RawLayoutResponse,
        instructions=(
            "You are LayoutGPT. Return CSS layout lines for the requested prompt. "
            "Each line must be `object {height: ?px; width: ?px; top: ?px; left: ?px; }`."
        ),
    )


class LayoutGPTAgent:
    """High-level LayoutGPT runner that ports vendor prompt and parse strategy."""

    def __init__(
        self,
        *,
        model: ModelLike = None,
        config: LayoutGPTConfig | None = None,
        token_counter: TokenCounter = default_token_counter,
    ) -> None:
        """Initialize the runner with provider and prompt configuration."""
        self.config = config or LayoutGPTConfig()
        self.agent = build_agent(model)
        self.token_counter = token_counter

    def build_prompt(
        self,
        prompt: str,
        *,
        train_examples: Sequence[LayoutExample],
        seed: int | None = None,
        generator: torch.Generator | None = None,
        query_embedding: EmbeddingProvider | None = None,
        example_embeddings: Sequence[Sequence[float]] | None = None,
    ) -> tuple[str | list[dict[str, str]], list[LayoutExample]]:
        """Select exemplars and serialize the prompt sent to the model."""
        exemplars = self._select_examples(
            prompt,
            train_examples=train_examples,
            seed=seed,
            generator=generator,
            query_embedding=query_embedding,
            example_embeddings=example_embeddings,
        )
        if self.config.chat:
            return (
                form_prompt_for_chatgpt(
                    prompt,
                    exemplars=exemplars,
                    canvas_size=self.config.canvas_size,
                    token_counter=self.token_counter,
                    input_length_limit=self.config.gpt_input_length_limit,
                ),
                exemplars,
            )
        return (
            form_prompt_for_gpt3(
                prompt,
                exemplars=exemplars,
                canvas_size=self.config.canvas_size,
                token_counter=self.token_counter,
                input_length_limit=self.config.gpt_input_length_limit,
            ),
            exemplars,
        )

    def run_sync(
        self,
        prompt: str,
        *,
        train_examples: Sequence[LayoutExample],
        model: ModelLike = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        query_embedding: EmbeddingProvider | None = None,
        example_embeddings: Sequence[Sequence[float]] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> LayoutGPTOutput:
        """Run LayoutGPT and parse the response into typed layout output."""
        model_prompt, exemplars = self.build_prompt(
            prompt,
            train_examples=train_examples,
            seed=seed,
            generator=generator,
            query_embedding=query_embedding,
            example_embeddings=example_embeddings,
        )
        prompt_text = _messages_to_text(model_prompt)
        run_result = self.agent.run_sync(
            prompt_text,
            model=model,
            model_settings=model_settings
            or ModelSettings(
                temperature=self.config.temperature, top_p=self.config.top_p
            ),
        )
        raw = cast(RawLayoutResponse, run_result.output)
        items = parse_layout_text(raw.text, canvas_size=self.config.canvas_size)
        id2label = {
            index: label
            for index, label in enumerate(dict.fromkeys(item.label for item in items))
        }
        return LayoutGPTOutput(
            prompt=prompt,
            canvas_size=self.config.canvas_size,
            items=items,
            raw_text=raw.text,
            id2label=id2label,
            selected_exemplar_ids=[example.id for example in exemplars],
            prompt_messages=model_prompt if isinstance(model_prompt, list) else None,
        )

    def __call__(
        self,
        *,
        prompt: str,
        train_examples: Sequence[LayoutExample],
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: str | ConditionType = ConditionType.TEXT,
        labels: torch.Tensor | list[object] | None = None,
        bbox: torch.Tensor | list[object] | None = None,
        mask: torch.Tensor | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: str | BoxFormat = BoxFormat.XYWH,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: str | OutputType = OutputType.DATACLASS,
        return_intermediates: bool = False,
        model: ModelLike = None,
        query_embedding: EmbeddingProvider | None = None,
        example_embeddings: Sequence[Sequence[float]] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate a layout through the common public generation surface."""
        del labels, bbox, mask, num_elements, normalized, num_inference_steps
        normalized_condition_type = coerce_enum(condition_type, ConditionType)
        normalized_box_format = coerce_enum(box_format, BoxFormat)
        normalized_output_type = coerce_enum(output_type, OutputType)
        del normalized_box_format
        if batch_size != 1:
            msg = "LayoutGPT currently supports batch_size=1 because provider calls are prompt-level."
            raise ValueError(msg)
        if normalized_condition_type is ConditionType.TEXT:
            pass
        elif normalized_condition_type is ConditionType.UNCONDITIONAL:
            pass
        else:
            assert_never(normalized_condition_type)
        if canvas_size is not None and canvas_size != (
            self.config.canvas_size,
            self.config.canvas_size,
        ):
            msg = "LayoutGPT uses the square canvas_size configured on the agent."
            raise ValueError(msg)
        output = self.run_sync(
            prompt,
            train_examples=train_examples,
            model=model,
            seed=seed,
            generator=generator,
            query_embedding=query_embedding,
            example_embeddings=example_embeddings,
            model_settings=model_settings,
        ).to_layout_generation_output()
        if not return_intermediates:
            output.intermediates = None
        if normalized_output_type is OutputType.DICT:
            return {
                "bbox": output.bbox,
                "labels": output.labels,
                "mask": output.mask,
                "id2label": output.id2label,
                "sequences": output.sequences,
                "scores": output.scores,
                "trajectory": output.trajectory,
                "intermediates": output.intermediates,
            }
        if normalized_output_type is OutputType.DATACLASS:
            return output
        assert_never(normalized_output_type)

    generate = __call__

    def _select_examples(
        self,
        prompt: str,
        *,
        train_examples: Sequence[LayoutExample],
        seed: int | None = None,
        generator: torch.Generator | None = None,
        query_embedding: EmbeddingProvider | None,
        example_embeddings: Sequence[Sequence[float]] | None,
    ) -> list[LayoutExample]:
        if self.config.icl_type is ICLType.FIXED_RANDOM:
            selection_seed = (
                int(generator.initial_seed())
                if generator is not None
                else seed
                if seed is not None
                else self.config.fixed_random_seed
            )
            return select_fixed_random(
                train_examples, k=self.config.k, seed=selection_seed
            )
        if self.config.icl_type is ICLType.K_SIMILAR:
            if query_embedding is None or example_embeddings is None:
                msg = "k-similar selection requires query_embedding and example_embeddings"
                raise ValueError(msg)
            return select_k_similar(
                train_examples,
                query=prompt,
                k=self.config.k,
                query_embedding=query_embedding,
                example_embeddings=example_embeddings,
            )
        assert_never(self.config.icl_type)


def _messages_to_text(messages: str | list[dict[str, str]]) -> str:
    if isinstance(messages, str):
        return messages
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
