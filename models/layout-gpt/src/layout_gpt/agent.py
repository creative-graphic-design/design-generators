"""Provider-agnostic Pydantic AI wrapper for LayoutGPT."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Final, assert_never, cast

import torch
from jaxtyping import Bool, Float, Int
from laygen.agents import BaseLayoutAgent, ModelLike
from laygen.common import ConditionType
from laygen.common.bbox import BoxFormat
from laygen.modeling_outputs import LayoutGenerationOutput
from pydantic_ai.settings import ModelSettings

from layout_gpt.enums import ICLType, OutputType, coerce_enum
from layout_gpt.exemplars import (
    EmbeddingProvider,
    LayoutExample,
    select_fixed_random,
    select_k_similar,
)
from layout_gpt.parser import parse_layout_text
from layout_gpt.prompts import (
    ChatMessage,
    TokenCounter,
    default_token_counter,
    form_prompt_for_chatgpt,
    form_prompt_for_gpt3,
)
from layout_gpt.schema import LayoutGPTConfig, LayoutGPTOutput, RawLayoutResponse
from layout_gpt.types import LayoutGPTOutputDict

DEFAULT_MODEL_ENV_VAR: Final[str] = "LAYOUT_GPT_MODEL"
INSTRUCTIONS: Final[str] = (
    "You are LayoutGPT. Return CSS layout lines for the requested prompt. "
    "Each line must be `object {height: ?px; width: ?px; top: ?px; left: ?px; }`."
)
SUPPORTED_CONDITION_TYPES: Final[tuple[ConditionType, ...]] = (
    ConditionType.text,
    ConditionType.unconditional,
)


def build_agent(model: ModelLike = None) -> object:
    """Build a Pydantic AI agent with provider selected by argument or env."""
    return BaseLayoutAgent[RawLayoutResponse](
        model=model,
        model_env_var=DEFAULT_MODEL_ENV_VAR,
        raw_response_type=RawLayoutResponse,
        instructions=INSTRUCTIONS,
    ).agent


class LayoutGPTAgent(BaseLayoutAgent[RawLayoutResponse]):
    """High-level LayoutGPT runner that ports released prompt and parse strategy."""

    def __init__(
        self,
        *,
        model: ModelLike = None,
        config: LayoutGPTConfig,
        token_counter: TokenCounter = default_token_counter,
    ) -> None:
        """Initialize the runner with provider and prompt configuration."""
        super().__init__(
            model=model,
            model_env_var=DEFAULT_MODEL_ENV_VAR,
            raw_response_type=RawLayoutResponse,
            instructions=INSTRUCTIONS,
        )
        self.config = config
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
    ) -> tuple[str | list[ChatMessage], list[LayoutExample]]:
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
        raw = self.run_raw_sync(
            model_prompt,
            model=model,
            model_settings=model_settings
            or ModelSettings(
                temperature=self.config.temperature, top_p=self.config.top_p
            ),
        )
        response_text = self.repair_response_text(raw.text)
        items = parse_layout_text(response_text, canvas_size=self.config.canvas_size)
        id2label = {
            index: label
            for index, label in enumerate(dict.fromkeys(item.label for item in items))
        }
        return LayoutGPTOutput(
            prompt=prompt,
            canvas_size=self.config.canvas_size,
            items=items,
            raw_text=response_text,
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
        condition_type: str | ConditionType = ConditionType.text,
        labels: Int[torch.Tensor, "batch elements"] | list[object] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"] | list[object] | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | list[object] | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: str | BoxFormat = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: str | OutputType = OutputType.dataclass,
        return_intermediates: bool = False,
        model: ModelLike = None,
        query_embedding: EmbeddingProvider | None = None,
        example_embeddings: Sequence[Sequence[float]] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> LayoutGenerationOutput | LayoutGPTOutputDict:
        """Generate a layout through the common public generation surface."""
        del labels, bbox, mask, num_elements, normalized, num_inference_steps
        normalized_condition_type, normalized_box_format = (
            self.validate_generation_request(
                batch_size=batch_size,
                condition_type=condition_type,
                box_format=box_format,
                canvas_size=canvas_size,
                configured_canvas_size=self.config.canvas_size,
                supported_condition_types=SUPPORTED_CONDITION_TYPES,
            )
        )
        normalized_output_type = coerce_enum(output_type, OutputType)
        del normalized_box_format
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
        if normalized_output_type is OutputType.dict:
            return cast(LayoutGPTOutputDict, self.output_to_dict(output))
        if normalized_output_type is OutputType.dataclass:
            return output
        assert_never(normalized_output_type)

    generate = __call__

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        """Persist prompt and parser configuration without provider state."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "layout_gpt_config.json").write_text(
            json.dumps(self.config.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        model: ModelLike = None,
        token_counter: TokenCounter = default_token_counter,
    ) -> "LayoutGPTAgent":
        """Load saved LayoutGPT prompt and parser configuration."""
        path = Path(pretrained_model_name_or_path) / "layout_gpt_config.json"
        config_data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model=model,
            config=LayoutGPTConfig(**config_data),
            token_counter=token_counter,
        )

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
        if self.config.icl_type is ICLType.fixed_random:
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
        if self.config.icl_type is ICLType.k_similar:
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
