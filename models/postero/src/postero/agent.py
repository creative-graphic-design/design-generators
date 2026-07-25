"""Provider-agnostic Pydantic AI wrapper for PosterO."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final, assert_never, cast

import torch
from jaxtyping import Bool, Float, Int
from laygen.agents import BaseLayoutAgent, ModelLike
from laygen.common import (
    BoxFormat,
    ConditionType,
    normalize_box_format,
    normalize_condition_type,
)
from laygen.modeling_outputs import LayoutGenerationOutput
from pydantic_ai.settings import ModelSettings

from postero.config import PosterOConfig
from postero.enums import OutputType, coerce_enum
from postero.exemplars import select_exemplars
from postero.parser import parse_svg_response
from postero.prompts import build_prompt
from postero.records import PosterORecord
from postero.schemas import PosterOOutput, RawPosterOResponse, merged_output

DEFAULT_MODEL_ENV_VAR: Final[str] = "POSTERO_MODEL"
INSTRUCTIONS: Final[str] = (
    "You are PosterO. Return SVG rectangles for content-aware poster layouts."
)
SUPPORTED_CONDITION_TYPES: Final[tuple[ConditionType, ...]] = (
    ConditionType.content_image,
    ConditionType.retrieval,
)


class PosterOAgent(BaseLayoutAgent[RawPosterOResponse]):
    """High-level PosterO runner for prompt construction, parsing, and retries.

    Args:
        model: Optional Pydantic AI model object or provider model id.
        config: Runtime prompt and parser configuration.

    Raises:
        ValueError: If config options are unsupported.

    Examples:
        >>> from pydantic_ai.models.test import TestModel
        >>> from postero.config import PosterOConfig
        >>> agent = PosterOAgent(model=TestModel(custom_output_args={"text": "<svg></svg>"}), config=PosterOConfig())
        >>> isinstance(agent, PosterOAgent)
        True
    """

    def __init__(self, *, model: ModelLike = None, config: PosterOConfig) -> None:
        """Create a PosterO runner from explicit config."""
        self.config = config
        super().__init__(
            model=model,
            model_env_var=DEFAULT_MODEL_ENV_VAR,
            raw_response_type=RawPosterOResponse,
            instructions=INSTRUCTIONS,
        )

    def build_prompt(
        self,
        query_record: PosterORecord,
        *,
        candidate_records: Sequence[PosterORecord],
        labels: Sequence[int | str] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[str, list[PosterORecord]]:
        """Select exemplars and build the provider prompt.

        Args:
            query_record: Query poster record.
            candidate_records: Candidate exemplar records.
            labels: Optional labels to allocate.
            seed: Optional deterministic seed for random selection.
            generator: Optional torch generator. Takes precedence over ``seed``.

        Returns:
            Prompt text and selected exemplars.
        """
        exemplars = select_exemplars(
            query_record,
            candidate_records,
            config=self.config,
            seed=seed,
            generator=generator,
        )
        return (
            build_prompt(query_record, exemplars, config=self.config, labels=labels),
            exemplars,
        )

    def run_sync(
        self,
        query_record: PosterORecord,
        *,
        candidate_records: Sequence[PosterORecord],
        labels: Sequence[int | str] | None = None,
        model: ModelLike = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        model_settings: ModelSettings | None = None,
    ) -> PosterOOutput:
        """Run the configured provider until enough valid SVGs are parsed.

        Args:
            query_record: Query poster record.
            candidate_records: Candidate exemplar records.
            labels: Optional labels to allocate.
            model: Optional per-call provider override.
            seed: Optional deterministic seed.
            generator: Optional torch generator. Takes precedence over ``seed``.
            model_settings: Optional provider sampling settings.

        Returns:
            Parsed PosterO output.

        Raises:
            RuntimeError: If no valid response is parsed within ``num_return``.
        """
        prompt, exemplars = self.build_prompt(
            query_record,
            candidate_records=candidate_records,
            labels=labels,
            seed=seed,
            generator=generator,
        )
        selected_ids = [record.id for record in exemplars]
        outputs: list[PosterOOutput] = []
        errors: list[str] = []
        for attempt in range(1, self.config.num_return + 1):
            raw = self.run_raw_sync(
                prompt,
                model=model,
                model_settings=model_settings
                or ModelSettings(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                ),
            )
            try:
                elements, diagnostics = parse_svg_response(raw.text, config=self.config)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            outputs.append(
                PosterOOutput(
                    prompt=prompt,
                    raw_text=raw.text,
                    elements=elements,
                    id2label=self.config.id2label or {},
                    canvas_size=self.config.canvas_size,
                    selected_exemplar_ids=selected_ids,
                    attempts=attempt,
                    parser_diagnostics=diagnostics,
                    parser_errors=list(errors),
                )
            )
            valid_count = sum(len(output.elements) for output in outputs)
            if valid_count >= self.config.n_valid_layouts:
                return merged_output(outputs, config=self.config)
        if outputs:
            return merged_output(outputs, config=self.config)
        msg = "PosterO could not parse a valid SVG response"
        raise RuntimeError(msg)

    def __call__(
        self,
        *,
        query_record: PosterORecord,
        candidate_records: Sequence[PosterORecord],
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: str | ConditionType = ConditionType.content_image,
        labels: Int[torch.Tensor, "batch elements"] | list[int | str] | None = None,
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
        model_settings: ModelSettings | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate a poster layout through the shared public surface.

        Args:
            query_record: Query poster record with content-aware regions.
            candidate_records: Candidate records for retrieval.
            batch_size: Shared API batch size. PosterO supports ``1``.
            seed: Optional deterministic seed.
            generator: Optional torch generator. Takes precedence over ``seed``.
            condition_type: ``content_image`` or ``retrieval``.
            labels: Optional labels to allocate.
            bbox: Accepted for shared interface compatibility.
            mask: Accepted for shared interface compatibility.
            num_elements: Accepted for shared interface compatibility.
            box_format: Public input box format name.
            normalized: Accepted for shared interface compatibility.
            canvas_size: Optional canvas override; must match config.
            num_inference_steps: Accepted for shared interface compatibility.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to include prompt/parser metadata.
            model: Optional per-call provider override.
            model_settings: Optional provider settings override.

        Returns:
            Shared layout output dataclass or dictionary.

        Raises:
            ValueError: If shared arguments request unsupported behavior.
            RuntimeError: If provider responses cannot be parsed.
        """
        del bbox, mask, num_elements, normalized, num_inference_steps
        self._validate_request(
            batch_size=batch_size,
            condition_type=condition_type,
            box_format=box_format,
            canvas_size=canvas_size,
        )
        normalized_output_type = coerce_enum(output_type, OutputType)
        label_values = _labels_from_public(labels)
        output = self.run_sync(
            query_record,
            candidate_records=candidate_records,
            labels=label_values,
            model=model,
            seed=seed,
            generator=generator,
            model_settings=model_settings,
        ).to_layout_generation_output(return_intermediates=return_intermediates)
        if normalized_output_type is OutputType.dataclass:
            return output
        if normalized_output_type is OutputType.dict:
            return self.output_to_dict(output)
        assert_never(normalized_output_type)

    generate = __call__

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        """Persist prompt and parser configuration without provider state.

        Args:
            save_directory: Target directory for ``postero_config.json``.

        Returns:
            None.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "postero_config.json").write_text(
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
    ) -> "PosterOAgent":
        """Load saved PosterO prompt and parser configuration.

        Args:
            pretrained_model_name_or_path: Directory containing
                ``postero_config.json``.
            model: Replacement provider model.

        Returns:
            Configured PosterO agent.
        """
        path = Path(pretrained_model_name_or_path) / "postero_config.json"
        config_data = json.loads(path.read_text(encoding="utf-8"))
        return cls(model=model, config=PosterOConfig(**config_data))

    def resolve_model(self, model: ModelLike = None) -> ModelLike:
        """Resolve explicit model, environment model, or Pydantic default."""
        return (
            model or os.getenv(DEFAULT_MODEL_ENV_VAR) or os.getenv("PYDANTIC_AI_MODEL")
        )

    def _validate_request(
        self,
        *,
        batch_size: int,
        condition_type: str | ConditionType,
        box_format: str | BoxFormat,
        canvas_size: tuple[int, int] | None,
    ) -> None:
        normalized_condition = normalize_condition_type(condition_type)
        normalize_box_format(box_format)
        if batch_size != 1:
            msg = "PosterO currently supports batch_size=1."
            raise ValueError(msg)
        if normalized_condition not in SUPPORTED_CONDITION_TYPES:
            msg = f"unsupported condition_type for PosterO: {normalized_condition}"
            raise ValueError(msg)
        if canvas_size is not None and canvas_size != self.config.canvas_size:
            msg = "PosterO canvas_size must match the saved prompt config."
            raise ValueError(msg)


def _labels_from_public(
    labels: Int[torch.Tensor, "batch elements"] | list[int | str] | None,
) -> list[int | str] | None:
    if labels is None:
        return None
    if isinstance(labels, torch.Tensor):
        return [int(value) for value in labels.flatten().tolist()]
    return cast(list[int | str], labels)
