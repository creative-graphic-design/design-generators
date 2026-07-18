"""Pipeline wrapper for Parse-Then-Place composite checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast  # noqa: TID251 - Transformers generate/save_pretrained stubs require Any casts.

import torch
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, set_seed  # ty: ignore[possibly-missing-import]
from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_parse_then_place import ParseThenPlaceConfig
from .processing_parse_then_place import ParseThenPlaceProcessor


class ParseThenPlacePipeline:
    """Compose standard seq2seq parser and placement models."""

    config: ParseThenPlaceConfig
    parser: PreTrainedModel | None
    placement: PreTrainedModel | None
    processor: ParseThenPlaceProcessor

    def __init__(
        self,
        config: ParseThenPlaceConfig,
        processor: ParseThenPlaceProcessor,
        *,
        parser: PreTrainedModel | None = None,
        placement: PreTrainedModel | None = None,
    ) -> None:
        """Initialize the composite pipeline."""
        self.config = config
        self.processor = processor
        self.parser = parser
        self.placement = placement

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        parser: PreTrainedModel | None = None,
        placement: PreTrainedModel | None = None,
        processor: ParseThenPlaceProcessor | None = None,
        local_files_only: bool = False,
        config: ParseThenPlaceConfig | PretrainedConfig | None = None,
    ) -> "ParseThenPlacePipeline":
        """Load a composite pipeline from a root directory."""
        if config is None:
            parsed_config = ParseThenPlaceConfig.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )
        elif isinstance(config, ParseThenPlaceConfig):
            parsed_config = config
        elif isinstance(config, PretrainedConfig):
            parsed_config = ParseThenPlaceConfig.from_dict(config.to_dict())
        else:
            raise TypeError("config must be a ParseThenPlaceConfig")
        root = Path(pretrained_model_name_or_path)
        if parser is None:
            parser_path = root / parsed_config.parser_subfolder
            if (parser_path / "config.json").exists():
                parser = AutoModelForSeq2SeqLM.from_pretrained(
                    parser_path,
                    local_files_only=local_files_only,
                )
        if placement is None:
            placement = AutoModelForSeq2SeqLM.from_pretrained(
                root / parsed_config.placement_subfolder,
                local_files_only=local_files_only,
            )
        if processor is None:
            processor = ParseThenPlaceProcessor.from_pretrained(
                root,
                local_files_only=local_files_only,
            )
        return cls(
            config=parsed_config,
            processor=processor,
            parser=parser,
            placement=placement,
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
        **kwargs: object,
    ) -> None:
        """Save composite config, processor, and available stage models."""
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(root)
        self.processor.save_pretrained(root)
        if self.parser is not None:
            cast(Any, self.parser).save_pretrained(
                root / self.config.parser_subfolder,
                is_main_process=is_main_process,
                **kwargs,
            )
        if self.placement is not None:
            cast(Any, self.placement).save_pretrained(
                root / self.config.placement_subfolder,
                is_main_process=is_main_process,
                **kwargs,
            )

    @torch.no_grad()
    def parse(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        generation_max_length: int | None = None,
        **generate_kwargs: object,
    ) -> torch.Tensor:
        """Generate logical-form token ids with the parser stage."""
        if self.parser is None:
            raise ValueError("Parser stage is not loaded")
        generated = cast(Any, self.parser).generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generation_max_length
            or self.config.parser_generation_max_length,
            **generate_kwargs,
        )
        return cast(torch.Tensor, generated)

    @torch.no_grad()
    def place(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        generation_max_length: int | None = None,
        num_return_sequences: int | None = None,
        temperature: float | None = None,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
        **generate_kwargs: object,
    ) -> torch.Tensor:
        """Generate layout token ids with the placement stage."""
        if self.placement is None:
            raise ValueError("Placement stage is not loaded")
        if generator is not None:
            generate_kwargs["generator"] = generator
        generated = cast(Any, self.placement).generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generation_max_length
            or self.config.placement_generation_max_length,
            num_return_sequences=num_return_sequences
            or self.config.num_return_sequences,
            temperature=temperature or self.config.temperature,
            do_sample=do_sample,
            **generate_kwargs,
        )
        return cast(torch.Tensor, generated)

    def __call__(
        self,
        *,
        prompt: str | Sequence[str] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.text,
        labels: torch.Tensor | list[object] | None = None,
        bbox: torch.Tensor | list[object] | None = None,
        mask: torch.Tensor | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        num_return_sequences: int | None = None,
        temperature: float | None = None,
        output_candidate: Literal["first", "all", "best"] = "first",
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        layout_text: str | list[str] | list[list[str]] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate a layout from natural-language text."""
        _ = (
            batch_size,
            labels,
            bbox,
            mask,
            num_elements,
            box_format,
            normalized,
            canvas_size,
            num_inference_steps,
        )
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.text:
            raise NotImplementedError(
                "Parse-Then-Place only supports condition_type='text'"
            )
        if layout_text is not None:
            layout_items = (
                [layout_text] if isinstance(layout_text, str) else layout_text
            )
            return self.processor.layout_text_to_output(
                layout_items,
                output_candidate=output_candidate,
                output_type=output_type,
                return_intermediates=return_intermediates,
            )
        if prompt is None:
            raise ValueError("prompt is required for Parse-Then-Place generation")
        if generator is None and seed is not None:
            set_seed(seed)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        parser_inputs = self.processor(prompts)
        if "input_ids" not in parser_inputs:
            raise ValueError("processor requires parser_tokenizer for model inference")
        parser_ids = self.parse(
            parser_inputs["input_ids"],
            attention_mask=parser_inputs.get("attention_mask"),
            generation_max_length=self.config.parser_generation_max_length,
        )
        value_maps = cast(list[dict[str, str] | None], parser_inputs.get("value_maps"))
        logical_forms = self.processor.postprocess_ir(
            parser_ids,
            value_maps=value_maps,
        )
        placement_inputs = self.processor.ir_to_placement_inputs(logical_forms)
        placement_encoded = self.processor.encode_placement_inputs(placement_inputs)
        if "input_ids" not in placement_encoded:
            raise ValueError(
                "processor requires placement_tokenizer for model inference"
            )
        return_sequences = num_return_sequences or self.config.num_return_sequences
        placement_ids = self.place(
            placement_encoded["input_ids"],
            attention_mask=placement_encoded.get("attention_mask"),
            num_return_sequences=return_sequences,
            temperature=temperature,
            generator=generator,
        )
        grouped = self.processor.decode_layout_sequences(
            placement_ids,
            batch_size=len(prompts),
            num_return_sequences=return_sequences,
        )
        output = self.processor.layout_text_to_output(
            grouped,
            output_candidate=output_candidate,
            output_type="dataclass",
            return_intermediates=True,
        )
        if isinstance(output, LayoutGenerationOutput):
            intermediates = (
                dict(output.intermediates)
                if isinstance(output.intermediates, dict)
                else {}
            )
            if return_intermediates:
                intermediates.update(
                    {
                        "prompt": prompts,
                        "logical_forms": logical_forms,
                        "placement_inputs": placement_inputs,
                    }
                )
            output.intermediates = intermediates if return_intermediates else None
        if output_type == "dict":
            return dict(output)
        return output

    generate = __call__
