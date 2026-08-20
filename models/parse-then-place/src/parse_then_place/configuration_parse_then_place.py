"""Configuration for Parse-Then-Place composite checkpoints."""

from __future__ import annotations

from typing import Literal

from transformers import PretrainedConfig

from .labels import (
    Stage2Mode,
    canvas_size_for_dataset,
    id2label_for_dataset,
    normalize_dataset_name,
    normalize_stage2_mode,
)


class ParseThenPlaceConfig(PretrainedConfig):
    """Stores Parse-Then-Place dataset and generation defaults."""

    model_type = "parse-then-place"

    def __init__(
        self,
        dataset_name: str = "rico",
        stage2_mode: Stage2Mode | str = Stage2Mode.finetune,
        parser_model_name: str = "google/t5-v1_1-base",
        parser_generation_max_length: int = 600,
        placement_generation_max_length: int = 500,
        temperature: float = 0.7,
        num_return_sequences: int = 5,
        canvas_size: tuple[int, int] | list[int] | None = None,
        id2label: dict[int | str, str] | None = None,
        parser_subfolder: str = "semantic_parser",
        placement_subfolder: str = "placement",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = 0,
        is_encoder_decoder: bool = True,
        transformers_version: str | None = None,
        architectures: list[str] | None = None,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        dtype: str | None = None,
        torch_dtype: str | None = None,
        chunk_size_feed_forward: int = 0,
        problem_type: Literal[
            "regression", "single_label_classification", "multi_label_classification"
        ]
        | None = None,
        name_or_path: str = "",
        _commit_hash: str | None = None,
        attn_implementation: str | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the composite checkpoint configuration."""
        dataset = normalize_dataset_name(dataset_name)
        mode = normalize_stage2_mode(stage2_mode)

        self.dataset_name = str(dataset)
        self.stage2_mode = str(mode)
        self.parser_model_name = parser_model_name
        self.parser_generation_max_length = parser_generation_max_length
        self.placement_generation_max_length = placement_generation_max_length
        self.temperature = temperature
        self.num_return_sequences = num_return_sequences
        self.canvas_size = tuple(canvas_size or canvas_size_for_dataset(dataset))
        self.parser_subfolder = parser_subfolder
        self.placement_subfolder = placement_subfolder

        label_map = id2label or id2label_for_dataset(dataset)
        normalized_id2label = {int(key): str(value) for key, value in label_map.items()}
        label2id = {value: key for key, value in normalized_id2label.items()}
        _ = kwargs.pop("id2label", None)
        _ = kwargs.pop("label2id", None)

        super().__init__(
            transformers_version=transformers_version,
            architectures=architectures,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            dtype=dtype or torch_dtype,
            chunk_size_feed_forward=chunk_size_feed_forward,
            is_encoder_decoder=is_encoder_decoder,
            id2label=normalized_id2label,
            label2id=label2id,
            problem_type=problem_type,
        )
        # Transformers v5 keeps model-specific token fields on the subclass;
        # only common configuration fields belong in the base call.
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.name_or_path = name_or_path
        self._commit_hash = _commit_hash
        self._attn_implementation = attn_implementation
        for key, value in kwargs.items():
            setattr(self, key, value)
