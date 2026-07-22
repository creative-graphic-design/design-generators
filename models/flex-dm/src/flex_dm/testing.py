"""Testing helpers for Flex-DM package tests."""

from __future__ import annotations

from .configuration_flex_dm import FlexDmConfig
from .data_specs import build_column_specs
from .modeling_flex_dm import FlexDmForMaskedDocumentModeling
from .pipeline_flex_dm import FlexDmPipeline
from .processing_flex_dm import FlexDmProcessor


def tiny_config() -> FlexDmConfig:
    """Return a small Flex-DM config for CPU tests."""
    input_columns = build_column_specs(dataset_name="crello", vocabulary={})
    for key, column in input_columns.items():
        if (
            column["is_sequence"]
            and column["type"] == "categorical"
            and key not in {"left", "top", "width", "height"}
        ):
            column["input_dim"] = min(int(column["input_dim"] or 4), 4)
        if key in {"image_embedding", "text_embedding"}:
            column["shape"] = (4,)
    return FlexDmConfig(
        dataset_name="crello",
        id2label={0: "coloredBackground", 1: "imageElement", 2: "textElement"},
        input_columns=input_columns,
        max_seq_length=3,
        latent_dim=16,
        num_blocks=1,
        dropout=0.0,
    )


def tiny_pipeline() -> FlexDmPipeline:
    """Return a small random-weight pipeline."""
    config = tiny_config()
    return FlexDmPipeline(
        model=FlexDmForMaskedDocumentModeling(config),
        processor=FlexDmProcessor.from_config(config),
    )
