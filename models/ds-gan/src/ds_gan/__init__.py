"""Transformers-style DS-GAN components for PosterLayout generation."""

from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_ds_gan import DSGANConfig, default_ds_gan_config
from .conversion import convert_vendor_state_dict
from .modeling_ds_gan import DSGANModel, DSGANModelOutput, random_initial_layout
from .pipeline_ds_gan import DSGANPipeline, OutputType
from .processing_ds_gan import (
    DSGANProcessor,
    annotations_from_pku_example,
    processor_for_dataset,
)

__all__ = [
    "ConditionType",
    "DSGANConfig",
    "DSGANModel",
    "DSGANModelOutput",
    "DSGANPipeline",
    "DSGANProcessor",
    "LayoutGenerationOutput",
    "OutputType",
    "convert_vendor_state_dict",
    "default_ds_gan_config",
    "annotations_from_pku_example",
    "processor_for_dataset",
    "random_initial_layout",
]
