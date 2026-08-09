"""Checkpoint conversion helpers for DLT."""

from __future__ import annotations

from pathlib import Path

from .configuration_dlt import DLTConfig
from .modeling_dlt import DLT
from .pipeline_dlt import DLTPipeline
from .processing_dlt import DLTProcessor
from .scheduling_dlt import DLTJointDiffusionScheduler


def build_pipeline(config: DLTConfig) -> DLTPipeline:
    """Build a randomly initialized DLT pipeline from a config.

    Args:
        config: DLT configuration.

    Returns:
        Pipeline with model, scheduler, and processor components.
    """
    model = DLT(
        categories_num=config.categories_num,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_r=config.dropout_r,
        activation=config.activation,
        cond_emb_size=config.cond_emb_size,
        cat_emb_size=config.cat_emb_size,
    )
    scheduler = DLTJointDiffusionScheduler(
        alpha=0.0,
        seq_max_length=config.max_num_comp,
        discrete_features_names=[("cat", config.categories_num)],
        num_discrete_steps=[config.num_discrete_steps],
        num_train_timesteps=config.num_cont_timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type="sample",
        clip_sample=False,
    )
    processor = DLTProcessor(
        dataset=config.dataset_name,
        labels=tuple(config.id2label.values()),
        max_num_comp=config.max_num_comp,
    )
    return DLTPipeline(
        model=model, scheduler=scheduler, config=config, processor=processor
    )


def convert_save_pretrained_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    config: DLTConfig,
) -> DLTPipeline:
    """Convert an original DLT ``save_pretrained`` directory into a pipeline.

    Args:
        checkpoint_dir: Directory containing the original DLT model files.
        output_dir: Destination directory for the converted pipeline.
        config: Dataset and scheduler metadata for the checkpoint.

    Returns:
        The saved converted pipeline.

    Raises:
        RuntimeError: If the checkpoint does not match the configured model.
    """
    pipe = build_pipeline(config)
    loaded_model = DLT.from_pretrained(checkpoint_dir)
    missing, unexpected = pipe.model.load_state_dict(
        loaded_model.state_dict(), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"Missing keys: {missing}; unexpected keys: {unexpected}")
    pipe.save_pretrained(output_dir)
    return pipe
