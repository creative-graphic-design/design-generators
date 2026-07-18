"""Checkpoint conversion helpers for LayouSyn."""

from __future__ import annotations

from pathlib import Path
import json

import torch

from .configuration_layousyn import LayouSynConfig
from .modeling_layousyn import LayouSynDiTModel, convert_vendor_state_dict
from .pipeline_layousyn import LayouSynPipeline
from .processing_layousyn import LayouSynProcessor
from .scheduling_layousyn import LayouSynScheduler


def convert_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    variant_name: str,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> LayouSynPipeline:
    """Convert a vendor checkpoint into a local Diffusers pipeline.

    Args:
        checkpoint_path: Vendor ``.pt`` checkpoint path.
        config_path: Vendor JSON config path.
        output_dir: Local output directory.
        variant_name: Human-readable checkpoint variant metadata.
        push_to_hub: Reserved; ordinary implementation PRs must leave this
            false.
        hub_repo_id: Optional Hub repository id for future publishing.

    Returns:
        Saved pipeline instance.

    Raises:
        ValueError: If Hub push is requested from this implementation helper.
    """
    if push_to_hub:
        raise ValueError("Hub push is disabled for implementation PR conversion")
    del hub_repo_id
    config = LayouSynConfig.from_vendor_json(config_path)
    model = LayouSynDiTModel(
        in_channels=config.in_channels,
        max_in_len=config.max_in_len,
        concept_in_channels=config.concept_in_channels,
        y_in_channels=config.y_in_channels,
        max_y_len=config.max_y_len,
        model_name=config.model_name,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
    )
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw["ema"] if isinstance(raw, dict) and "ema" in raw else raw
    missing, unexpected = model.load_state_dict(
        convert_vendor_state_dict(state_dict), strict=False
    )
    if unexpected:
        raise ValueError(
            f"Unexpected checkpoint keys for {variant_name}: {unexpected[:5]}"
        )
    scheduler = LayouSynScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_schedule=config.noise_schedule,
        alpha_scale=config.scale,
    )
    processor = LayouSynProcessor(
        layout_type=config.layout_type,
        max_in_len=config.max_in_len,
        max_y_len=config.max_y_len or 120,
        concept_in_channels=config.concept_in_channels,
        y_in_channels=config.y_in_channels or 768,
    )
    pipe = LayouSynPipeline(model=model, scheduler=scheduler, processor=processor)
    out = Path(output_dir)
    pipe.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "variant_name": variant_name,
                "missing_keys": list(missing),
                "license": "cc-by-nc-4.0",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return pipe
