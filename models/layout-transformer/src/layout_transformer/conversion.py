"""Conversion helpers for original LT-Net checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from .configuration_layout_transformer import LayoutTransformerConfig
from .modeling_layout_transformer import LayoutTransformerForLayoutGeneration
from .processing_layout_transformer import LayoutTransformerProcessor
from .vendor_state_dict import load_original_state_dict


def convert_original_checkpoint(
    *,
    checkpoint_path: str | Path,
    cfg_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    dataset_name: Literal["coco", "vg_msdn"],
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    strict: bool = True,
) -> None:
    """Convert an original LT-Net checkpoint into local HF-style files.

    Args:
        checkpoint_path: Vendor ``.pth`` checkpoint containing ``state_dict`` or
            a raw state dict.
        cfg_path: Vendor YAML config path. Stored in metadata for traceability.
        vocab_path: JSON id-to-token vocabulary exported from
            ``object_pred_idx_to_name.pkl``.
        output_dir: Directory to write converted model/processor files.
        dataset_name: Dataset identifier for the converted checkpoint.
        push_to_hub: Reserved publish flag; implementation PRs keep it false.
        hub_model_id: Optional Hub repo id used only when publishing is enabled.
        strict: Whether checkpoint keys must exactly match the converted model.

    Raises:
        NotImplementedError: If Hub upload is requested from this helper.

    Examples:
        >>> convert_original_checkpoint  # doctest: +ELLIPSIS
        <function convert_original_checkpoint at ...>
    """
    if push_to_hub or hub_model_id is not None:
        raise NotImplementedError(
            "Hub upload is intentionally not part of PR conversion"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Path(vocab_path).open() as f:
        vocab = json.load(f)
    token_count = len(vocab)
    config = LayoutTransformerConfig(
        dataset_name=dataset_name,
        vocab_size=token_count,
        refine=True,
        use_vendor_modules=True,
    )
    model = LayoutTransformerForLayoutGeneration(config)
    state_dict = load_original_state_dict(checkpoint_path)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "cfg_path": str(cfg_path),
        "vocab_path": str(vocab_path),
        "dataset_name": dataset_name,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_vendor_key_mapping": strict,
    }
    model.save_pretrained(out)
    processor = LayoutTransformerProcessor.from_config(
        dataset_name=dataset_name,
        max_sequence_length=config.max_sequence_length,
        id2label=cast(dict[int, str], config.id2label),
        relation_id2label=config.relation_id2label,
    )
    processor.save_pretrained(out)
    with (out / "conversion_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
