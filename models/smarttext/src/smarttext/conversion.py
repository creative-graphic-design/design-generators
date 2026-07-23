"""Checkpoint conversion helpers for SmartText."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import torch

from .configuration_smarttext import SmartTextConfig
from .image_processing_smarttext import SmartTextImageProcessor
from .modeling_basnet import SmartTextBASNet
from .modeling_smarttext import SmartTextScorer
from .pipeline_smarttext import SmartTextPipeline
from .processing_smarttext import SmartTextProcessor

CONVERSION_REPORT: Final[str] = "conversion_report.json"


def strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove ``DataParallel`` ``module.`` prefixes.

    Args:
        state_dict: Raw PyTorch state dict.

    Returns:
        State dict with prefixes removed.

    Examples:
        >>> strip_module_prefix({"module.a": torch.tensor(1)})["a"].item()
        1
    """
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def file_sha256(path: Path) -> str:
    """Compute SHA256 for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_original_checkpoints(
    *,
    smt_checkpoint: Path,
    basnet_checkpoint: Path,
    output_dir: Path,
    config: SmartTextConfig,
) -> dict[str, object]:
    """Convert raw SmartText checkpoints into a pipeline directory.

    Args:
        smt_checkpoint: Raw SMT scorer checkpoint.
        basnet_checkpoint: Raw BASNet checkpoint.
        output_dir: Output pipeline directory.
        config: SmartText config.

    Returns:
        Conversion report dictionary.

    Raises:
        RuntimeError: If converted keys do not strictly match the target models.
    """
    scorer = SmartTextScorer(config)
    saliency_model = SmartTextBASNet(config)
    smt_state = strip_module_prefix(torch.load(smt_checkpoint, map_location="cpu"))
    basnet_state = strip_module_prefix(
        torch.load(basnet_checkpoint, map_location="cpu")
    )
    scorer_missing, scorer_unexpected = scorer.load_state_dict(smt_state, strict=False)
    basnet_missing, basnet_unexpected = saliency_model.load_state_dict(
        basnet_state, strict=False
    )
    report = {
        "smt_checkpoint": str(smt_checkpoint),
        "basnet_checkpoint": str(basnet_checkpoint),
        "smt_sha256": file_sha256(smt_checkpoint),
        "basnet_sha256": file_sha256(basnet_checkpoint),
        "smt_source_key_count": len(smt_state),
        "basnet_source_key_count": len(basnet_state),
        "scorer_missing_keys": list(scorer_missing),
        "scorer_unexpected_keys": list(scorer_unexpected),
        "basnet_missing_keys": list(basnet_missing),
        "basnet_unexpected_keys": list(basnet_unexpected),
        "roi_rod_alignment": "PyTorch port of vendor RoI/RoD forward kernels; strict parity should be checked against compiled vendor references",
    }
    if scorer_missing or scorer_unexpected or basnet_missing or basnet_unexpected:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    processor = SmartTextProcessor(
        image_processor=SmartTextImageProcessor.from_config(config),
        config=config,
    )
    pipeline = SmartTextPipeline(
        scorer=scorer,
        saliency_model=saliency_model,
        processor=processor,
        config=config,
    )
    pipeline.save_pretrained(output_dir)
    (output_dir / CONVERSION_REPORT).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
