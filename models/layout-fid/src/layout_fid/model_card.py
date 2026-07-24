"""Model-card helpers for layout FID evaluator checkpoints."""

from __future__ import annotations

from .configuration_layout_fid import LayoutFIDConfig


def model_card_metadata(config: LayoutFIDConfig, *, hub_id: str) -> dict[str, object]:
    """Return model-card metadata for a converted layout FID checkpoint."""
    return {
        "hub_id": hub_id,
        "library_name": "transformers",
        "pipeline_tag": "other",
        "tags": ["layout-generation", "layout-evaluation", "fid"],
        "datasets": [
            (
                "creative-graphic-design/Rico"
                if config.dataset_name == "rico25"
                else "creative-graphic-design/PubLayNet"
            )
        ],
    }
