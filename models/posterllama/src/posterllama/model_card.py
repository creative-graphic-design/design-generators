"""Model-card metadata for PosterLlama recipe artifacts."""

from __future__ import annotations

from .configuration_posterllama import PosterLlamaConfig


def model_card_metadata(config: PosterLlamaConfig, *, hub_id: str) -> dict[str, object]:
    """Return model-card metadata for a PosterLlama recipe artifact."""
    _ = config
    return {
        "hub_id": hub_id,
        "library_name": "transformers",
        "pipeline_tag": "image-text-to-text",
        "tags": ["layout-generation", "poster-generation", "posterllama"],
        "datasets": ["creative-graphic-design/CGL"],
    }
