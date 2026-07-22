"""Model-card helpers for SmartText."""

from __future__ import annotations

from collections.abc import Mapping

from laygen.common.model_card import build_layout_model_card


def build_smarttext_model_card(
    *,
    hub_id: str = "creative-graphic-design/smarttext-smt",
    parity_results: Mapping[str, object] | None = None,
) -> str:
    """Render a SmartText Hub model card.

    Args:
        hub_id: Target Hub repository id.
        parity_results: Optional parity result payload.

    Returns:
        Markdown model-card text.
    """
    card = build_layout_model_card(
        model_id=hub_id,
        model_name="SmartText",
        dataset_ids=[],
        license="other",
        library_name="transformers",
        pipeline_tag="other",
        tags=[
            "smarttext",
            "text-placement",
            "poster-generation",
            "content-image",
            "layout-generation",
        ],
        model_details="SmartText places text regions on natural images using saliency, candidate generation, and candidate scoring.",
        intended_uses="Content-aware text placement research and reproducibility.",
        limitations="Converted weight redistribution and upstream license provenance require review before publication.",
        how_to_use=(
            "from smarttext import SmartTextPipeline\n"
            f"pipe = SmartTextPipeline.from_pretrained({hub_id!r})\n"
            "output = pipe(image, prompt='Title', condition_type='content_image')"
        ),
        training_data="Original SmartText training data is not distributed in this repository.",
        parity_metrics=[],
        citation_bibtex="@article{li2021smarttext, title={Harmonious Textual Layout Generation over Natural Images via Deep Aesthetics Learning}}",
        original_implementation_url="https://github.com/chenqi008/SmartText",
        model_summary="SmartText content-image text placement checkpoint.",
        results_summary=f"Parity results: {dict(parity_results or {})}",
    )
    return card.text
