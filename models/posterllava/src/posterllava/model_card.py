"""Hub model-card generation helpers for PosterLLaVA."""

from __future__ import annotations

from huggingface_hub import ModelCard

from laygen.common.model_card import build_layout_model_card


def build_posterllava_model_card(
    *,
    model_id: str = "creative-graphic-design/posterllava-v0",
) -> ModelCard:
    """Build a PosterLLaVA Hub model-card draft.

    Args:
        model_id: Planned Hub model id.

    Returns:
        Hugging Face model card object.

    Examples:
        >>> card = build_posterllava_model_card()
        >>> card.data.to_dict()["library_name"]
        'transformers'
    """
    return build_layout_model_card(
        model_id=model_id,
        model_name="PosterLLaVA",
        dataset_ids=[],
        license="other",
        library_name="transformers",
        pipeline_tag="other",
        tags=["layout-generation", "poster-layout", "llava", "multimodal"],
        model_details=(
            "PosterLLaVA is a LLaVA-style multimodal recipe for generating poster "
            "layout JSON from a background image and layout instructions."
        ),
        intended_uses=(
            "Use the package locally with the upstream PosterLLaVA checkpoint to "
            "parse generated JSON layouts into normalized layout tensors."
        ),
        limitations=(
            "The upstream Hugging Face metadata advertises Apache-2.0, while the "
            "original implementation license and usage note are CC-BY-NC-4.0 "
            "and non-commercial. Redistribution is blocked until that mismatch "
            "is resolved."
        ),
        how_to_use=(
            "from posterllava import PosterLlavaConfig, PosterLlavaPipeline\n"
            'config = PosterLlavaConfig(dataset_name="ad_banner")\n'
            'pipe = PosterLlavaPipeline.from_pretrained("./local-posterllava", config=config)'
        ),
        training_data=(
            "The released checkpoint documentation names Ad Banner, CGL, "
            "PosterLayout, and QB-Poster data. The package does not redistribute "
            "datasets or checkpoint weights."
        ),
        parity_metrics=[
            {
                "dataset": "ci-safe",
                "tokenizer_exact": "prompt/parser/image preprocessing unit tests",
                "deterministic_exact": "full 7B generation parity gated",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
        citation_bibtex=(
            "@article{posterllava2024,\n"
            "  title = {PosterLLaVA: Constructing a Unified Multi-modal Layout Generator with LLM},\n"
            "  year = {2024}\n"
            "}"
        ),
        original_implementation_url="https://github.com/PosterLLaVA/PosterLLaVA",
        model_summary="PosterLLaVA processor and local inference recipe.",
        model_type="Multimodal poster layout generation recipe.",
        base_model="LLaVA-v1.5-style causal language model checkpoint.",
        paper="https://arxiv.org/abs/2406.02884",
        preprocessing="Images are square padded with the CLIP image mean.",
        results_summary="CI-safe parity covers prompt bytes, JSON parsing, and image padding.",
    )


__all__ = ["build_posterllava_model_card"]
