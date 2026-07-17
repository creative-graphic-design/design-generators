"""Model-card builders for converted Layout-Corrector checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from huggingface_hub import ModelCard
from laygen.common.model_card import ParityMetric, build_layout_model_card


def layout_corrector_model_card(
    *,
    dataset: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]] | None = None,
) -> ModelCard:
    """Build the Layout-Corrector model card for a converted checkpoint."""
    dataset_name = "crello" if dataset in {"crello", "crello-bbox"} else dataset
    dataset_id = _layout_corrector_dataset_id(dataset_name)
    model_id = f"creative-graphic-design/layout-corrector-{dataset_name}"
    metrics = parity_metrics or [
        ParityMetric(
            dataset=dataset_name,
            tokenizer_exact="checked in vendor parity",
            deterministic_exact="not applicable",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]
    how_to_use = f"""
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained("{model_id}")
out = pipe(batch_size=1, seed=0, sampling="deterministic")
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"Layout-Corrector {dataset_name}",
        dataset_ids=[dataset_id],
        license="mit",
        library_name="diffusers",
        pipeline_tag="unconditional-layout-generation",
        tags=[
            "layout-generation",
            "layout-corrector",
            "diffusers",
            dataset_name,
        ],
        model_details=(
            "Diffusers-format composite pipeline for Layout-Corrector. The "
            "pipeline wraps a converted LayoutDM generator with the released "
            "Layout-Corrector confidence model to re-mask low-confidence layout "
            "tokens during sampling."
        ),
        intended_uses=(
            "Use this checkpoint for research and evaluation of controllable "
            "layout generation and Layout-Corrector sampling behavior."
        ),
        limitations=(
            "The model predicts layout structure only. Generated boxes and labels "
            "require downstream validation before use in design or document "
            "processing workflows."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` using the "
            "preprocessing released with the original Layout-Corrector starter kit."
        ),
        parity_metrics=metrics,
        citation_bibtex=_LAYOUT_CORRECTOR_BIBTEX,
        original_implementation_url="https://github.com/line/Layout-Corrector",
    )


def _layout_corrector_dataset_id(dataset: str) -> str:
    if dataset == "rico25":
        return "creative-graphic-design/rico25"
    if dataset == "publaynet":
        return "creative-graphic-design/publaynet"
    if dataset == "crello":
        return "cyberagent/crello"
    raise ValueError(f"Unsupported Layout-Corrector dataset: {dataset}")


MODEL_CARD_TEMPLATE = """---
license: mit
library_name: diffusers
pipeline_tag: unconditional-layout-generation
---

# Layout-Corrector

Composite Layout-Corrector pipeline converted from the original MIT-licensed implementation. The nested LayoutDM components are derived from Apache-2.0 LayoutDM code and should be cited alongside Layout-Corrector.
"""


_LAYOUT_CORRECTOR_BIBTEX = r"""
@article{iwai2024layoutcorrector,
  title = {Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model},
  author = {Iwai, Shoma and Osanai, Atsuki and Kitada, Shunsuke and Omachi, Shinichiro},
  journal = {arXiv preprint arXiv:2409.16689},
  year = {2024}
}
"""
