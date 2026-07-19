"""Hub model-card helper for converted DS-GAN checkpoints."""

from __future__ import annotations

from pathlib import Path

from laygen.common.model_card import build_layout_model_card


def dsgan_model_card() -> object:
    """Build the DS-GAN PKU PosterLayout Hub model card."""
    return build_layout_model_card(
        model_id="creative-graphic-design/ds-gan-pku-posterlayout",
        model_name="DS-GAN PosterLayout",
        dataset_ids=["creative-graphic-design/PKU-PosterLayout"],
        license="other",
        library_name="transformers",
        pipeline_tag="image-to-image",
        tags=["ds-gan", "posterlayout", "layout-generation", "poster-generation"],
        model_details=(
            "DS-GAN predicts poster element boxes and semantic labels from an RGB "
            "poster/background image plus saliency. The converted package returns "
            "normalized center xywh boxes, zero-based labels, and mask-based padding."
        ),
        intended_uses=(
            "Research use for content-aware poster layout generation and checkpoint "
            "parity studies against the original CVPR 2023 implementation."
        ),
        limitations=(
            "The upstream repository does not include a redistribution license. "
            "Converted weights should not be published until license permission is "
            "resolved."
        ),
        how_to_use=(
            "from ds_gan import DSGANPipeline\n\n"
            "pipe = DSGANPipeline.from_pretrained(\n"
            '    "creative-graphic-design/ds-gan-pku-posterlayout"\n'
            ")\n"
            "out = pipe(images=image, saliency=saliency, seed=0)"
        ),
        training_data=(
            "PKU PosterLayout via creative-graphic-design/PKU-PosterLayout; "
            "annotations contain pixel ltrb boxes and an INVALID class that is "
            "excluded from public semantic labels."
        ),
        parity_metrics=[
            {
                "dataset": "PKU PosterLayout",
                "tokenizer_exact": "not applicable",
                "deterministic_exact": "pending local vendor assets",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
        citation_bibtex=(
            "@inproceedings{Hsu-2023-posterlayout,\n"
            "  title={PosterLayout: A New Benchmark and Approach for "
            "Content-Aware Visual-Textual Presentation Layout},\n"
            "  author={HsiaoYuan Hsu and Xiangteng He and Yuxin Peng and "
            "Hao Kong and Qing Zhang},\n"
            "  booktitle={CVPR},\n"
            "  year={2023}\n"
            "}"
        ),
        original_implementation_url=(
            "https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023"
        ),
    )


def write_dsgan_model_card(output_dir: str | Path) -> Path:
    """Write ``README.md`` for a converted DS-GAN checkpoint directory."""
    path = Path(output_dir) / "README.md"
    path.write_text(str(dsgan_model_card()), encoding="utf-8")
    return path
