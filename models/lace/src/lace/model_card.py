"""Model-card generation for converted LACE checkpoints."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import ModelCard
from laygen.common.model_card import ParityMetric, build_layout_model_card

from .configuration_lace import normalize_dataset

_MODEL_IDS = {
    "publaynet": "creative-graphic-design/lace-publaynet",
    "rico13": "creative-graphic-design/lace-rico13",
    "rico25": "creative-graphic-design/lace-rico25",
}

_DATASET_IDS = {
    "publaynet": "creative-graphic-design/publaynet",
    "rico13": "creative-graphic-design/rico13",
    "rico25": "creative-graphic-design/rico25",
}

_PARITY_METRICS = {
    "publaynet": ParityMetric(
        dataset="publaynet",
        tokenizer_exact="n/a",
        deterministic_exact="n/a",
        logits_max_abs=0.0,
        logits_max_rel=0.0,
    ),
    "rico25": ParityMetric(
        dataset="rico25",
        tokenizer_exact="n/a",
        deterministic_exact="n/a",
        logits_max_abs=0.0,
        logits_max_rel=0.0,
    ),
}

_LACE_BIBTEX = r"""
@inproceedings{
    chen2024towards,
    title={Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints},
    author={Jian Chen and Ruiyi Zhang and Yufan Zhou and Changyou Chen},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=kJ0qp9Xdsh}
}
"""


def lace_model_card(
    dataset: str,
    *,
    parity_metrics: list[ParityMetric] | None = None,
) -> ModelCard:
    """Create the Hugging Face model card for a LACE checkpoint.

    Args:
        dataset: LACE dataset name or alias.
        parity_metrics: Optional parity metrics to embed in the evaluation section.

    Returns:
        Rendered Hugging Face Hub model card.

    Raises:
        ValueError: If the dataset is unsupported.
    """
    dataset_key = str(normalize_dataset(dataset))
    model_id = _MODEL_IDS[dataset_key]
    metrics = parity_metrics
    if metrics is None:
        metrics = (
            [_PARITY_METRICS[dataset_key]] if dataset_key in _PARITY_METRICS else []
        )
    how_to_use = f"""
from lace import LacePipeline

pipe = LacePipeline.from_pretrained("{model_id}")
out = pipe(batch_size=1, seed=0, num_inference_steps=100)
print(out.bbox, out.labels, out.mask)
"""
    details = (
        "Diffusers-format conversion of the LACE checkpoint for "
        f"`{dataset_key}`. LACE is a continuous diffusion layout generation model "
        "from the paper 'Towards Aligned Layout Generation via Diffusion Model "
        "with Aesthetic Constraints' (https://arxiv.org/abs/2402.04754). "
        "The pipeline generates normalized center `xywh` layout boxes, "
        "category labels, and masks."
    )
    if dataset_key == "rico13":
        details += (
            " The public vendor checkpoint archive used for local parity "
            "verification does not include `rico13_best.pt`; Rico13 parity "
            "metrics are therefore not reported here."
        )
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LACE {dataset_key}",
        dataset_ids=[_DATASET_IDS[dataset_key]],
        license="mit",
        library_name="diffusers",
        pipeline_tag="text-to-image",
        tags=[
            "layout-generation",
            "lace",
            "diffusers",
            dataset_key,
        ],
        model_details=details,
        intended_uses=(
            "Use this checkpoint for research and evaluation of document and UI "
            "layout generation workflows."
        ),
        limitations=(
            "The converted checkpoint follows the original LACE release and is "
            "intended for layout synthesis, not for image rendering or OCR. "
            "Generated layouts may require downstream filtering for task-specific "
            "aesthetic or overlap constraints."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{_DATASET_IDS[dataset_key]}` "
            "as released by the original LACE project."
        ),
        parity_metrics=metrics,
        citation_bibtex=_LACE_BIBTEX,
        original_implementation_url="https://github.com/puar-playground/LACE",
        developers="Jian Chen, Ruiyi Zhang, Yufan Zhou, and Changyou Chen; converted by creative-graphic-design contributors.",
        model_type="Continuous diffusion model for aesthetic-constrained layout generation.",
        paper_url="https://openreview.net/forum?id=kJ0qp9Xdsh",
        direct_use=(
            "Generate normalized layout boxes, labels, and masks from the converted "
            "Diffusers pipeline without fine-tuning."
        ),
        downstream_use=(
            "Use generated layouts as candidates for document or UI rendering, "
            "layout editing, data augmentation, or evaluation workflows."
        ),
        out_of_scope_use=(
            "Do not use the model as an image renderer, OCR system, safety-critical "
            "document generation system, or a replacement for human review of "
            "layout quality and content suitability."
        ),
        bias_recommendations=(
            "Inspect outputs for dataset bias, element overlap, category balance, "
            "and downstream rendering constraints before release or deployment."
        ),
        preprocessing=(
            "The original LACE workflow represents layouts as max-25 continuous "
            "sequences with one-hot category channels and normalized bounding boxes."
        ),
        training_regime="Original training regime from the upstream LACE release.",
        testing_data=(
            f"Vendor parity tests use `{_DATASET_IDS[dataset_key]}` checkpoint fixtures "
            "from the original LACE release when the public checkpoint is available."
        ),
        testing_factors=(
            "Parity is checked per available dataset and covers checkpoint conversion "
            "smoke tests plus denoiser logits for PubLayNet and Rico25."
        ),
        testing_metrics=(
            "Conversion smoke output shapes and bbox range, plus denoiser logits "
            "maximum absolute and relative differences."
        ),
        model_specs=(
            "Converted LACE transformer denoiser, processor, and DDIM-style scheduler "
            "wrapped in a Diffusers-style pipeline."
        ),
        software="Python, PyTorch, Diffusers, and laygen.common.",
    )


def write_lace_model_card(
    output_dir: str | Path,
    dataset: str,
    *,
    parity_metrics: list[ParityMetric] | None = None,
) -> Path:
    """Write the LACE model card into a model directory.

    Args:
        output_dir: Directory where ``README.md`` is written.
        dataset: LACE dataset name or alias.
        parity_metrics: Optional parity metrics to embed in the evaluation section.

    Returns:
        Path to the written ``README.md``.

    Raises:
        ValueError: If the dataset is unsupported.
    """
    path = Path(output_dir) / "README.md"
    path.write_text(
        str(lace_model_card(dataset, parity_metrics=parity_metrics)),
        encoding="utf-8",
    )
    return path
