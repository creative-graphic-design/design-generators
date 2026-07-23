"""Model-card generation for converted LACE checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import ModelCard
from laygen.common.labels import DatasetName
from laygen.common.model_card import ParityMetric, build_layout_model_card

from .configuration_lace import normalize_dataset

_MODEL_IDS: Final[dict[DatasetName, str]] = {
    DatasetName.publaynet: "creative-graphic-design/lace-publaynet",
    DatasetName.rico13: "creative-graphic-design/lace-rico13",
    DatasetName.rico25: "creative-graphic-design/lace-rico25",
}

_DATASET_IDS: Final[dict[DatasetName, str]] = {
    DatasetName.publaynet: "creative-graphic-design/PubLayNet",
    DatasetName.rico13: "creative-graphic-design/Rico",
    DatasetName.rico25: "creative-graphic-design/Rico",
}

_PARITY_METRICS: Final[dict[DatasetName, ParityMetric]] = {
    DatasetName.publaynet: ParityMetric(
        dataset="publaynet",
        tokenizer_exact="n/a",
        deterministic_exact="n/a",
        logits_max_abs=0.0,
        logits_max_rel=0.0,
    ),
    DatasetName.rico25: ParityMetric(
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
    dataset: DatasetName | str,
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
    dataset_key = normalize_dataset(dataset)
    dataset_name = str(dataset_key)
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
        f"`{dataset_name}`. LACE is a continuous diffusion layout generation model "
        "from the paper 'Towards Aligned Layout Generation via Diffusion Model "
        "with Aesthetic Constraints' (https://arxiv.org/abs/2402.04754). "
        "The pipeline generates normalized center `xywh` layout boxes, "
        "category labels, and masks."
    )
    if dataset_key is DatasetName.rico13:
        details += (
            " The public checkpoint archive used for local parity "
            "verification does not include `rico13_best.pt`; Rico13 parity "
            "metrics are therefore not reported here."
        )
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LACE {dataset_name}",
        dataset_ids=[_DATASET_IDS[dataset_key]],
        license="mit",
        library_name="diffusers",
        pipeline_tag="other",
        tags=[
            "layout-generation",
            "lace",
            "diffusers",
            dataset_name,
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
    )


def write_lace_model_card(
    output_dir: str | Path,
    dataset: DatasetName | str,
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
