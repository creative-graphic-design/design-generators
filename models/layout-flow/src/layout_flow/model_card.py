"""Model-card generation for converted LayoutFlow checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import ModelCard
from laygen.common.labels import DatasetName
from laygen.common.model_card import build_layout_model_card

from .configuration_layout_flow import normalize_dataset_name


LAYOUTFLOW_BIBTEX: Final[str] = r"""
@inproceedings{guerreiro2024layoutflow,
  title={LayoutFlow: Flow Matching For Layout Generation},
  author={Guerreiro, Julian Jorge Andrade and Inoue, Naoto and Masui, Kento and Otani, Mayu and Nakayama, Hideki},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2024},
  organization={Springer}
}
"""

LAYOUTFLOW_DATASET_IDS: Final[dict[DatasetName, str]] = {
    DatasetName.rico25: "creative-graphic-design/Rico",
    DatasetName.publaynet: "creative-graphic-design/PubLayNet",
}


def layoutflow_model_card(dataset: str) -> ModelCard:
    """Build a model card for a converted LayoutFlow checkpoint.

    Args:
        dataset: LayoutFlow dataset name or alias.

    Returns:
        Validated Hugging Face model card.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Examples:
        >>> card = layoutflow_model_card("publaynet")
        >>> card.data.to_dict()["library_name"]
        'diffusers'
    """
    dataset_name = normalize_dataset_name(dataset)
    dataset_id = _dataset_id(dataset_name)
    dataset_value = str(dataset_name)
    model_id = f"creative-graphic-design/layout-flow-{dataset_value}"
    how_to_use = f"""
from layout_flow import LayoutFlowPipeline

pipe = LayoutFlowPipeline.from_pretrained("{model_id}")
out = pipe(batch_size=1, num_elements=8, seed=0, num_inference_steps=100)
print(out.bbox, out.labels, out.mask, out.id2label)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LayoutFlow {dataset_value}",
        dataset_ids=[dataset_id],
        license="mit",
        library_name="diffusers",
        pipeline_tag="other",
        tags=[
            "layout-generation",
            "layout-flow",
            "flow-matching",
            "diffusers",
            dataset_value,
        ],
        model_details=(
            "Diffusers-format conversion of the LayoutFlow checkpoint for "
            f"`{dataset_value}`. LayoutFlow is a flow-matching layout generator that "
            "integrates a learned vector field over continuous geometry and "
            "analog-bit category labels. Public outputs are normalized center "
            "`xywh` boxes, dataset-local labels, masks, and `id2label` metadata."
        ),
        intended_uses=(
            "Use this checkpoint for research and evaluation of UI/document "
            "layout generation and controllable layout completion workflows."
        ),
        limitations=(
            "This checkpoint generates layout geometry and category labels only; "
            "it does not render images or text. Generation is stochastic unless a "
            "`torch.Generator` or `seed` is supplied. The converted artifact "
            "preserves the original checkpoint behavior and should be evaluated "
            "within the original dataset domain. Do not use it as an image "
            "renderer, OCR system, accessibility checker, safety classifier, or "
            "unreviewed production UI generator."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` using the "
            "splits distributed by the LayoutFlow authors through "
            "`JulianGuerreiro/LayoutFlow`."
        ),
        parity_metrics=[
            {
                "dataset": dataset_value,
                "tokenizer_exact": "n/a",
                "deterministic_exact": "Euler trajectory not measured by parity test",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
        citation_bibtex=LAYOUTFLOW_BIBTEX,
        original_implementation_url="https://github.com/julianguerreiro/LayoutFlow",
    )


def save_layoutflow_model_card(output_dir: str | Path, *, dataset: str) -> Path:
    """Write a LayoutFlow model card as ``README.md``.

    Args:
        output_dir: Directory that receives ``README.md``.
        dataset: LayoutFlow dataset name or alias.

    Returns:
        Path to the written README.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     path = save_layoutflow_model_card(tmp, dataset="publaynet")
        ...     path.name
        'README.md'
    """
    output_path = Path(output_dir) / "README.md"
    output_path.write_text(str(layoutflow_model_card(dataset)), encoding="utf-8")
    return output_path


def _dataset_id(dataset: DatasetName) -> str:
    try:
        return LAYOUTFLOW_DATASET_IDS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported LayoutFlow dataset: {dataset}") from exc
