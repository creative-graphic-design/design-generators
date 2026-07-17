"""Model card builders for LayoutGAN++ checkpoint packages."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import ModelCard
from laygen.common.model_card import build_layout_model_card

from .datasets import DatasetName, normalize_dataset_name


_DATASET_IDS: Final[dict[DatasetName, str]] = {
    DatasetName.rico: "creative-graphic-design/rico",
    DatasetName.publaynet: "creative-graphic-design/publaynet",
    DatasetName.magazine: "creative-graphic-design/magazine",
}

_CHECKPOINT_IDS: Final[dict[DatasetName, str]] = {
    dataset: f"creative-graphic-design/layoutganpp-{dataset}"
    for dataset in _DATASET_IDS
}

_EXAMPLE_LABELS: Final[dict[DatasetName, str]] = {
    DatasetName.rico: '[["Toolbar", "Image"]]',
    DatasetName.publaynet: '[["text", "figure"]]',
    DatasetName.magazine: '[["text", "image"]]',
}

_PARITY_METRICS: Final[dict[DatasetName, dict[str, str]]] = {
    DatasetName.rico: {"shape": "(3, 9, 4)", "smoke_shape": "(1, 2, 4)"},
    DatasetName.publaynet: {"shape": "(3, 9, 4)", "smoke_shape": "(1, 2, 4)"},
    DatasetName.magazine: {"shape": "(3, 33, 4)", "smoke_shape": "(1, 2, 4)"},
}

_BIBTEX: Final[str] = r"""
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {ACM International Conference on Multimedia},
    series = {MM '21},
    year = {2021},
    pages = {88--96},
    doi = {10.1145/3474085.3475497}
}
"""


def layoutganpp_model_card(dataset: DatasetName | str) -> ModelCard:
    """Build a Hugging Face model card for a LayoutGAN++ dataset.

    Args:
        dataset: Dataset key or alias for the converted checkpoint.

    Returns:
        A populated `ModelCard` for the selected checkpoint.

    Raises:
        ValueError: If `dataset` is not a supported LayoutGAN++ dataset.

    Examples:
        >>> card = layoutganpp_model_card("rico")
        >>> "layoutganpp-rico" in str(card)
        True
    """
    dataset_name = normalize_dataset_name(dataset)
    dataset_key = str(dataset_name)
    model_id = _CHECKPOINT_IDS[dataset_name]
    metrics = _PARITY_METRICS[dataset_name]
    how_to_use = f"""
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained("{model_id}")
out = pipe(labels={_EXAMPLE_LABELS[dataset_name]}, seed=0)
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LayoutGAN++ {dataset_key}",
        dataset_ids=[_DATASET_IDS[dataset_name]],
        license="agpl-3.0",
        library_name="transformers",
        pipeline_tag="text-to-image",
        tags=[
            "layout-generation",
            "layoutganpp",
            "layoutgan++",
            "transformers",
            dataset_key,
        ],
        model_details=(
            "Transformers-style conversion of the LayoutGAN++ generator from "
            "`Constrained Graphic Layout Generation via Latent Optimization` "
            f"for `{dataset_key}`. The model generates normalized center `xywh` "
            "layout boxes from category-label conditions. Vendor parity compares "
            f"bbox tensors with shape {metrics['shape']} and observed max_abs=0 "
            "and max_rel=0 against local const-layout fixtures."
        ),
        intended_uses=(
            "Use this checkpoint for research on constrained graphic layout "
            "generation and for regression tests that need deterministic "
            "LayoutGAN++ bbox generation from fixed labels and latents."
        ),
        limitations=(
            "This package ports the released generator only. It does not include "
            "LayoutGAN++ latent optimization loops, training code, rendered image "
            "generation, or dataset preprocessing pipelines."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{_DATASET_IDS[dataset_name]}` "
            "following the upstream LayoutGAN++ release."
        ),
        parity_metrics=[
            {
                "dataset": dataset_key,
                "tokenizer_exact": "n/a",
                "deterministic_exact": "bbox exact",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
        citation_bibtex=_BIBTEX,
        original_implementation_url="https://github.com/ktrk115/const_layout",
    )


def write_layoutganpp_model_card(output_dir: Path, dataset: DatasetName | str) -> Path:
    """Write a LayoutGAN++ model card to an output directory.

    Args:
        output_dir: Directory that will receive `README.md`.
        dataset: Dataset key or alias for the converted checkpoint.

    Returns:
        Path to the written `README.md` file.

    Raises:
        ValueError: If `dataset` is not a supported LayoutGAN++ dataset.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     path = write_layoutganpp_model_card(Path(tmp), "rico")
        ...     path.name
        'README.md'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(str(layoutganpp_model_card(dataset)), encoding="utf-8")
    return readme_path
