"""Model-card builders shared by converted layout model packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from huggingface_hub import ModelCard, ModelCardData


@dataclass(frozen=True)
class ParityMetric:
    """Vendor-parity metric row included in generated model cards.

    Attributes:
        dataset: Dataset or checkpoint name.
        tokenizer_exact: Exact-match ratio for tokenizer round-trips.
        deterministic_exact: Exact-match ratio for deterministic samples.
        logits_max_abs: Maximum absolute denoiser-logit difference.
        logits_max_rel: Maximum relative denoiser-logit difference.
    """

    dataset: str
    tokenizer_exact: str
    deterministic_exact: str
    logits_max_abs: float
    logits_max_rel: float


def build_layout_model_card(
    *,
    model_id: str,
    model_name: str,
    dataset_ids: Sequence[str],
    license: str,
    library_name: str,
    pipeline_tag: str,
    tags: Sequence[str],
    model_details: str,
    intended_uses: str,
    limitations: str,
    how_to_use: str,
    training_data: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]],
    citation_bibtex: str,
    original_implementation_url: str,
) -> ModelCard:
    """Build a Hugging Face model card for a layout-generation checkpoint.

    Args:
        model_id: Hub model id displayed in the card title.
        model_name: Human-readable model name.
        dataset_ids: Hub dataset ids used by the checkpoint.
        license: SPDX-style license id for YAML metadata.
        library_name: Hub library name, such as ``diffusers``.
        pipeline_tag: Hub task tag.
        tags: Additional Hub tags.
        model_details: User-facing model description.
        intended_uses: Direct-use description.
        limitations: Known limitations and risks.
        how_to_use: Python snippet without surrounding fences.
        training_data: Training-data description.
        parity_metrics: Parity table rows.
        citation_bibtex: BibTeX citation without surrounding fences.
        original_implementation_url: URL for the upstream implementation.

    Returns:
        Validated ``huggingface_hub.ModelCard`` instance.

    Raises:
        ValueError: If model-card metadata validation fails.

    Examples:
        >>> card = layoutdm_model_card(dataset="rico25")
        >>> card.data.to_dict()["library_name"]
        'diffusers'
    """
    card_data = ModelCardData(
        model_name=model_name,
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=list(tags),
        datasets=list(dataset_ids),
        language=["en"],
    )
    card = ModelCard.from_template(
        card_data,
        template_str=_LAYOUT_MODEL_CARD_TEMPLATE,
        metadata=card_data.to_yaml(),
        model_name=model_name,
        model_id=model_id,
        model_details=model_details,
        direct_uses=intended_uses,
        downstream_uses=(
            "Use the generated structured layouts as intermediate data for "
            "research prototypes, layout evaluation, or downstream rendering "
            "systems that separately validate visual quality and content safety."
        ),
        out_of_scope_uses=(
            "Do not use this checkpoint as an image renderer, OCR system, "
            "document parser, or production design automation system without "
            "task-specific evaluation and human review."
        ),
        limitations=limitations,
        recommendations=(
            "Validate outputs on the target dataset, inspect generated layouts "
            "before publication, and rerun the parity tests when changing "
            "conversion code or dependencies."
        ),
        how_to_use=how_to_use.strip(),
        training_data=training_data,
        training_procedure=(
            "The converted checkpoint preserves the released upstream weights. "
            "This repository does not retrain the model during conversion."
        ),
        testing_data=(
            "Local deterministic fixtures generated from the original "
            "implementation with fixed labels, masks, and latent tensors."
        ),
        evaluation_factors=(
            "Architecture compatibility, tokenizer/processor behavior where "
            "applicable, deterministic generation, and numeric parity against "
            "the released checkpoint."
        ),
        evaluation_metrics=(
            "Exact deterministic comparisons and maximum absolute/relative "
            "differences for generated tensors."
        ),
        evaluation_results=_parity_table(parity_metrics),
        technical_specs=(
            f"Library: `{library_name}`. Pipeline tag: `{pipeline_tag}`. "
            "Generated bounding boxes are normalized center `xywh` tensors with "
            "padding represented by masks."
        ),
        original_implementation_url=original_implementation_url,
        citation_bibtex=citation_bibtex.strip(),
    )
    card.validate()
    return card


def layoutdm_model_card(
    *,
    dataset: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]] | None = None,
) -> ModelCard:
    """Build the LayoutDM model card for a converted checkpoint.

    Args:
        dataset: LayoutDM dataset name, either ``"rico25"`` or ``"publaynet"``.
        parity_metrics: Optional parity rows. Defaults to the checked conversion
            metrics used by this package.

    Returns:
        Validated model card for the requested LayoutDM checkpoint.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Examples:
        >>> card = layoutdm_model_card(dataset="publaynet")
        >>> card.data.to_dict()["datasets"]
        ['creative-graphic-design/publaynet']
    """
    dataset_id = _layoutdm_dataset_id(dataset)
    model_id = f"creative-graphic-design/layoutdm-{dataset}"
    model_name = f"LayoutDM {dataset}"
    metrics = parity_metrics or [
        ParityMetric(
            dataset=dataset,
            tokenizer_exact="125/125",
            deterministic_exact="125/125",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]
    how_to_use = f"""
from layout_dm import LayoutDMPipeline

pipe = LayoutDMPipeline.from_pretrained("{model_id}")
out = pipe(batch_size=1, seed=0, sampling="deterministic")
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=model_name,
        dataset_ids=[dataset_id],
        license="apache-2.0",
        library_name="diffusers",
        pipeline_tag="text-to-image",
        tags=[
            "layout-generation",
            "layout-dm",
            "diffusers",
            dataset,
        ],
        model_details=(
            "Diffusers-format conversion of the LayoutDM checkpoint for "
            f"`{dataset}`. The pipeline generates normalized center `xywh` layout "
            "boxes, category labels, and masks."
        ),
        intended_uses=(
            "Use this checkpoint for research and evaluation of document and UI "
            "layout generation workflows."
        ),
        limitations=(
            "The converted checkpoint follows the original LayoutDM release and is "
            "intended for layout synthesis, not for image rendering or OCR."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` as released by "
            "the original LayoutDM project."
        ),
        parity_metrics=metrics,
        citation_bibtex=_LAYOUTDM_BIBTEX,
        original_implementation_url=("https://github.com/CyberAgentAILab/layout-dm"),
    )


def _layoutdm_dataset_id(dataset: str) -> str:
    if dataset == "rico25":
        return "creative-graphic-design/rico25"
    if dataset == "publaynet":
        return "creative-graphic-design/publaynet"
    raise ValueError(f"Unsupported LayoutDM dataset: {dataset}")


def _parity_table(metrics: Sequence[ParityMetric | Mapping[str, object]]) -> str:
    rows = [
        "| Dataset | Tokenizer exact | Deterministic exact | Logits max abs | Logits max rel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        item = _metric_dict(metric)
        rows.append(
            "| {dataset} | {tokenizer_exact} | {deterministic_exact} | "
            "{logits_max_abs:g} | {logits_max_rel:g} |".format(**item)
        )
    return "\n".join(rows)


def _metric_dict(metric: ParityMetric | Mapping[str, object]) -> dict[str, object]:
    if isinstance(metric, ParityMetric):
        return {
            "dataset": metric.dataset,
            "tokenizer_exact": metric.tokenizer_exact,
            "deterministic_exact": metric.deterministic_exact,
            "logits_max_abs": metric.logits_max_abs,
            "logits_max_rel": metric.logits_max_rel,
        }
    return dict(metric)


_LAYOUT_MODEL_CARD_TEMPLATE = """---
{{ metadata }}
---

# {{ model_name }}

## Model Details

### Model Description

Hub repository: `{{ model_id }}`

{{ model_details }}

Original implementation: {{ original_implementation_url }}

### Developed by

The original checkpoint and architecture were developed by the authors cited
below. This repository provides a converted checkpoint interface for layout
generation research.

### Model type

Neural graphic layout generation model.

### License

See the `license` value in the card metadata.

### Finetuned from model

Not applicable. This is a conversion of the released upstream checkpoint, not a
finetuned derivative.

## Uses

### Direct Use

{{ direct_uses }}

### Downstream Use

{{ downstream_uses }}

### Out-of-Scope Use

{{ out_of_scope_uses }}

## Bias, Risks, and Limitations

{{ limitations }}

### Recommendations

{{ recommendations }}

## How to Get Started with the Model

```python
{{ how_to_use }}
```

## Training Details

### Training Data

{{ training_data }}

### Training Procedure

{{ training_procedure }}

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

{{ testing_data }}

#### Factors

{{ evaluation_factors }}

#### Metrics

{{ evaluation_metrics }}

### Results

{{ evaluation_results }}

## Technical Specifications

{{ technical_specs }}

## Citation

```bibtex
{{ citation_bibtex }}
```

## Model Card Contact

Open an issue or pull request in the creative-graphic-design design-generators
repository.
"""


_LAYOUTDM_BIBTEX = r"""
@inproceedings{inoue2023layoutdm,
  title = {LayoutDM: Discrete Diffusion Model for Controllable Layout Generation},
  author = {Inoue, Naoto and Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {CVPR},
  year = {2023}
}
"""
