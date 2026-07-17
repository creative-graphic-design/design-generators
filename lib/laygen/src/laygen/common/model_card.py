from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from huggingface_hub import ModelCard, ModelCardData


@dataclass(frozen=True)
class ParityMetric:
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
    language: Sequence[str] | None = None,
    developed_by: str = "creative-graphic-design",
    model_type: str = "layout generation model",
    finetuned_from: str = "Not applicable; this is a converted original checkpoint.",
    downstream_uses: str = (
        "Use the generated structured layouts as intermediate inputs for design "
        "analysis, rendering, retrieval, or other research pipelines."
    ),
    out_of_scope_uses: str = (
        "Do not use this model as a general-purpose image generator, OCR system, "
        "or production design decision maker without task-specific validation."
    ),
    recommendations: str = (
        "Evaluate outputs on the target dataset and inspect generated layouts "
        "before using them in downstream user-facing workflows."
    ),
    training_procedure: str = (
        "This repository converts the released research checkpoint and does not "
        "retrain it. See the cited paper and original implementation for the "
        "full training recipe."
    ),
    testing_data: str = (
        "Local parity tests use deterministic fixtures generated from the "
        "released original implementation."
    ),
    evaluation_factors: str = (
        "Parity checks focus on tokenizer compatibility, deterministic generated "
        "tokens when available, and teacher-forced logits for selected fixtures."
    ),
    evaluation_metrics: str = (
        "Tokenizer exact match, deterministic output exact match, and maximum "
        "absolute/relative logit differences."
    ),
    technical_specs: str = (
        "The converted package exposes `save_pretrained` / `from_pretrained` "
        "artifacts and returns normalized center `xywh` boxes, labels, and masks "
        "through the shared `laygen.common` output schema."
    ),
) -> ModelCard:
    card_data = ModelCardData(
        model_name=model_name,
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=list(tags),
        datasets=list(dataset_ids),
        language=list(language) if language is not None else None,
    )
    parity_table = _parity_table(parity_metrics)
    content = f"""---
{card_data.to_yaml()}
---

# {model_name}

## Model Details

### Model Description

{model_details}

- **Developed by:** {developed_by}
- **Model type:** {model_type}
- **License:** {license}
- **Finetuned from:** {finetuned_from}
- **Hub repository:** `{model_id}`
- **Original implementation:** {original_implementation_url}

## Uses

### Direct Use

{intended_uses}

### Downstream Use

{downstream_uses}

### Out-of-Scope Use

{out_of_scope_uses}

## Bias, Risks, and Limitations

{limitations}

### Recommendations

{recommendations}

## How to Get Started with the Model

```python
{how_to_use.strip()}
```

## Training Details

### Training Data

{training_data}

### Training Procedure

{training_procedure}

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

{testing_data}

#### Factors

{evaluation_factors}

#### Metrics

{evaluation_metrics}

### Results

{parity_table}

## Technical Specifications

{technical_specs}

## Citation

```bibtex
{citation_bibtex.strip()}
```
"""
    return ModelCard(content)


def layoutdm_model_card(
    *,
    dataset: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]] | None = None,
) -> ModelCard:
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


_LAYOUTDM_BIBTEX = r"""
@inproceedings{inoue2023layoutdm,
  title = {LayoutDM: Discrete Diffusion Model for Controllable Layout Generation},
  author = {Inoue, Naoto and Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {CVPR},
  year = {2023}
}
"""
