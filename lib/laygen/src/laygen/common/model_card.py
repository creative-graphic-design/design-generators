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
    model_summary: str,
    developed_by: str,
    model_type: str,
    paper_url: str,
    original_implementation_url: str,
    direct_use: str,
    downstream_use: str,
    out_of_scope_use: str,
    bias_risks_limitations: str,
    recommendations: str,
    how_to_use: str,
    training_data: str,
    training_procedure: str,
    testing_data: str,
    evaluation_factors: str,
    evaluation_metrics: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]],
    technical_specs: str,
    citation_bibtex: str,
    languages: Sequence[str] = ("en",),
) -> ModelCard:
    card_data = ModelCardData(
        model_name=model_name,
        language=list(languages),
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=list(tags),
        datasets=list(dataset_ids),
    )
    card = ModelCard.from_template(
        card_data,
        template_str=_LAYOUT_MODEL_CARD_TEMPLATE,
        model_id=model_id,
        model_name=model_name,
        model_summary=model_summary,
        developed_by=developed_by,
        model_type=model_type,
        license=license,
        paper_url=paper_url,
        original_implementation_url=original_implementation_url,
        direct_use=direct_use,
        downstream_use=downstream_use,
        out_of_scope_use=out_of_scope_use,
        bias_risks_limitations=bias_risks_limitations,
        recommendations=recommendations,
        how_to_use=how_to_use.strip(),
        training_data=training_data,
        training_procedure=training_procedure,
        testing_data=testing_data,
        evaluation_factors=evaluation_factors,
        evaluation_metrics=evaluation_metrics,
        parity_table=_parity_table(parity_metrics),
        technical_specs=technical_specs,
        citation_bibtex=citation_bibtex.strip(),
    )
    card.validate()
    return card


def layoutdm_model_card(
    *,
    dataset: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]] | None = None,
) -> ModelCard:
    dataset_name = "crello" if dataset == "crello-bbox" else dataset
    dataset_id = _layoutdm_dataset_id(dataset_name)
    model_id = f"creative-graphic-design/layoutdm-{dataset_name}"
    model_name = f"LayoutDM {dataset_name}"
    metrics = parity_metrics or [
        ParityMetric(
            dataset=dataset_name,
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
            dataset_name,
        ],
        model_summary=(
            "Diffusers-format conversion of the LayoutDM checkpoint for "
            f"`{dataset_name}`. The pipeline generates normalized center `xywh` "
            "layout boxes, category labels, and masks."
        ),
        developed_by="CyberAgent AI Lab; converted by creative-graphic-design.",
        model_type="Discrete diffusion model for controllable layout generation.",
        paper_url=(
            "https://openaccess.thecvf.com/content/CVPR2023/html/"
            "Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_"
            "Layout_Generation_CVPR_2023_paper.html"
        ),
        original_implementation_url="https://github.com/CyberAgentAILab/layout-dm",
        direct_use=(
            "Generate synthetic document or UI layouts for research, benchmarking, "
            "and qualitative analysis."
        ),
        downstream_use=(
            "Use generated boxes and labels as intermediate layout plans for design "
            "automation, renderer experiments, or controllable generation studies."
        ),
        out_of_scope_use=(
            "Do not use this checkpoint as an image renderer, OCR system, document "
            "understanding model, or as a substitute for human review in production "
            "design workflows."
        ),
        bias_risks_limitations=(
            "The model inherits the category set, preprocessing choices, and visual "
            "layout distribution of the original dataset and release. It may produce "
            "invalid, overlapping, sparse, or dataset-specific layouts."
        ),
        recommendations=(
            "Inspect generated layouts before downstream use and report metrics per "
            "dataset and condition type rather than assuming cross-domain behavior."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The released checkpoint was trained on `{dataset_id}` with the "
            "preprocessing used by the original LayoutDM project."
        ),
        training_procedure=(
            "This repository does not retrain the model. It remaps the released "
            "checkpoint into a Diffusers-style pipeline and preserves the original "
            "scheduler and tokenizer behavior."
        ),
        testing_data=(
            "Vendor parity uses local original starter-kit fixtures generated from "
            "the released preprocessing and checkpoint bundle."
        ),
        evaluation_factors=(
            "Dataset, tokenizer exactness, deterministic sequence exactness, and "
            "denoiser logits numerical agreement."
        ),
        evaluation_metrics=(
            "Exact token match counts and maximum absolute/relative logits error "
            "against the original implementation."
        ),
        parity_metrics=metrics,
        technical_specs=(
            "PyTorch weights are loaded through Diffusers ModelMixin components. "
            "Layouts use a 5-token element order: class, x, y, w, h."
        ),
        citation_bibtex=_LAYOUTDM_BIBTEX,
    )


def layout_corrector_model_card(
    *,
    dataset: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]] | None = None,
) -> ModelCard:
    dataset_name = "crello" if dataset == "crello-bbox" else dataset
    dataset_id = _layout_dataset_id(dataset_name)
    model_id = f"creative-graphic-design/layout-corrector-{dataset_name}"
    model_name = f"Layout-Corrector {dataset_name}"
    metrics = parity_metrics or [
        ParityMetric(
            dataset=dataset_name,
            tokenizer_exact="250/250",
            deterministic_exact="250/250",
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
        model_name=model_name,
        dataset_ids=[dataset_id],
        license="mit",
        library_name="diffusers",
        pipeline_tag="unconditional-layout-generation",
        tags=[
            "layout-generation",
            "layout-corrector",
            "layout-dm",
            "diffusers",
            dataset_name,
        ],
        model_summary=(
            "Diffusers-format composite conversion of the Layout-Corrector "
            f"checkpoint for `{dataset_name}`. The pipeline wraps a converted "
            "LayoutDM denoiser with the released Layout-Corrector confidence model."
        ),
        developed_by=(
            "LY Corporation and Tohoku University; converted by "
            "creative-graphic-design."
        ),
        model_type=(
            "Token-level confidence corrector for discrete diffusion layout generation."
        ),
        paper_url="https://arxiv.org/abs/2409.16689",
        original_implementation_url="https://github.com/line/Layout-Corrector",
        direct_use=(
            "Generate layouts with a LayoutDM backbone and use the corrector to "
            "identify low-confidence reconstructed tokens during sampling."
        ),
        downstream_use=(
            "Use generated layouts as intermediate structure for design research, "
            "layout editing studies, or controlled rendering experiments."
        ),
        out_of_scope_use=(
            "Do not use generated layouts as final production designs without human "
            "review, as an image renderer, or as a document understanding or OCR "
            "system."
        ),
        bias_risks_limitations=(
            "The corrector inherits the dataset taxonomy, preprocessing, and "
            "LayoutDM failure modes. It can still preserve invalid geometry or "
            "dataset-specific layout artifacts."
        ),
        recommendations=(
            "Run vendor parity after conversion, inspect output distributions per "
            "dataset, and keep generated layouts subject to downstream validation."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The released corrector checkpoint was trained on `{dataset_id}`. "
            "For Crello, `cyberagent/crello` is the canonical Hugging Face dataset; "
            "local exact-parity fixtures may still use the original starter-kit "
            "processed split."
        ),
        training_procedure=(
            "This repository does not retrain the corrector. It composes the "
            "released corrector checkpoint with a converted LayoutDM checkpoint and "
            "saves both components through Diffusers-style APIs."
        ),
        testing_data=(
            "Vendor parity uses the local Layout-Corrector starter kit and released "
            "checkpoints for Rico25, PubLayNet, and Crello."
        ),
        evaluation_factors=(
            "Dataset, seed, token exactness for deterministic parity inputs, and "
            "corrector logits numerical agreement."
        ),
        evaluation_metrics=(
            "Exact token match counts and maximum absolute/relative logits error "
            "against the original CategoricalAggregatedTransformer."
        ),
        parity_metrics=metrics,
        technical_specs=(
            "Composite Diffusers pipeline containing a converted LayoutDM pipeline "
            "and a LayoutCorrectorModel. Released corrector weights use shrink-ratio "
            "hidden dimensions inferred from checkpoint tensors."
        ),
        citation_bibtex=_LAYOUT_CORRECTOR_BIBTEX,
    )


def _layoutdm_dataset_id(dataset: str) -> str:
    return _layout_dataset_id(dataset)


def _layout_dataset_id(dataset: str) -> str:
    if dataset == "rico25":
        return "creative-graphic-design/rico25"
    if dataset == "publaynet":
        return "creative-graphic-design/publaynet"
    if dataset == "crello":
        return "cyberagent/crello"
    raise ValueError(f"Unsupported layout dataset: {dataset}")


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
{{ card_data }}
---

# Model Card for {{ model_name }}

{{ model_summary }}

## Model Details

### Model Description

{{ model_summary }}

- **Developed by:** {{ developed_by }}
- **Model type:** {{ model_type }}
- **Language(s):** English metadata and Python APIs; generated outputs are structured layouts, not natural language.
- **License:** {{ license }}
- **Finetuned from model:** Not finetuned from a Hugging Face base model.

### Model Sources

- **Repository:** {{ original_implementation_url }}
- **Paper:** {{ paper_url }}
- **Hub repository:** `{{ model_id }}`

## Uses

### Direct Use

{{ direct_use }}

### Downstream Use

{{ downstream_use }}

### Out-of-Scope Use

{{ out_of_scope_use }}

## Bias, Risks, and Limitations

{{ bias_risks_limitations }}

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

### Testing Data

{{ testing_data }}

### Factors

{{ evaluation_factors }}

### Metrics

{{ evaluation_metrics }}

### Results

{{ parity_table }}

## Technical Specifications

{{ technical_specs }}

## Citation

```bibtex
{{ citation_bibtex }}
```
"""

_LAYOUTDM_BIBTEX = r"""
@inproceedings{inoue2023layoutdm,
  title = {LayoutDM: Discrete Diffusion Model for Controllable Layout Generation},
  author = {Inoue, Naoto and Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {CVPR},
  year = {2023}
}
"""

_LAYOUT_CORRECTOR_BIBTEX = r"""
@article{iwai2024layoutcorrector,
  title = {Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model},
  author = {Iwai, Shoma and Osanai, Atsuki and Kitada, Shunsuke and Omachi, Shinichiro},
  journal = {arXiv preprint arXiv:2409.16689},
  year = {2024}
}
"""
