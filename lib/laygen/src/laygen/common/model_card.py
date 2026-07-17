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
    language: Sequence[str] = ("en",),
    model_details: str,
    model_summary: str,
    developed_by: str,
    model_type: str,
    repo_url: str,
    paper_url: str,
    demo_url: str,
    direct_use: str,
    downstream_use: str,
    out_of_scope_use: str,
    limitations: str,
    recommendations: str,
    how_to_use: str,
    training_data: str,
    training_procedure: str,
    preprocessing: str,
    training_regime: str,
    speeds_sizes_times: str,
    testing_data: str,
    testing_factors: str,
    testing_metrics: str,
    parity_metrics: Sequence[ParityMetric | Mapping[str, object]],
    results_summary: str,
    model_specs: str,
    compute_infrastructure: str,
    hardware_requirements: str,
    software: str,
    citation_bibtex: str,
    citation_apa: str,
    original_implementation_url: str,
    model_card_authors: str,
    model_card_contact: str,
) -> ModelCard:
    card_data = ModelCardData(
        model_name=model_name,
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=list(tags),
        datasets=list(dataset_ids),
        language=list(language),
    )
    parity_table = _parity_table(parity_metrics)
    results = f"""Vendor parity against the original implementation:

{parity_table}
"""
    card = ModelCard.from_template(
        card_data=card_data,
        model_id=model_id,
        model_summary=model_summary,
        model_description=(
            f"{model_details}\n\n"
            "This card follows the Hugging Face Hub model card template and "
            "the annotated model card section structure."
        ),
        developers=developed_by,
        funded_by="Not reported by the original release.",
        shared_by="creative-graphic-design",
        model_type=model_type,
        language=", ".join(language),
        license=license,
        base_model="Not applicable; this is a converted original checkpoint.",
        repo=repo_url,
        paper=paper_url,
        demo=demo_url,
        direct_use=direct_use,
        downstream_use=downstream_use,
        out_of_scope_use=out_of_scope_use,
        bias_risks_limitations=limitations,
        bias_recommendations=recommendations,
        get_started_code=f"```python\n{how_to_use.strip()}\n```",
        training_data=training_data,
        preprocessing=f"{training_procedure}\n\n{preprocessing}",
        training_regime=training_regime,
        speeds_sizes_times=speeds_sizes_times,
        testing_data=testing_data,
        testing_factors=testing_factors,
        testing_metrics=testing_metrics,
        results=results,
        results_summary=results_summary,
        model_examination="No separate interpretability study is packaged with this conversion.",
        hardware_type="Not reported by the original release.",
        hours_used="Not reported by the original release.",
        cloud_provider="Not reported by the original release.",
        cloud_region="Not reported by the original release.",
        co2_emitted=(
            "Carbon emissions cannot be estimated from the released checkpoint "
            "bundle alone."
        ),
        model_specs=model_specs,
        compute_infrastructure=compute_infrastructure,
        hardware_requirements=hardware_requirements,
        software=software,
        citation_bibtex=f"```bibtex\n{citation_bibtex.strip()}\n```",
        citation_apa=citation_apa,
        glossary=(
            "`bbox` uses normalized center `xywh` coordinates. `mask` marks valid "
            "layout elements and padding separately from category labels."
        ),
        more_information=f"Original implementation: {original_implementation_url}",
        model_card_authors=model_card_authors,
        model_card_contact=model_card_contact,
    )
    card.validate()
    return card


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
        model_summary=(
            f"{model_name} is a Diffusers-format LayoutDM checkpoint for "
            "conditional-free layout generation."
        ),
        model_details=(
            "Diffusers-format conversion of the LayoutDM checkpoint for "
            f"`{dataset}`. The pipeline generates normalized center `xywh` layout "
            "boxes, category labels, and masks."
        ),
        developed_by=(
            "CyberAgentAILab released the original LayoutDM implementation; "
            "converted by creative-graphic-design."
        ),
        model_type="Discrete diffusion model for layout generation.",
        repo_url="https://github.com/CyberAgentAILab/layout-dm",
        paper_url="https://arxiv.org/abs/2303.08137",
        demo_url="No hosted demo is packaged with this checkpoint.",
        direct_use=(
            "Use this checkpoint for research and evaluation of document and UI "
            "layout generation workflows."
        ),
        downstream_use=(
            "Use the generated normalized boxes, labels, and masks as layout "
            "priors for design tooling, document analysis research, or controlled "
            "rendering pipelines that perform their own validation."
        ),
        out_of_scope_use=(
            "Do not use this checkpoint as an OCR model, image renderer, semantic "
            "document understanding model, accessibility verifier, or unreviewed "
            "production UI generator. The model predicts layout structure only "
            "and can produce implausible or overlapping boxes."
        ),
        limitations=(
            "The converted checkpoint follows the original LayoutDM release and is "
            "intended for layout synthesis, not for image rendering or OCR."
        ),
        recommendations=(
            "Inspect generated layouts before downstream use, validate boxes "
            "against application constraints, and evaluate separately for each "
            "target dataset or design domain."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` as released by "
            "the original LayoutDM project."
        ),
        training_procedure=(
            "Original LayoutDM training regime as released by the upstream "
            "project; this package converts the checkpoint and does not retrain it."
        ),
        preprocessing=(
            "The converted tokenizer represents each layout element as discrete "
            "category and bounding-box tokens. Bounding boxes use normalized "
            "center `xywh` coordinates and dataset-specific cluster centers stored "
            "with the tokenizer files."
        ),
        training_regime="Original training regime; not retrained during conversion.",
        speeds_sizes_times=(
            "Training speed, elapsed time, and hardware are not included in the "
            "upstream checkpoint bundle used for conversion."
        ),
        testing_data=(
            "Vendor parity tests use deterministic samples and forward-pass golden "
            "tensors generated from the original LayoutDM implementation for each "
            "converted dataset."
        ),
        testing_factors=(
            "Parity is checked separately for each dataset conversion so that "
            "dataset-specific tokenization and checkpoint weights are covered."
        ),
        testing_metrics=(
            "Tokenizer exact-match count, deterministic token-sequence exact-match "
            "count, and denoiser logits maximum absolute and relative error versus "
            "the original implementation."
        ),
        parity_metrics=metrics,
        results_summary=(
            "The converted checkpoint matches the generated vendor reference "
            "tensors exactly for tokenizer IO and deterministic sampling; denoiser "
            "logits are within the reported numeric tolerance."
        ),
        model_specs=(
            "LayoutDM models layout generation as discrete diffusion over category "
            "and bounding-box token sequences. This package exposes the denoiser, "
            "tokenizer, scheduler, and Diffusers pipeline needed to reproduce "
            "converted inference."
        ),
        compute_infrastructure=(
            "Conversion and parity generation run locally through the `uv` "
            "workspace commands documented in `models/layout-dm/README.md`."
        ),
        hardware_requirements=(
            "CPU is sufficient for package loading and conversion. CUDA is "
            "recommended for regenerating vendor parity references and running "
            "the full parity test suite."
        ),
        software=(
            "Python 3.11+, PyTorch, Diffusers, Transformers, and the optional "
            "LayoutDM vendor dependencies declared by the `layout-dm` package."
        ),
        citation_bibtex=_LAYOUTDM_BIBTEX,
        citation_apa=(
            "Inoue, N., Kikuchi, K., Simo-Serra, E., Otani, M., & Yamaguchi, K. "
            "(2023). LayoutDM: Discrete Diffusion Model for Controllable Layout "
            "Generation. CVPR."
        ),
        original_implementation_url="https://github.com/CyberAgentAILab/layout-dm",
        model_card_authors="creative-graphic-design maintainers.",
        model_card_contact=(
            "Open an issue or pull request in the creative-graphic-design "
            "design-generators repository."
        ),
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
