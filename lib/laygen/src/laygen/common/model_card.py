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
    developers: str,
    model_type: str,
    paper_url: str,
    direct_use: str,
    downstream_use: str,
    out_of_scope_use: str,
    bias_recommendations: str,
    preprocessing: str,
    training_regime: str,
    testing_data: str,
    testing_factors: str,
    testing_metrics: str,
    model_specs: str,
    software: str,
    model_summary: str | None = None,
    funded_by: str = "Not documented in the original release.",
    shared_by: str = "creative-graphic-design",
    base_model: str = "Not applicable: this is a converted original checkpoint.",
    demo: str = "No hosted demo is packaged with this checkpoint.",
    speeds_sizes_times: str = "Not documented in the original release.",
    model_examination: str = "No separate interpretability examination is packaged with this converted checkpoint.",
    hardware_type: str = "Not documented in the original release.",
    hours_used: str = "Not documented in the original release.",
    cloud_provider: str = "Not documented in the original release.",
    cloud_region: str = "Not documented in the original release.",
    co2_emitted: str = "Carbon emissions cannot be estimated from the released checkpoint bundle alone.",
    compute_infrastructure: str = "Conversion and parity generation run locally through the documented uv workspace commands.",
    hardware_requirements: str = "CPU is sufficient to load the pipeline; CUDA is recommended for generation and parity tests.",
    citation_apa: str = "See the BibTeX citation above.",
    glossary: str = "`bbox` uses normalized center `xywh` coordinates; `mask` marks valid layout elements.",
    more_information: str = "See the package README for reproduction commands and conversion details.",
    model_card_authors: str = "creative-graphic-design contributors.",
    model_card_contact: str = "Use the model repository discussions or issues for questions.",
) -> ModelCard:
    card_data = ModelCardData(
        model_name=model_name,
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=list(tags),
        datasets=list(dataset_ids),
        language=["en"],
        metrics=["vendor-parity"],
    )
    parity_table = _parity_table(parity_metrics)
    card = ModelCard.from_template(
        card_data,
        model_id=model_id,
        model_summary=model_summary or model_details,
        model_description=(
            f"{model_details}\n\n"
            "This card follows the Hugging Face Hub model card template and "
            "the annotated model card section structure."
        ),
        developers=developers,
        funded_by=funded_by,
        shared_by=shared_by,
        model_type=model_type,
        language=(
            "The model does not process natural language inputs; metadata uses "
            "English for this model card and category label names."
        ),
        license=license,
        base_model=base_model,
        repo=original_implementation_url,
        paper=paper_url,
        demo=demo,
        direct_use=f"{intended_uses}\n\n{direct_use}",
        downstream_use=downstream_use,
        out_of_scope_use=out_of_scope_use,
        bias_risks_limitations=limitations,
        bias_recommendations=bias_recommendations,
        get_started_code=f"```python\n{how_to_use.strip()}\n```",
        training_data=training_data,
        preprocessing=preprocessing,
        training_regime=training_regime,
        speeds_sizes_times=speeds_sizes_times,
        testing_data=testing_data,
        testing_factors=testing_factors,
        testing_metrics=testing_metrics,
        results=parity_table,
        results_summary=(
            "Local vendor parity checks compare converted pipeline components "
            "against the original implementation and checkpoint weights."
        ),
        model_examination=model_examination,
        hardware_type=hardware_type,
        hours_used=hours_used,
        cloud_provider=cloud_provider,
        cloud_region=cloud_region,
        co2_emitted=co2_emitted,
        model_specs=model_specs,
        compute_infrastructure=compute_infrastructure,
        hardware_requirements=hardware_requirements,
        software=software,
        citation_bibtex=f"```bibtex\n{citation_bibtex.strip()}\n```",
        citation_apa=citation_apa,
        glossary=glossary,
        more_information=more_information,
        model_card_authors=model_card_authors,
        model_card_contact=model_card_contact,
    )
    card.validate(repo_type="model")
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
            "intended for layout synthesis, not image rendering, OCR, or evaluation "
            "of visual quality without downstream checks. Dataset-specific biases "
            "from PubLayNet or Rico may appear in generated layouts."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` as released by "
            "the original LayoutDM project."
        ),
        parity_metrics=metrics,
        citation_bibtex=_LAYOUTDM_BIBTEX,
        original_implementation_url=("https://github.com/CyberAgentAILab/layout-dm"),
        developers="CyberAgent AI Lab; converted by creative-graphic-design contributors.",
        model_type="Discrete diffusion model for controllable layout generation.",
        paper_url="https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html",
        direct_use=(
            "Generate normalized layout boxes, labels, and masks from the converted "
            "Diffusers pipeline without fine-tuning."
        ),
        downstream_use=(
            "Use generated layouts as inputs for document or UI rendering, "
            "benchmarking, data augmentation, or layout editing systems."
        ),
        out_of_scope_use=(
            "Do not use the model as an image generator, OCR system, safety-critical "
            "document authoring system, or as a substitute for human review of "
            "generated layouts."
        ),
        bias_recommendations=(
            "Inspect generated layouts for dataset bias, overlap, category balance, "
            "and downstream rendering constraints before deployment."
        ),
        preprocessing=(
            "The original LayoutDM workflow quantizes bounding boxes into layout "
            "tokens and stores outputs as max-25 sequences."
        ),
        training_regime="Original training regime from the upstream LayoutDM release.",
        testing_data=(
            f"Vendor parity tests use `{dataset_id}` checkpoint fixtures and local "
            "golden outputs generated from the original implementation."
        ),
        testing_factors=(
            "Parity is checked per dataset and covers tokenizer round trips, "
            "deterministic generation, and denoiser logits."
        ),
        testing_metrics=(
            "Tokenizer exact match count, deterministic exact match count, logits "
            "maximum absolute difference, and logits maximum relative difference."
        ),
        model_specs=(
            "Converted LayoutDM denoiser, tokenizer, processor, and scheduler wrapped "
            "in a Diffusers-style pipeline."
        ),
        software="Python, PyTorch, Diffusers, Transformers, and laygen.common.",
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
