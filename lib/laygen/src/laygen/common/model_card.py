"""Model-card builders shared by converted layout model packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from huggingface_hub import ModelCard, ModelCardData

from .labels import DatasetName
from .serialization import sanitize_for_yaml


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
        model_name=cast(str, sanitize_for_yaml(model_name)),
        license=cast(str, sanitize_for_yaml(license)),
        library_name=cast(str, sanitize_for_yaml(library_name)),
        pipeline_tag=cast(str, sanitize_for_yaml(pipeline_tag)),
        tags=cast(list[str], sanitize_for_yaml(list(tags))),
        datasets=cast(list[str], sanitize_for_yaml(list(dataset_ids))),
        language=["en"],
    )
    parity_table = _parity_table(parity_metrics)
    card = ModelCard.from_template(
        card_data,
        model_id=model_id,
        model_summary=(
            f"{model_name} is a Diffusers-format LayoutDM checkpoint for "
            "conditional-free layout generation."
        ),
        model_description=model_details,
        developers="CyberAgentAILab released the original LayoutDM implementation.",
        funded_by=(
            "Funding for the original checkpoint is not separately reported in "
            "this converted artifact."
        ),
        shared_by="creative-graphic-design",
        model_type="Discrete diffusion model for layout generation.",
        language=(
            "The model does not process natural language inputs; metadata uses "
            "English for this model card and category label names."
        ),
        license=license,
        base_model=(
            "Not applicable. This is a direct conversion of the original "
            "LayoutDM checkpoint, not a fine-tuned derivative of a Hub model."
        ),
        repo=original_implementation_url,
        paper="https://arxiv.org/abs/2303.08137",
        demo="No hosted demo is packaged with this checkpoint.",
        direct_use=intended_uses,
        downstream_use=(
            "Use the generated normalized boxes, labels, and masks as layout "
            "priors for design tooling, document analysis research, or "
            "controlled rendering pipelines that perform their own validation."
        ),
        out_of_scope_use=(
            "Do not use this checkpoint as an OCR model, image renderer, "
            "semantic document understanding model, accessibility verifier, or "
            "unreviewed production UI generator. The model predicts layout "
            "structure only and can produce implausible or overlapping boxes."
        ),
        bias_risks_limitations=limitations,
        bias_recommendations=(
            "Inspect generated layouts before downstream use, validate boxes "
            "against application constraints, and evaluate separately for each "
            "target dataset or design domain."
        ),
        get_started_code=f"```python\n{how_to_use.strip()}\n```",
        training_data=training_data,
        preprocessing=(
            "The converted tokenizer represents each layout element as "
            "discrete category and bounding-box tokens. Bounding boxes use "
            "normalized center `xywh` coordinates and dataset-specific cluster "
            "centers stored with the tokenizer files."
        ),
        training_regime=(
            "Original LayoutDM training regime as released by the upstream "
            "project; this package converts the checkpoint and does not "
            "retrain it."
        ),
        speeds_sizes_times=(
            "Training speed, elapsed time, and hardware are not included in "
            "the upstream checkpoint bundle used for conversion."
        ),
        testing_data=(
            "Vendor parity tests use deterministic samples and forward-pass "
            "golden tensors generated from the original LayoutDM implementation "
            "for each converted dataset."
        ),
        testing_factors=(
            "Parity is checked separately for each dataset conversion so that "
            "dataset-specific tokenization and checkpoint weights are covered."
        ),
        testing_metrics=(
            "Tokenizer exact-match count, deterministic token-sequence "
            "exact-match count, and denoiser logits maximum absolute and "
            "relative error versus the original implementation."
        ),
        results=parity_table,
        results_summary=(
            "The converted checkpoint matches the generated vendor reference "
            "tensors exactly for tokenizer IO and deterministic sampling; "
            "denoiser logits are within the reported numeric tolerance."
        ),
        model_examination=(
            "No separate interpretability study is packaged with this converted "
            "checkpoint."
        ),
        hardware_type=(
            "Original training hardware is not reported in this converted "
            "artifact. Vendor parity regeneration is documented for "
            "`CUDA_VISIBLE_DEVICES=0` when a CUDA device is available."
        ),
        hours_used=(
            "Original training hours are not reported in this converted artifact."
        ),
        cloud_provider=(
            "Original training cloud provider is not reported in this converted "
            "artifact."
        ),
        cloud_region=(
            "Original training compute region is not reported in this converted "
            "artifact."
        ),
        co2_emitted=(
            "Carbon emissions cannot be estimated from the released checkpoint "
            "bundle alone."
        ),
        model_specs=(
            "LayoutDM models layout generation as discrete diffusion over "
            "category and bounding-box token sequences. This package exposes "
            "the denoiser, tokenizer, scheduler, and Diffusers pipeline needed "
            "to reproduce converted inference."
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
        citation_bibtex=f"```bibtex\n{citation_bibtex.strip()}\n```",
        citation_apa=(
            "Inoue, N., Kikuchi, K., Simo-Serra, E., Otani, M., & Yamaguchi, K. "
            "(2023). LayoutDM: Discrete Diffusion Model for Controllable Layout "
            "Generation. CVPR."
        ),
        glossary=(
            "`xywh` means normalized center-x, center-y, width, and height. "
            "`Tokenizer exact` counts matching encoded and decoded token "
            "positions. `Logits max abs` and `logits max rel` are maximum "
            "differences against the original denoiser outputs."
        ),
        more_information=(
            "See the package README for copy-paste reproduction commands, "
            "checkpoint conversion, and vendor parity fixture generation."
        ),
        model_card_authors="creative-graphic-design maintainers.",
        model_card_contact=(
            "Open an issue or pull request in the creative-graphic-design "
            "design-generators repository."
        ),
    )
    card.validate()
    return card


def layoutdm_model_card(
    *,
    dataset: DatasetName | str,
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


def _layoutdm_dataset_id(dataset: DatasetName | str) -> str:
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
