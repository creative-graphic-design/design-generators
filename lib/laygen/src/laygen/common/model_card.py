"""Model-card builders shared by converted layout model packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final, TypeAlias, TypedDict, cast

from huggingface_hub import ModelCard, ModelCardData

from .labels import DatasetName
from .serialization import sanitize_for_yaml


class ModelCardMetadataKey(StrEnum):
    """YAML metadata keys emitted by generated Hub model cards."""

    model_name = auto()
    license = auto()
    library_name = auto()
    pipeline_tag = auto()
    tags = auto()
    datasets = auto()
    language = auto()


MODEL_CARD_METADATA_KEYS: Final[tuple[ModelCardMetadataKey, ...]] = tuple(
    ModelCardMetadataKey
)


class ModelCardMetadata(TypedDict):
    """Structured metadata passed to ``ModelCardData``."""

    model_name: str
    license: str
    library_name: str
    pipeline_tag: str
    tags: list[str]
    datasets: list[str]
    language: list[str]


class ParityMetricKey(StrEnum):
    """Column keys used in generated parity metric rows."""

    dataset = auto()
    tokenizer_exact = auto()
    deterministic_exact = auto()
    logits_max_abs = auto()
    logits_max_rel = auto()


PARITY_METRIC_KEYS: Final[tuple[ParityMetricKey, ...]] = tuple(ParityMetricKey)


@dataclass(frozen=True)
class ParityMetric:
    """Reference-parity metric row included in generated model cards.

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


class ParityMetricRow(TypedDict):
    """Structured parity metric row accepted by model-card generation."""

    dataset: str
    tokenizer_exact: str
    deterministic_exact: str
    logits_max_abs: float
    logits_max_rel: float


ParityMetricInput: TypeAlias = ParityMetric | ParityMetricRow | Mapping[str, object]


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
    parity_metrics: Sequence[ParityMetricInput],
    citation_bibtex: str,
    original_implementation_url: str,
    model_summary: str | None = None,
    developers: str | None = None,
    model_type: str = "Layout generation model.",
    base_model: str | None = None,
    paper: str | None = None,
    preprocessing: str | None = None,
    training_regime: str | None = None,
    testing_data: str | None = None,
    testing_metrics: str | None = None,
    results_summary: str | None = None,
    model_specs: str | None = None,
    compute_infrastructure: str | None = None,
    hardware_requirements: str | None = None,
    software: str | None = None,
    citation_apa: str | None = None,
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
        model_summary: Short model summary for the card header.
        developers: Original developer attribution.
        model_type: User-facing model family/type.
        base_model: Base-model or conversion-relationship statement.
        paper: Paper URL.
        preprocessing: Preprocessing description.
        training_regime: Training-regime description.
        testing_data: Evaluation data or fixture description.
        testing_metrics: Evaluation metric description.
        results_summary: Summary of recorded parity results.
        model_specs: Architecture and objective summary.
        compute_infrastructure: Conversion/parity compute description.
        hardware_requirements: Runtime and parity hardware requirements.
        software: Runtime and parity software requirements.
        citation_apa: Optional APA-style citation.

    Returns:
        Validated ``huggingface_hub.ModelCard`` instance.

    Raises:
        ValueError: If model-card metadata validation fails.

    Examples:
        >>> card = layoutdm_model_card(dataset="rico25")
        >>> card.data.to_dict()["library_name"]
        'diffusers'
    """
    metadata = _model_card_metadata(
        model_name=model_name,
        license=license,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        tags=tags,
        dataset_ids=dataset_ids,
    )
    card_data = ModelCardData(**metadata)
    parity_table = _parity_table(parity_metrics)
    card = ModelCard.from_template(
        card_data,
        model_id=model_id,
        model_summary=(
            model_summary
            or f"{model_name} is a converted checkpoint for layout generation."
        ),
        model_description=model_details,
        developers=developers or "See the original implementation and citation.",
        funded_by=(
            "Funding for the original checkpoint is not separately reported in "
            "this converted artifact."
        ),
        shared_by="creative-graphic-design",
        model_type=model_type,
        language=(
            "The model does not process natural language inputs; metadata uses "
            "English for this model card and category label names."
        ),
        license=license,
        base_model=(
            base_model
            or "Not applicable. This is a conversion of the original checkpoint, "
            "not a fine-tuned derivative of a Hub model."
        ),
        repo=original_implementation_url,
        paper=paper or "See the citation and original implementation.",
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
            preprocessing
            or "Package adapters convert upstream layout representations to "
            "normalized center `xywh` boxes, dataset-local labels, and mask-based "
            "padding at the public API boundary."
        ),
        training_regime=(
            training_regime
            or "Original upstream training regime; this package converts released "
            "artifacts and does not retrain them."
        ),
        speeds_sizes_times=(
            "Training speed, elapsed time, and hardware are not included in "
            "the upstream checkpoint bundle used for conversion."
        ),
        testing_data=(
            testing_data
            or "Reference parity tests use local outputs from the original "
            "implementation for each converted dataset or checkpoint."
        ),
        testing_factors=(
            "Parity is checked separately for each dataset conversion so that "
            "dataset-specific tokenization and checkpoint weights are covered."
        ),
        testing_metrics=(
            testing_metrics
            or "Recorded parity metrics report exact-match counts or numeric "
            "maximum absolute and relative errors against the original "
            "implementation."
        ),
        results=parity_table,
        results_summary=(
            results_summary
            or "Recorded parity results are listed in the table above; see the "
            "package README for commands that regenerate local references."
        ),
        model_examination=(
            "No separate interpretability study is packaged with this converted "
            "checkpoint."
        ),
        hardware_type=(
            "Original training hardware is not reported in this converted "
            "artifact. Reference regeneration is documented for "
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
            model_specs
            or "The converted package exposes the model, preprocessing, and "
            "pipeline or agent components needed to reproduce local inference."
        ),
        compute_infrastructure=(
            compute_infrastructure
            or "Conversion and parity generation run locally through the `uv` "
            "workspace commands documented in the package README."
        ),
        hardware_requirements=(
            hardware_requirements
            or "CPU is sufficient for package loading and lightweight smoke tests. "
            "CUDA may be required for heavyweight reference parity depending on "
            "the original implementation."
        ),
        software=(
            software
            or "Python 3.11+, the package workspace dependencies, and any optional "
            "original-code dependencies documented by the package."
        ),
        citation_bibtex=f"```bibtex\n{citation_bibtex.strip()}\n```",
        citation_apa=citation_apa or "See the BibTeX citation above.",
        glossary=(
            "`xywh` means normalized center-x, center-y, width, and height. "
            "`Tokenizer exact` counts matching encoded and decoded token "
            "positions. `Logits max abs` and `logits max rel` are maximum "
            "differences against the original denoiser outputs."
        ),
        more_information=(
            "See the package README for copy-paste reproduction commands, "
            "checkpoint conversion, and reference fixture generation."
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
    parity_metrics: Sequence[ParityMetricInput] | None = None,
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
        ['creative-graphic-design/PubLayNet']
    """
    dataset_id = _layoutdm_dataset_id(dataset)
    dataset_config = _layoutdm_dataset_config(dataset)
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

path = ".cache/layout-dm/converted/layoutdm-{dataset}"
# After Hub publication: from_pretrained("{model_id}")
pipe = LayoutDMPipeline.from_pretrained(path)
out = pipe(batch_size=1, seed=0, sampling="deterministic")
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=model_name,
        dataset_ids=[dataset_id],
        license="apache-2.0",
        library_name="diffusers",
        pipeline_tag="other",
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
            f"The original checkpoint was trained on `{dataset_id}`"
            f"{dataset_config} as released by "
            "the original LayoutDM project."
        ),
        parity_metrics=metrics,
        citation_bibtex=_LAYOUTDM_BIBTEX,
        original_implementation_url=("https://github.com/CyberAgentAILab/layout-dm"),
        model_summary=(
            f"{model_name} is a Diffusers-format LayoutDM checkpoint for "
            "conditional-free layout generation."
        ),
        developers="CyberAgentAILab released the original LayoutDM implementation.",
        model_type="Discrete diffusion model for layout generation.",
        base_model=(
            "Not applicable. This is a direct conversion of the original "
            "LayoutDM checkpoint, not a fine-tuned derivative of a Hub model."
        ),
        paper="https://arxiv.org/abs/2303.08137",
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
        testing_data=(
            "Reference parity tests use deterministic samples and forward-pass "
            "golden tensors generated from the original LayoutDM implementation "
            "for each converted dataset."
        ),
        testing_metrics=(
            "Tokenizer exact-match count, deterministic token-sequence "
            "exact-match count, and denoiser logits maximum absolute and "
            "relative error versus the original implementation."
        ),
        results_summary=(
            "The converted checkpoint matches the generated reference "
            "tensors exactly for tokenizer IO and deterministic sampling; "
            "denoiser logits are within the reported numeric tolerance."
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
            "recommended for regenerating reference parity outputs and running "
            "the full parity test suite."
        ),
        software=(
            "Python 3.11+, PyTorch, Diffusers, Transformers, and the optional "
            "LayoutDM original-code dependencies declared by the `layout-dm` package."
        ),
        citation_apa=(
            "Inoue, N., Kikuchi, K., Simo-Serra, E., Otani, M., & Yamaguchi, K. "
            "(2023). LayoutDM: Discrete Diffusion Model for Controllable Layout "
            "Generation. CVPR."
        ),
    )


def _layoutdm_dataset_id(dataset: DatasetName | str) -> str:
    if dataset == "rico25":
        return "creative-graphic-design/Rico"
    if dataset == "publaynet":
        return "creative-graphic-design/PubLayNet"
    raise ValueError(f"Unsupported LayoutDM dataset: {dataset}")


def _layoutdm_dataset_config(dataset: DatasetName | str) -> str:
    if dataset == "rico25":
        return " with config `ui-screenshots-and-hierarchies-with-semantic-annotations`"
    if dataset == "publaynet":
        return " with the default config"
    raise ValueError(f"Unsupported LayoutDM dataset: {dataset}")


def _model_card_metadata(
    *,
    model_name: str,
    license: str,
    library_name: str,
    pipeline_tag: str,
    tags: Sequence[str],
    dataset_ids: Sequence[str],
) -> ModelCardMetadata:
    return {
        ModelCardMetadataKey.model_name.value: cast(str, sanitize_for_yaml(model_name)),
        ModelCardMetadataKey.license.value: cast(str, sanitize_for_yaml(license)),
        ModelCardMetadataKey.library_name.value: cast(
            str, sanitize_for_yaml(library_name)
        ),
        ModelCardMetadataKey.pipeline_tag.value: cast(
            str, sanitize_for_yaml(pipeline_tag)
        ),
        ModelCardMetadataKey.tags.value: cast(list[str], sanitize_for_yaml(list(tags))),
        ModelCardMetadataKey.datasets.value: cast(
            list[str], sanitize_for_yaml(list(dataset_ids))
        ),
        ModelCardMetadataKey.language.value: ["en"],
    }


def _parity_table(metrics: Sequence[ParityMetricInput]) -> str:
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


def _metric_dict(metric: ParityMetricInput) -> dict[str, object]:
    if isinstance(metric, ParityMetric):
        return {
            ParityMetricKey.dataset.value: metric.dataset,
            ParityMetricKey.tokenizer_exact.value: metric.tokenizer_exact,
            ParityMetricKey.deterministic_exact.value: metric.deterministic_exact,
            ParityMetricKey.logits_max_abs.value: metric.logits_max_abs,
            ParityMetricKey.logits_max_rel.value: metric.logits_max_rel,
        }
    return dict(metric)


_LAYOUTDM_BIBTEX: Final[str] = r"""
@inproceedings{inoue2023layoutdm,
  title = {LayoutDM: Discrete Diffusion Model for Controllable Layout Generation},
  author = {Inoue, Naoto and Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {CVPR},
  year = {2023}
}
"""
