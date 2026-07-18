"""Checkpoint conversion helpers for LayoutFormer++."""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

import torch
from huggingface_hub import ModelCard
from laygen.common import ConditionType, DatasetName
from laygen.common.model_card import ParityMetric, build_layout_model_card

from .tasks import (
    LayoutFormerPPTask,
    TASK_TO_CONDITION,
    layoutformerpp_dataset_slug,
    normalize_layoutformerpp_dataset,
    normalize_layoutformerpp_task,
)


LAYOUTFORMERPP_BIBTEX = r"""
@inproceedings{jiang2023layoutformerpp,
  title = {LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Yihan and Deng, Jingwen and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {CVPR},
  year = {2023}
}
"""


class DatasetCardMetadata(TypedDict):
    """Hub-facing metadata for one LayoutFormer++ dataset."""

    hub_slug: str
    dataset_id: str


DATASET_CARD_METADATA: Final[dict[DatasetName, DatasetCardMetadata]] = {
    DatasetName.rico25: {
        "hub_slug": "rico",
        "dataset_id": "creative-graphic-design/Rico",
    },
    DatasetName.publaynet: {
        "hub_slug": "publaynet",
        "dataset_id": "creative-graphic-design/PubLayNet",
    },
}


def layoutformerpp_hub_id(
    dataset: DatasetName | str,
    task: LayoutFormerPPTask | ConditionType | str,
) -> str:
    """Return the task-specific Hub id for a converted checkpoint."""
    normalized_task = normalize_layoutformerpp_task(task)
    suffix = TASK_TO_CONDITION[normalized_task].replace("_", "-")
    return (
        "creative-graphic-design/layoutformerpp-"
        f"{layoutformerpp_dataset_slug(dataset)}-{suffix}"
    )


def load_original_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a published LayoutFormer++ checkpoint and strip DDP prefixes."""
    raw = torch.load(path, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def layoutformerpp_model_card(
    *,
    dataset: DatasetName | str,
    task: LayoutFormerPPTask | ConditionType | str,
    parity_metrics: list[ParityMetric | dict[str, object]] | None = None,
) -> ModelCard:
    """Build a Hub model card for one LayoutFormer++ checkpoint."""
    normalized_dataset = normalize_layoutformerpp_dataset(dataset)
    normalized_task = normalize_layoutformerpp_task(task)
    model_id = layoutformerpp_hub_id(normalized_dataset, normalized_task)
    condition = TASK_TO_CONDITION[normalized_task]
    dataset_metadata = DATASET_CARD_METADATA[normalized_dataset]
    dataset_slug = dataset_metadata["hub_slug"]
    dataset_id = dataset_metadata["dataset_id"]
    metrics = parity_metrics or [
        ParityMetric(
            dataset=f"{dataset_slug}_{normalized_task}",
            tokenizer_exact="vocab.json exact",
            deterministic_exact="not run",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]
    how_to_use = f"""
from layoutformerpp import LayoutFormerPPPipeline

pipe = LayoutFormerPPPipeline.from_pretrained("{model_id}")

out = pipe(condition_type="{condition}", labels=[["Text"]], max_length=8)
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LayoutFormer++ {dataset_slug} {normalized_task}",
        dataset_ids=[dataset_id],
        license="mit",
        library_name="transformers",
        pipeline_tag="text-generation",
        tags=[
            "layout-generation",
            "layoutformer++",
            "transformers",
            dataset_slug,
            str(normalized_task),
        ],
        model_details=(
            "Transformers-format conversion of the LayoutFormer++ autoregressive "
            f"layout transformer checkpoint for `{dataset_slug}` / "
            f"`{normalized_task}`. The "
            "processor returns normalized center `xywh` boxes, dataset-local "
            "labels, masks, and `id2label` using the shared `laygen.common` schema."
        ),
        intended_uses=(
            "Use this checkpoint to reproduce and evaluate LayoutFormer++ "
            f"`{dataset_slug}` / `{normalized_task}` conditional graphic layout "
            "generation in a Transformers-style API."
        ),
        limitations=(
            "This conversion preserves the released LayoutFormer++ checkpoint "
            "contract and inherits the dataset and task coverage of the original "
            "research release. Local vendor parity covers tokenizer behavior, "
            "teacher-forced logits, vendor greedy/top-k generation, and constrained "
            "label or label-size generation for every public `rico` and `publaynet` "
            "LayoutFormer++ task checkpoint. This checkpoint is not intended for "
            "OCR, document understanding, or unreviewed production design decisions."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` using the "
            "preprocessed LayoutFormer++ release artifacts from `jzy124/LayoutFormer`."
        ),
        parity_metrics=metrics,
        citation_bibtex=LAYOUTFORMERPP_BIBTEX,
        original_implementation_url=(
            "https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B"
        ),
    )


def write_layoutformerpp_model_card(
    output_dir: Path,
    *,
    dataset: DatasetName | str,
    task: LayoutFormerPPTask | ConditionType | str,
    parity_metrics: list[ParityMetric | dict[str, object]] | None = None,
) -> Path:
    """Write the checkpoint README model card next to converted weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        str(
            layoutformerpp_model_card(
                dataset=dataset,
                task=task,
                parity_metrics=parity_metrics,
            )
        ),
        encoding="utf-8",
    )
    return readme_path
