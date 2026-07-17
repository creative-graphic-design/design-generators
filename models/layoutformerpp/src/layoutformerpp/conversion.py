"""Checkpoint conversion helpers for LayoutFormer++."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import ModelCard
from laygen.common.model_card import ParityMetric, build_layout_model_card


LAYOUTFORMERPP_BIBTEX = r"""
@inproceedings{jiang2023layoutformerpp,
  title = {LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Yihan and Deng, Jingwen and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {CVPR},
  year = {2023}
}
"""


def layoutformerpp_hub_id(dataset: str, task: str) -> str:
    """Return the task-specific Hub id for a converted checkpoint."""
    return f"creative-graphic-design/layoutformerpp-{dataset}-{task.replace('_', '-')}"


def load_original_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a published LayoutFormer++ checkpoint and strip DDP prefixes."""
    raw = torch.load(path, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def layoutformerpp_model_card(
    *,
    dataset: str,
    task: str,
    parity_metrics: list[ParityMetric | dict[str, object]] | None = None,
) -> ModelCard:
    """Build a Hub model card for one LayoutFormer++ checkpoint."""
    model_id = layoutformerpp_hub_id(dataset, task)
    condition = {
        "ugen": "unconditional",
        "gen_t": "label",
        "gen_ts": "label_size",
        "gen_r": "relation",
        "completion": "completion",
        "refinement": "refinement",
    }[task]
    dataset_id = {
        "rico": "creative-graphic-design/Rico",
        "publaynet": "creative-graphic-design/PubLayNet",
    }[dataset]
    metrics = parity_metrics or [
        ParityMetric(
            dataset=f"{dataset}_{task}",
            tokenizer_exact="vocab.json exact",
            deterministic_exact="not run",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]
    how_to_use = f"""
from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
)

model = LayoutFormerPPForConditionalGeneration.from_pretrained("{model_id}")
processor = LayoutFormerPPProcessor.from_pretrained("{model_id}")
pipe = LayoutFormerPPPipeline(model=model, processor=processor)

out = pipe(condition_type="{condition}", labels=[["Text"]], max_length=8)
print(out.bbox, out.labels, out.mask)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LayoutFormer++ {dataset} {task}",
        dataset_ids=[dataset_id],
        license="mit",
        library_name="transformers",
        pipeline_tag="text-generation",
        language=["en"],
        tags=[
            "layout-generation",
            "layoutformer++",
            "transformers",
            dataset,
            task,
        ],
        model_details=(
            "Transformers-format conversion of the LayoutFormer++ autoregressive "
            f"layout transformer checkpoint for `{dataset}` / `{task}`. The "
            "processor returns normalized center `xywh` boxes, dataset-local "
            "labels, masks, and `id2label` using the shared `laygen.common` schema."
        ),
        model_type=(
            "Autoregressive Transformer checkpoint converted to a "
            "`transformers.PreTrainedModel` layout-generation package."
        ),
        intended_uses=(
            "Use this checkpoint to reproduce and evaluate LayoutFormer++ "
            f"`{dataset}` / `{task}` conditional graphic layout generation in a "
            "Transformers-style API."
        ),
        downstream_uses=(
            "Generated layouts can be consumed by research pipelines for layout "
            "rendering, design analysis, metric evaluation, or as structured "
            "conditioning for downstream design tools after task-specific checks."
        ),
        out_of_scope_uses=(
            "This checkpoint is not intended for general image generation, OCR, "
            "document understanding, or unsupervised production design decisions. "
            "It should not be treated as a safety-validated user interface system."
        ),
        limitations=(
            "This conversion preserves the released LayoutFormer++ checkpoint "
            "contract and inherits the dataset and task coverage of the original "
            "research release. Current local parity covers the `rico_gen_t` "
            "teacher-forced logits; full constrained-decoding generation parity "
            "for every public task remains follow-up work."
        ),
        recommendations=(
            "Use the package-local vendor parity commands before publishing a "
            "converted checkpoint, inspect layout samples for the target task, and "
            "report dataset/task-specific metrics with any downstream release."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` using the "
            "preprocessed LayoutFormer++ release artifacts from `jzy124/LayoutFormer`."
        ),
        training_procedure=(
            "This repository converts the released PyTorch checkpoint and does not "
            "retrain or fine-tune it. The original LayoutFormer++ training "
            "procedure serializes layout constraints into tokens and trains an "
            "autoregressive transformer with task-specific decoding restrictions."
        ),
        testing_data=(
            "Local tests use fixtures from `jzy124/LayoutFormer` checkpoint files "
            "and vendor reference metadata generated from "
            "`vendor/ms-layout-generation/LayoutFormer++`."
        ),
        evaluation_factors=(
            "The current parity gate checks the RICO `gen_t` checkpoint, exact "
            "`vocab.json` token mapping, and teacher-forced logits for a fixed "
            "reference example."
        ),
        evaluation_metrics=(
            "Tokenizer equality and maximum absolute/relative logit differences "
            "against the vendor implementation."
        ),
        parity_metrics=metrics,
        technical_specs=(
            "The converted artifact stores a `LayoutFormerPPConfig`, "
            "`LayoutFormerPPForConditionalGeneration` weights, tokenizer files, "
            "and a `LayoutFormerPPProcessor`. Public outputs use normalized center "
            "`xywh` boxes, dataset-local labels, masks, and `id2label` through "
            "`laygen.common.outputs.LayoutGenerationOutput`."
        ),
        citation_bibtex=LAYOUTFORMERPP_BIBTEX,
        original_implementation_url=(
            "https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B"
        ),
    )


def write_layoutformerpp_model_card(
    output_dir: Path,
    *,
    dataset: str,
    task: str,
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
