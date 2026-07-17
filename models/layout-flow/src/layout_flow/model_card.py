"""Model-card generation for converted LayoutFlow checkpoints."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import ModelCard
from laygen.common.model_card import build_layout_model_card


LAYOUTFLOW_BIBTEX = r"""
@inproceedings{guerreiro2024layoutflow,
  title={LayoutFlow: Flow Matching For Layout Generation},
  author={Guerreiro, Julian Jorge Andrade and Inoue, Naoto and Masui, Kento and Otani, Mayu and Nakayama, Hideki},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2024},
  organization={Springer}
}
"""


def layoutflow_model_card(dataset: str) -> ModelCard:
    """Build a model card for a converted LayoutFlow checkpoint.

    Args:
        dataset: LayoutFlow dataset name or alias.

    Returns:
        Validated Hugging Face model card.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Examples:
        >>> card = layoutflow_model_card("publaynet")
        >>> card.data.to_dict()["library_name"]
        'diffusers'
    """
    dataset = _normalize_dataset(dataset)
    model_id = f"creative-graphic-design/layout-flow-{dataset}"
    dataset_id = _dataset_id(dataset)
    how_to_use = f"""
from layout_flow import LayoutFlowPipeline

pipe = LayoutFlowPipeline.from_pretrained("{model_id}")
out = pipe(batch_size=1, num_elements=8, seed=0, num_inference_steps=100)
print(out.bbox, out.labels, out.mask, out.id2label)
"""
    return build_layout_model_card(
        model_id=model_id,
        model_name=f"LayoutFlow {dataset}",
        dataset_ids=[dataset_id],
        license="mit",
        library_name="diffusers",
        pipeline_tag="text-to-image",
        tags=[
            "layout-generation",
            "layout-flow",
            "flow-matching",
            "diffusers",
            dataset,
        ],
        model_summary=(
            "Diffusers-format conversion of the LayoutFlow flow-matching layout "
            f"generation checkpoint for `{dataset}`."
        ),
        model_details=(
            "Diffusers-format conversion of the LayoutFlow checkpoint for "
            f"`{dataset}`. LayoutFlow is a flow-matching layout generator that "
            "integrates a learned vector field over continuous geometry and "
            "analog-bit category labels. Public outputs are normalized center "
            "`xywh` boxes, dataset-local labels, masks, and `id2label` metadata."
        ),
        developed_by=(
            "Julian Guerreiro, Naoto Inoue, Kento Masui, Mayu Otani, and Hideki "
            "Nakayama; converted by creative-graphic-design."
        ),
        model_type="Flow-matching transformer model for layout generation.",
        repo_url="https://github.com/julianguerreiro/LayoutFlow",
        paper_url="https://arxiv.org/abs/2403.18187",
        demo_url="https://julianguerreiro.github.io/layoutflow/",
        direct_use=(
            "Use this checkpoint for research and evaluation of UI/document "
            "layout generation and controllable layout completion workflows."
        ),
        downstream_use=(
            "Generated layouts can seed design-tool prototypes, dataset analysis, "
            "or downstream rendering systems after task-specific validation."
        ),
        out_of_scope_use=(
            "Do not use this checkpoint as an image renderer, OCR system, "
            "accessibility checker, safety classifier, or an automated source of "
            "production UI/document decisions without human review."
        ),
        limitations=(
            "This checkpoint generates layout geometry and category labels only; "
            "it does not render images or text. Generation is stochastic unless a "
            "`torch.Generator` or `seed` is supplied. The converted artifact "
            "preserves the original checkpoint behavior and should be evaluated "
            "within the original dataset domain."
        ),
        recommendations=(
            "Validate outputs on the intended document or UI domain, keep humans "
            "in the loop for design decisions, and treat category taxonomies as "
            "dataset-specific rather than universal."
        ),
        how_to_use=how_to_use,
        training_data=(
            f"The original checkpoint was trained on `{dataset_id}` using the "
            "splits distributed by the LayoutFlow authors through "
            "`JulianGuerreiro/LayoutFlow`."
        ),
        training_procedure=(
            "The training procedure follows the original LayoutFlow release; this "
            "repository only converts released checkpoints and does not retrain."
        ),
        preprocessing=(
            "Layouts are represented as normalized center `xywh` boxes, padding "
            "masks, and analog-bit encodings of dataset-local category labels."
        ),
        training_regime="Original training regime; not retrained during conversion.",
        speeds_sizes_times="No additional training was run for this conversion.",
        testing_data=(
            "Vendor parity tests use deterministic synthetic fixtures and the "
            f"released `{dataset}` checkpoint from `JulianGuerreiro/LayoutFlow`."
        ),
        testing_factors=(
            "Dataset checkpoint, learned vector field, and Euler ODE trajectory "
            "through the original vendor sampling path."
        ),
        testing_metrics=(
            "Maximum absolute and relative deviation between converted and vendor "
            "vector-field outputs and Euler trajectories."
        ),
        parity_metrics=[
            {
                "dataset": dataset,
                "tokenizer_exact": "n/a",
                "deterministic_exact": "Euler exact",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
        results_summary=(
            "PubLayNet and RICO25 vector-field and Euler-trajectory parity are "
            "exact within the tested tolerances: max_abs=0.0 and max_rel=0.0."
        ),
        model_specs=(
            "LayoutFlow uses a transformer vector-field model over continuous "
            "box coordinates and analog-bit label encodings, sampled with an "
            "increasing-time Euler ODE scheduler from `t=0` to `t=1`."
        ),
        compute_infrastructure=(
            "Inference requires Python, PyTorch, Diffusers, and this workspace "
            "package. GPU is recommended for full parity or batch evaluation."
        ),
        hardware_requirements="CPU works for smoke tests; CUDA is recommended for parity.",
        software="Python 3.11+, PyTorch, Diffusers, and Hugging Face Hub tooling.",
        citation_bibtex=LAYOUTFLOW_BIBTEX,
        citation_apa=(
            "Guerreiro, J. J. A., Inoue, N., Masui, K., Otani, M., & Nakayama, H. "
            "(2024). LayoutFlow: Flow Matching For Layout Generation. ECCV."
        ),
        original_implementation_url="https://github.com/julianguerreiro/LayoutFlow",
        model_card_authors="creative-graphic-design",
        model_card_contact="Use the repository issue tracker for questions.",
    )


def save_layoutflow_model_card(output_dir: str | Path, *, dataset: str) -> Path:
    """Write a LayoutFlow model card as ``README.md``.

    Args:
        output_dir: Directory that receives ``README.md``.
        dataset: LayoutFlow dataset name or alias.

    Returns:
        Path to the written README.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     path = save_layoutflow_model_card(tmp, dataset="publaynet")
        ...     path.name
        'README.md'
    """
    output_path = Path(output_dir) / "README.md"
    output_path.write_text(str(layoutflow_model_card(dataset)), encoding="utf-8")
    return output_path


def _normalize_dataset(dataset: str) -> str:
    if dataset in {"rico", "rico25"}:
        return "rico25"
    if dataset == "publaynet":
        return "publaynet"
    raise ValueError(f"Unsupported LayoutFlow dataset: {dataset}")


def _dataset_id(dataset: str) -> str:
    if dataset == "rico25":
        return "creative-graphic-design/Rico"
    if dataset == "publaynet":
        return "creative-graphic-design/PubLayNet"
    raise ValueError(f"Unsupported LayoutFlow dataset: {dataset}")
