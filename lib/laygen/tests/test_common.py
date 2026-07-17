from dataclasses import dataclass

import matplotlib.pyplot as plt
import pytest
import torch
from transformers.utils import ModelOutput

from laygen.common.bbox import (
    BoxFormat,
    denormalize_boxes,
    linear_continuize,
    linear_discretize,
    ltrb_to_xywh,
    normalize_boxes,
    xywh_to_ltrb,
    xywh_to_ltwh,
)
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.common.discrete import (
    SamplingMode,
    batch_topk_mask,
    extract,
    gumbel_noise_like,
    index_to_log_onehot,
    log_add_exp,
    log_onehot_to_index,
    log_sample_categorical,
    normalize_sampling_mode,
    sample_categorical,
    top_k_logits,
)
from laygen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    label2id_for_dataset,
    labels_for_dataset,
    normalize_dataset_name,
)
from laygen.common.model_card import layoutdm_model_card
from laygen.common.outputs import LayoutGenerationOutput
from laygen.common.outputs_diffusers import (
    LayoutGenerationOutput as DiffusersLayoutGenerationOutput,
)
from laygen.common.testing import (
    assert_generator_reproducible,
    assert_layout_output_schema,
    assert_normalized_xywh,
)
from laygen.common.visualization import render_layout


def test_bbox_conversions_roundtrip():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
    assert torch.allclose(ltrb_to_xywh(xywh_to_ltrb(bbox)), bbox)
    assert torch.allclose(xywh_to_ltwh(bbox), torch.tensor([[[0.4, 0.3, 0.2, 0.4]]]))
    pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltrb")
    assert torch.allclose(
        normalize_boxes(pixels, canvas_size=(100, 200), box_format="ltrb"), bbox
    )
    for box_format in (BoxFormat.ltrb, BoxFormat.ltwh, BoxFormat.xywh):
        pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format=box_format)
        assert torch.allclose(
            normalize_boxes(pixels, canvas_size=(100, 200), box_format=box_format),
            bbox,
        )
    with pytest.raises(ValueError, match="Unsupported box_format"):
        normalize_boxes(pixels, canvas_size=(100, 200), box_format="bad")


def test_linear_bins_roundtrip_shape():
    values = torch.tensor([0.0, 0.25, 0.99])
    ids = linear_discretize(values, num_bins=4)
    assert ids.tolist() == [0, 1, 3]
    assert linear_continuize(ids, num_bins=4).shape == values.shape


def test_discrete_log_onehot_roundtrip():
    ids = torch.tensor([[0, 2, 1]])
    assert torch.equal(log_onehot_to_index(index_to_log_onehot(ids, 3)), ids)
    with pytest.raises(ValueError, match="exceeds vocab_size"):
        index_to_log_onehot(torch.tensor([[3]]), 3)


def test_discrete_sampling_helpers_cover_modes():
    logits = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    assert normalize_sampling_mode(SamplingMode.random) is SamplingMode.random
    with pytest.raises(ValueError, match="Unsupported sampling mode"):
        normalize_sampling_mode("unknown")
    assert torch.equal(
        sample_categorical(logits, sampling=SamplingMode.deterministic),
        torch.tensor([[2, 0]]),
    )

    generator = torch.Generator().manual_seed(0)
    for mode in (
        SamplingMode.random,
        SamplingMode.gumbel,
        SamplingMode.top_k,
        SamplingMode.top_p,
        SamplingMode.top_k_top_p,
    ):
        sampled = sample_categorical(
            logits,
            sampling=mode,
            temperature=0.7,
            top_k=2,
            top_p=0.8,
            generator=generator,
        )
        assert sampled.shape == logits.shape[:-1]

    assert torch.equal(top_k_logits(logits, k=0), logits)
    masked = top_k_logits(logits, k=1)
    assert masked[0, 0, 0].item() == pytest.approx(-70.0)
    assert torch.allclose(
        log_add_exp(torch.tensor([0.0]), torch.tensor([0.0])),
        torch.tensor([torch.log(torch.tensor(2.0))]),
    )
    assert extract(
        torch.arange(4.0), torch.tensor([1, 3]), torch.Size([2, 3, 4])
    ).shape == (
        2,
        1,
        1,
    )
    assert gumbel_noise_like(torch.zeros(2, 3), generator=generator).shape == (2, 3)
    assert log_sample_categorical(logits.movedim(-1, 1), generator=generator).shape == (
        1,
        2,
    )


def test_batch_topk_mask_edges():
    scores = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 0.0]])
    assert torch.equal(
        batch_topk_mask(scores, torch.tensor([2, 0])),
        torch.tensor([[False, True, True], [False, False, False]]),
    )
    assert not batch_topk_mask(scores, torch.tensor([0, 0])).any()
    with pytest.raises(ValueError, match="rank-2"):
        batch_topk_mask(scores.unsqueeze(0), torch.tensor([1]))


def test_label_registry_aliases_and_errors():
    assert normalize_dataset_name(DatasetName.rico25) is DatasetName.rico25
    assert normalize_dataset_name("rico25-max25") is DatasetName.rico25
    assert labels_for_dataset("publaynet") == (
        "text",
        "title",
        "list",
        "table",
        "figure",
    )
    assert label2id_for_dataset("rico25")["Text"] == 0
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        normalize_dataset_name("unknown")


def test_condition_registry_aliases_and_errors():
    assert normalize_condition_type(ConditionType.label) is ConditionType.label
    assert normalize_condition_type("cat_cond") is ConditionType.label
    assert normalize_condition_type("label-size") is ConditionType.label_size
    assert normalize_condition_type("refine") is ConditionType.refinement
    with pytest.raises(ValueError, match="Unsupported condition_type"):
        normalize_condition_type("unknown")


def test_output_schema():
    output = LayoutGenerationOutput(
        bbox=torch.zeros(1, 2, 4),
        labels=torch.zeros(1, 2, dtype=torch.long),
        mask=torch.tensor([[True, False]]),
        id2label=id2label_for_dataset(DatasetName.publaynet),
    )
    assert_layout_output_schema(output, batch_size=1)


@dataclass
class _ToyOutput:
    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]


def test_testing_helper_checks_generator_reproducibility():
    def sample(*, generator: torch.Generator) -> _ToyOutput:
        return _ToyOutput(
            bbox=torch.rand(1, 2, 4, generator=generator),
            labels=torch.zeros(1, 2, dtype=torch.long),
            mask=torch.ones(1, 2, dtype=torch.bool),
            id2label=id2label_for_dataset(DatasetName.publaynet),
        )

    assert_generator_reproducible(sample)
    assert_normalized_xywh(
        torch.tensor([[[2.0, 2.0, 1.0, 1.0]]]), torch.tensor([[False]])
    )


def test_render_layout_adds_patches_and_text():
    bbox = torch.tensor([[0.5, 0.5, 0.4, 0.2], [0.2, 0.2, 0.1, 0.1]])
    labels = torch.tensor([0, 99])
    mask = torch.tensor([True, True])
    ax = render_layout(
        bbox,
        labels,
        mask,
        {0: "text"},
        canvas_size=(100, 200),
        colors=["red", "blue"],
    )
    assert len(ax.patches) == 2
    assert [text.get_text() for text in ax.texts] == ["text", "99"]
    plt.close("all")


def test_output_variants_share_schema_and_mapping_behavior():
    bbox = torch.zeros(1, 2, 4)
    labels = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.tensor([[True, False]])
    id2label = id2label_for_dataset("publaynet")
    canonical = LayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label=id2label,
    )
    diffusers = DiffusersLayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label=id2label,
    )

    assert isinstance(canonical, ModelOutput)
    assert_layout_output_schema(canonical, batch_size=1)
    assert_layout_output_schema(diffusers, batch_size=1)
    assert canonical["bbox"] is canonical.bbox
    assert diffusers["bbox"] is diffusers.bbox
    assert "scores" not in canonical
    assert "scores" not in diffusers
    assert canonical.to_tuple() == (
        canonical.bbox,
        canonical.labels,
        canonical.mask,
        canonical.id2label,
    )
    assert diffusers.to_tuple() == (
        diffusers.bbox,
        diffusers.labels,
        diffusers.mask,
        diffusers.id2label,
    )


def test_layoutdm_model_card_metadata_and_sections():
    card = layoutdm_model_card(dataset="rico25")
    metadata = card.data.to_dict()
    text = str(card)

    assert metadata["license"] == "apache-2.0"
    assert metadata["library_name"] == "diffusers"
    assert metadata["pipeline_tag"] == "text-to-image"
    assert metadata["language"] == ["en"]
    assert "layout-generation" in metadata["tags"]
    assert metadata["datasets"] == ["creative-graphic-design/rico25"]
    card.validate()
    assert "## Model Details" in text
    assert "### Model Description" in text
    assert "## Uses" in text
    assert "### Direct Use" in text
    assert "### Downstream Use" in text
    assert "### Out-of-Scope Use" in text
    assert "## Bias, Risks, and Limitations" in text
    assert "### Recommendations" in text
    assert "## How to Get Started with the Model" in text
    assert "LayoutDMPipeline.from_pretrained" in text
    assert "## Training Details" in text
    assert "### Training Data" in text
    assert "### Training Procedure" in text
    assert "## Evaluation" in text
    assert "### Testing Data, Factors & Metrics" in text
    assert "### Results" in text
    assert "Tokenizer exact" in text
    assert "## Technical Specifications" in text
    assert "## Model Card Contact" in text
    assert "[More Information Needed]" not in text
    assert "This card follows" not in text
    assert "annotated model card" not in text
    assert "model card template" not in text
    assert "## Citation" in text
    assert "https://github.com/CyberAgentAILab/layout-dm" in text


def test_layoutdm_model_card_mapping_inputs_and_errors():
    card = layoutdm_model_card(
        dataset="publaynet",
        parity_metrics=[
            {
                "dataset": "publaynet",
                "tokenizer_exact": "1/1",
                "deterministic_exact": "1/1",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
    )
    assert "| publaynet | 1/1 | 1/1 | 0 | 0 |" in str(card)
    with pytest.raises(ValueError, match="Unsupported LayoutDM dataset"):
        layoutdm_model_card(dataset="unknown")
