from dataclasses import dataclass

import matplotlib.pyplot as plt
import pytest
import torch
import yaml
from transformers.utils import ModelOutput

from laygen.common.bbox import (
    BoxFormat,
    denormalize_boxes,
    linear_continuize,
    linear_discretize,
    ltrb_to_xywh,
    normalize_boxes,
    xywh_to_ltrb,
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
from laygen.common.model_card import layout_corrector_model_card, layoutdm_model_card
from laygen.common.outputs_diffusers import (
    LayoutGenerationOutput as DiffusersLayoutGenerationOutput,
)
from laygen.common.outputs import LayoutGenerationOutput
from laygen.common.testing import (
    assert_generator_reproducible,
    assert_layout_output_schema,
    assert_normalized_xywh,
)
from laygen.common.visualization import render_layout


def test_bbox_conversions_roundtrip():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
    assert torch.allclose(ltrb_to_xywh(xywh_to_ltrb(bbox)), bbox)
    pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltrb")
    assert torch.allclose(
        normalize_boxes(pixels, canvas_size=(100, 200), box_format="ltrb"), bbox
    )
    pixels_from_enum = denormalize_boxes(
        bbox, canvas_size=(100, 200), box_format=BoxFormat.ltrb
    )
    assert torch.allclose(pixels_from_enum, pixels)
    ltwh_pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltwh")
    xywh_pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="xywh")
    assert torch.allclose(
        normalize_boxes(ltwh_pixels, canvas_size=(100, 200), box_format="ltwh"),
        bbox,
    )
    assert torch.allclose(
        normalize_boxes(xywh_pixels, canvas_size=(100, 200), box_format="xywh"),
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
    logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]])
    generator = torch.Generator().manual_seed(0)

    assert normalize_sampling_mode(SamplingMode.random) is SamplingMode.random
    with pytest.raises(ValueError, match="Unsupported sampling mode"):
        normalize_sampling_mode("bad")
    assert torch.equal(
        sample_categorical(logits, sampling=SamplingMode.deterministic),
        torch.tensor([[1, 0]]),
    )
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
            top_k=2,
            top_p=0.75,
            generator=generator,
        )
        assert sampled.shape == logits.shape[:-1]

    log_sampled = log_sample_categorical(logits.permute(0, 2, 1), generator=generator)
    assert log_sampled.shape == logits.shape[:-1]
    assert int(log_sampled.max()) < logits.shape[-1]
    assert torch.equal(top_k_logits(logits, 0), logits)
    masked = top_k_logits(logits, 1)
    assert masked[0, 0, 0] < -60.0
    assert log_add_exp(
        torch.tensor([0.0]), torch.tensor([0.0])
    ).item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    values = torch.tensor([10.0, 20.0, 30.0])
    assert extract(values, torch.tensor([0, 2]), torch.Size([2, 3, 4])).shape == (
        2,
        1,
        1,
    )
    assert gumbel_noise_like(torch.zeros(2, 3), generator=generator).shape == (2, 3)


def test_batch_topk_mask_edges():
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.7]])
    mask = batch_topk_mask(scores, torch.tensor([1, 2]))
    assert mask.tolist() == [[False, True, False], [True, False, True]]
    assert not batch_topk_mask(scores, torch.tensor([0, 0])).any()
    with pytest.raises(ValueError, match="rank-2"):
        batch_topk_mask(scores.unsqueeze(0), torch.tensor([1]))


def test_label_registry_aliases_and_errors():
    assert {item.value for item in DatasetName} == {
        "rico25",
        "rico13",
        "publaynet",
        "magazine",
    }
    assert normalize_dataset_name(DatasetName.rico25) is DatasetName.rico25
    assert normalize_dataset_name("rico25-max25") is DatasetName.rico25
    assert normalize_dataset_name("rico13") is DatasetName.rico13
    assert labels_for_dataset("rico13")[:3] == ("Text", "Image", "Icon")
    assert labels_for_dataset("publaynet") == (
        "text",
        "title",
        "list",
        "table",
        "figure",
    )
    assert label2id_for_dataset("rico25")["Text"] == 0
    assert id2label_for_dataset("rico13")[12] == "Advertisement"
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        normalize_dataset_name("crello")
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        normalize_dataset_name("unknown")


def test_condition_type_aliases_and_errors():
    assert {item.value for item in ConditionType} == {
        "unconditional",
        "label",
        "label_size",
        "completion",
        "refinement",
        "text",
        "content_image",
        "relation",
        "hierarchical",
        "retrieval",
    }
    assert normalize_condition_type(ConditionType.text) is ConditionType.text
    assert normalize_condition_type("ugen") is ConditionType.unconditional
    assert normalize_condition_type("gen_t") is ConditionType.label
    assert normalize_condition_type("gen_ts") is ConditionType.label_size
    assert normalize_condition_type("partial") is ConditionType.completion
    assert normalize_condition_type("complete") is ConditionType.completion
    assert normalize_condition_type("refine") is ConditionType.refinement
    assert normalize_condition_type("gen_r") is ConditionType.relation
    assert normalize_condition_type("text-to-layout") is ConditionType.text
    assert normalize_condition_type("content") is ConditionType.content_image
    assert normalize_condition_type("coarse-to-fine") is ConditionType.hierarchical
    assert normalize_condition_type("retrieval_examples") is ConditionType.retrieval
    with pytest.raises(ValueError, match="Unknown condition_type"):
        normalize_condition_type("unknown")


def test_output_schema():
    output = LayoutGenerationOutput(
        bbox=torch.zeros(1, 2, 4),
        labels=torch.zeros(1, 2, dtype=torch.long),
        mask=torch.tensor([[True, False]]),
        id2label=id2label_for_dataset(DatasetName.publaynet),
    )
    assert_layout_output_schema(output, batch_size=1)


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
        dataset=DatasetName.publaynet,
        parity_metrics=[
            {
                "dataset": DatasetName.publaynet,
                "tokenizer_exact": "1/1",
                "deterministic_exact": "1/1",
                "logits_max_abs": 0.0,
                "logits_max_rel": 0.0,
            }
        ],
    )
    text = str(card)
    assert "| publaynet | 1/1 | 1/1 | 0 | 0 |" in text
    assert "DatasetName" not in text
    metadata = yaml.safe_load(text.split("---", maxsplit=2)[1])
    assert metadata["datasets"] == ["creative-graphic-design/publaynet"]
    crello_text = str(layoutdm_model_card(dataset="crello-bbox"))
    crello_metadata = yaml.safe_load(crello_text.split("---", maxsplit=2)[1])
    assert crello_metadata["datasets"] == ["cyberagent/crello"]
    assert "DatasetName" not in crello_text
    with pytest.raises(ValueError, match="Unsupported layout dataset"):
        layoutdm_model_card(dataset="unknown")


def test_layout_corrector_model_card_metadata_and_sections():
    card = layout_corrector_model_card(dataset="crello-bbox")
    metadata = card.data.to_dict()
    text = str(card)

    assert metadata["license"] == "mit"
    assert metadata["library_name"] == "diffusers"
    assert metadata["datasets"] == ["cyberagent/crello"]
    assert metadata["language"] == ["en"]
    assert "layout-corrector" in metadata["tags"]
    assert "LayoutCorrectorPipeline.from_pretrained" in text
    assert "## Uses" in text
    assert "### Out-of-Scope Use" in text
    assert "## Bias, Risks, and Limitations" in text
    assert "## Evaluation" in text
    assert "### Results" in text
    assert "https://github.com/line/Layout-Corrector" in text
    assert "cyberagent/crello" in text
    assert "More Information Needed" not in text
    assert card.validate() is None


@dataclass
class _ToyOutput:
    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]


def test_testing_helpers_and_visualization():
    bbox = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [1.5, 0.5, 0.2, 0.2]]])
    labels = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    output = _ToyOutput(
        bbox=bbox.clamp(0, 1), labels=labels, mask=mask, id2label={0: "text"}
    )
    assert_layout_output_schema(output, batch_size=1)
    assert_normalized_xywh(
        torch.tensor([[[2.0, 2.0, 1.0, 1.0]]]), torch.tensor([[False]])
    )

    def make_output(*, generator: torch.Generator) -> _ToyOutput:
        value = torch.rand((), generator=generator)
        return _ToyOutput(
            bbox=torch.full((1, 1, 4), value.item()),
            labels=torch.tensor([[1]]),
            mask=torch.tensor([[True]]),
            id2label={1: "box"},
        )

    assert_generator_reproducible(make_output)
    ax = render_layout(
        bbox.squeeze(0),
        labels.squeeze(0),
        mask.squeeze(0),
        {0: "text"},
        canvas_size=(100, 200),
        colors=["red"],
    )
    assert len(ax.patches) == 1
    assert len(ax.texts) == 1
    plt.close("all")
