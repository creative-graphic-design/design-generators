import torch
import pytest
from transformers.utils import ModelOutput

from laygen.common.bbox import (
    BoxFormat,
    denormalize_boxes,
    linear_continuize,
    linear_discretize,
    ltrb_to_xywh,
    normalize_box_format,
    normalize_boxes,
    xywh_to_ltwh,
    xywh_to_ltrb,
)
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
from laygen.common.outputs_diffusers import (
    LayoutGenerationOutput as DiffusersLayoutGenerationOutput,
)
from laygen.common.outputs import LayoutGenerationOutput
from laygen.common.testing import (
    assert_generator_reproducible,
    assert_layout_output_schema,
    assert_mask_valid,
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
    pixels_from_enum = denormalize_boxes(
        bbox, canvas_size=(100, 200), box_format=BoxFormat.ltrb
    )
    assert torch.allclose(pixels_from_enum, pixels)
    assert torch.allclose(
        denormalize_boxes(bbox, canvas_size=(100, 200), box_format="xywh"),
        torch.tensor([[[50.0, 100.0, 20.0, 80.0]]]),
    )
    assert torch.allclose(
        denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltwh"),
        torch.tensor([[[40.0, 60.0, 20.0, 80.0]]]),
    )
    assert torch.allclose(
        normalize_boxes(
            torch.tensor([[[40.0, 60.0, 20.0, 80.0]]]),
            canvas_size=(100, 200),
            box_format="ltwh",
        ),
        bbox,
    )
    with pytest.raises(ValueError, match="Unsupported box_format"):
        normalize_box_format("bad")


def test_linear_bins_roundtrip_shape():
    values = torch.tensor([0.0, 0.25, 0.99])
    ids = linear_discretize(values, num_bins=4)
    assert ids.tolist() == [0, 1, 3]
    assert linear_continuize(ids, num_bins=4).shape == values.shape


def test_discrete_log_onehot_roundtrip():
    ids = torch.tensor([[0, 2, 1]])
    assert torch.equal(log_onehot_to_index(index_to_log_onehot(ids, 3)), ids)


def test_discrete_sampling_modes_and_masks():
    logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]])
    assert normalize_sampling_mode(SamplingMode.gumbel) is SamplingMode.gumbel
    with pytest.raises(ValueError, match="Unsupported sampling mode"):
        normalize_sampling_mode("bad")
    with pytest.raises(ValueError, match="exceeds vocab_size"):
        index_to_log_onehot(torch.tensor([[3]]), 3)
    assert log_add_exp(torch.tensor([0.0]), torch.tensor([0.0])).shape == (1,)
    assert extract(
        torch.arange(4.0), torch.tensor([1, 3]), torch.Size([2, 3])
    ).shape == (
        2,
        1,
    )
    generator = torch.Generator().manual_seed(0)
    assert gumbel_noise_like(logits, generator=generator).shape == logits.shape
    assert log_sample_categorical(logits.transpose(1, 2)).shape == (1, 2)
    assert torch.equal(
        sample_categorical(logits, sampling="deterministic"), torch.tensor([[1, 0]])
    )
    for mode in ["random", "gumbel", "top_k", "top_p", "top_k_top_p"]:
        sampled = sample_categorical(
            logits,
            sampling=mode,
            temperature=1.0,
            top_k=2,
            top_p=0.8,
            generator=torch.Generator().manual_seed(0),
        )
        assert sampled.shape == (1, 2)
    assert torch.equal(top_k_logits(logits, 0), logits)
    assert top_k_logits(logits, 2).shape == logits.shape
    assert not batch_topk_mask(torch.ones(2, 3), torch.zeros(2, dtype=torch.long)).any()
    assert batch_topk_mask(torch.ones(2, 3), torch.tensor([1, 2])).shape == (2, 3)
    with pytest.raises(ValueError, match="rank-2"):
        batch_topk_mask(torch.ones(2, 3, 1), torch.tensor([1, 2]))


def test_label_registry_aliases_and_errors():
    assert normalize_dataset_name("rico13-max25") is DatasetName.rico13
    assert len(labels_for_dataset("rico13")) == 13
    assert label2id_for_dataset("publaynet")["text"] == 0
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        id2label_for_dataset("unknown")


def test_output_schema():
    output = LayoutGenerationOutput(
        bbox=torch.zeros(1, 2, 4),
        labels=torch.zeros(1, 2, dtype=torch.long),
        mask=torch.tensor([[True, False]]),
        id2label=id2label_for_dataset(DatasetName.publaynet),
    )
    assert_layout_output_schema(output, batch_size=1)
    assert_mask_valid(output.mask)
    assert_normalized_xywh(output.bbox)
    with pytest.raises(AssertionError):
        assert_normalized_xywh(torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]))


def test_generator_reproducibility_helper_and_visualization():
    def make_output(generator: torch.Generator):
        bbox = torch.rand(1, 1, 4, generator=generator)
        labels = torch.zeros(1, 1, dtype=torch.long)
        mask = torch.ones(1, 1, dtype=torch.bool)
        return LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label={0: "text"},
        )

    assert_generator_reproducible(make_output)
    ax = render_layout(
        torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        torch.tensor([0, 9]),
        torch.tensor([True, False]),
        {0: "text"},
        canvas_size=(100, 100),
        colors=["red"],
    )
    assert len(ax.patches) == 1


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
    assert "layout-generation" in metadata["tags"]
    assert metadata["datasets"] == ["creative-graphic-design/rico25"]
    assert metadata["language"] == ["en"]
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
    assert "[More Information Needed]" not in text
    card.validate(repo_type="model")
