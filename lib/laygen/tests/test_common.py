import torch
import pytest
from transformers.utils import ModelOutput

from laygen.common.bbox import (
    BoxFormat,
    clamp_boxes,
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
from laygen.common.labels import DatasetName, id2label_for_dataset
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
    assert torch.allclose(xywh_to_ltwh(bbox), torch.tensor([[[0.4, 0.3, 0.2, 0.4]]]))
    assert torch.allclose(
        clamp_boxes(torch.tensor([[-1.0, 2.0]])), torch.tensor([[0.0, 1.0]])
    )
    pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltrb")
    assert torch.allclose(
        normalize_boxes(pixels, canvas_size=(100, 200), box_format="ltrb"), bbox
    )
    pixels_from_enum = denormalize_boxes(
        bbox, canvas_size=(100, 200), box_format=BoxFormat.ltrb
    )
    assert torch.allclose(pixels_from_enum, pixels)
    assert torch.allclose(
        denormalize_boxes(bbox, canvas_size=(100, 200), box_format=BoxFormat.ltwh),
        torch.tensor([[[40.0, 60.0, 20.0, 80.0]]]),
    )
    assert normalize_box_format(BoxFormat.xywh) is BoxFormat.xywh
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


def test_discrete_helpers_cover_sampling_modes():
    logits = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    assert normalize_sampling_mode(SamplingMode.random) is SamplingMode.random
    with pytest.raises(ValueError, match="Unsupported sampling mode"):
        normalize_sampling_mode("bad")
    with pytest.raises(ValueError, match="exceeds vocab_size"):
        index_to_log_onehot(torch.tensor([3]), 3)

    assert torch.allclose(
        log_add_exp(torch.tensor([0.0]), torch.tensor([0.0])),
        torch.log(torch.tensor([2.0])),
    )
    values = torch.tensor([0.1, 0.2, 0.3])
    timesteps = torch.tensor([2, 0])
    assert extract(values, timesteps, torch.Size([2, 3, 4])).shape == (2, 1, 1)

    generator_a = torch.Generator().manual_seed(0)
    generator_b = torch.Generator().manual_seed(0)
    assert torch.equal(
        gumbel_noise_like(logits, generator=generator_a),
        gumbel_noise_like(logits, generator=generator_b),
    )
    log_probs = logits.permute(0, 2, 1).log_softmax(dim=1)
    assert log_sample_categorical(
        log_probs, generator=torch.Generator().manual_seed(1)
    ).shape == (1, 2)
    assert torch.equal(
        sample_categorical(logits, sampling=SamplingMode.deterministic),
        torch.tensor([[2, 0]]),
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
            top_p=0.8,
            generator=torch.Generator().manual_seed(2),
        )
        assert sampled.shape == (1, 2)

    masked = top_k_logits(logits, k=2)
    assert masked[0, 0, 0] < -60
    assert torch.equal(top_k_logits(logits, k=0), logits)
    assert torch.equal(
        batch_topk_mask(torch.tensor([[0.1, 0.9, 0.5]]), torch.tensor([2])),
        torch.tensor([[False, True, True]]),
    )
    assert not batch_topk_mask(torch.ones(1, 3), torch.tensor([0])).any()
    with pytest.raises(ValueError, match="rank-2"):
        batch_topk_mask(torch.ones(1, 1, 3), torch.tensor([1]))


def test_output_schema():
    output = LayoutGenerationOutput(
        bbox=torch.zeros(1, 2, 4),
        labels=torch.zeros(1, 2, dtype=torch.long),
        mask=torch.tensor([[True, False]]),
        id2label=id2label_for_dataset(DatasetName.publaynet),
    )
    assert_layout_output_schema(output, batch_size=1)
    assert_normalized_xywh(torch.empty(0, 4))


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


def test_testing_helper_and_visualization():
    def callable_(generator: torch.Generator):
        ids = torch.multinomial(
            torch.tensor([[0.4, 0.6]]),
            1,
            generator=generator,
        )
        labels = ids.reshape(1, 1)
        return LayoutGenerationOutput(
            bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
            labels=labels,
            mask=torch.tensor([[True]]),
            id2label={0: "text", 1: "title"},
        )

    assert_generator_reproducible(callable_)
    ax = render_layout(
        torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]),
        torch.tensor([1, 99]),
        torch.tensor([True, False]),
        {1: "title"},
        canvas_size=(100, 200),
        colors=["red"],
    )
    assert len(ax.patches) == 1
    assert ax.texts[0].get_text() == "title"


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
    assert "### Testing Data" in text
    assert "### Factors" in text
    assert "### Metrics" in text
    assert "### Results" in text
    assert "Tokenizer exact" in text
    assert "## Technical Specifications" in text
    assert "[More Information Needed]" not in text
    assert "This card follows" not in text
    assert "annotated model card" not in text
    assert "model card template" not in text
    assert "## Citation" in text
    assert "https://github.com/CyberAgentAILab/layout-dm" in text
    assert "More Information Needed" not in text
    assert card.validate() is None


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
