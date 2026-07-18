import torch

from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from layousyn.processing_layousyn import LayouSynProcessor


def test_string_labels_use_batch_local_union_id2label() -> None:
    processor = LayouSynProcessor(max_in_len=3, concept_in_channels=2, y_in_channels=2)
    encoded = processor(
        prompt=["one", "two"],
        labels=[["cat", "tree"], ["tree", "cat"]],
        caption_embeds=torch.zeros(2, 120, 2),
        caption_padding_mask=torch.ones(2, 120, dtype=torch.bool),
        concept_embeds=torch.zeros(2, 2, 2),
    )
    assert encoded["id2label"] == {0: "cat", 1: "tree"}
    assert encoded["id2label_per_example"] == [
        {0: "cat", 1: "tree"},
        {0: "tree", 1: "cat"},
    ]
    assert encoded["concept_padding_mask"].tolist() == [
        [False, False, True],
        [False, False, True],
    ]


def test_integer_labels_require_id2label() -> None:
    processor = LayouSynProcessor(max_in_len=2)
    try:
        processor(labels=torch.tensor([[0]]))
    except ValueError as exc:
        assert "id2label" in str(exc)
    else:
        raise AssertionError("integer labels without id2label should fail")


def test_postprocess_xyxy_to_public_xywh() -> None:
    processor = LayouSynProcessor(max_in_len=2, layout_type="xyxy")
    output = processor.postprocess(
        torch.tensor([[[-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]),
        labels=[["cat"]],
        id2label={0: "cat"},
    )
    assert isinstance(output, LayoutGenerationOutput)
    assert torch.allclose(output.bbox[0, 0], torch.tensor([0.5, 0.5, 1.0, 1.0]))
    assert output.mask.tolist() == [[True, False]]
