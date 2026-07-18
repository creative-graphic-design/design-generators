from typing import Literal, cast

import pytest
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


def test_processor_save_load_and_integer_labels(tmp_path) -> None:
    processor = LayouSynProcessor(max_in_len=2, id2label={0: "cat"})
    processor.save_pretrained(tmp_path)
    loaded = LayouSynProcessor.from_pretrained(tmp_path)
    encoded = loaded(
        labels=torch.tensor([0]),
        caption_embeds=torch.zeros(1, 120, 768),
        caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
        concept_embeds=torch.zeros(1, 1, 768),
    )
    assert encoded["label_texts"] == [["cat"]]
    assert loaded.id2label == {0: "cat"}


def test_processor_rejects_push_to_hub(tmp_path) -> None:
    processor = LayouSynProcessor()
    try:
        processor.save_pretrained(tmp_path, push_to_hub=True)
    except ValueError as exc:
        assert "does not push" in str(exc)
    else:
        raise AssertionError("push_to_hub should fail")


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


def test_postprocess_cxcywh_dict_and_intermediates() -> None:
    processor = LayouSynProcessor(max_in_len=1, layout_type="cxcywh")
    output = cast(
        dict[str, object],
        processor.postprocess(
            torch.zeros(1, 1, 4),
            labels=[["cat"]],
            id2label={0: "cat"},
            id2label_per_example=[{0: "cat"}],
            output_type="dict",
            return_intermediates=True,
            intermediates={"raw": True},
        ),
    )
    assert cast(torch.Tensor, output["bbox"]).shape == (1, 1, 4)
    intermediates = cast(dict[str, object], output["intermediates"])
    assert intermediates["vendor_layout_type"] == "cxcywh"


def test_postprocess_rejects_bad_output_type() -> None:
    processor = LayouSynProcessor(max_in_len=1)
    with pytest.raises(ValueError, match="Unsupported output_type"):
        processor.postprocess(
            torch.zeros(1, 1, 4),
            labels=[["cat"]],
            id2label={0: "cat"},
            output_type=cast(Literal["dataclass"], "bad"),
        )


def test_processor_input_validation_and_bbox_paths() -> None:
    processor = LayouSynProcessor(max_in_len=2, concept_in_channels=4, y_in_channels=4)
    encoded = processor(
        labels=["cat"],
        bbox=[[[0, 0, 10, 10]]],
        box_format="ltrb",
        normalized=False,
        canvas_size=(10, 10),
        aspect_ratio=[1.0],
        caption_embeds=torch.zeros(1, 120, 4),
        caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
        concept_embeds=torch.zeros(1, 1, 4),
        mask=[True],
    )
    assert encoded["aspect_ratio"].tolist() == [1.0]
    with pytest.raises(ValueError, match="requires labels"):
        processor()
    with pytest.raises(ValueError, match="batch sizes"):
        processor(
            prompt=["a", "b"],
            labels=[["cat"]],
            caption_embeds=torch.zeros(1, 120, 4),
            caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
            concept_embeds=torch.zeros(1, 1, 4),
        )
    with pytest.raises(ValueError, match="batch size"):
        processor(
            labels=["cat"],
            caption_embeds=torch.zeros(1, 120, 4),
            caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
            concept_embeds=torch.zeros(2, 1, 4),
        )
    with pytest.raises(ValueError, match="shape"):
        processor(
            labels=["cat"],
            caption_embeds=torch.zeros(1, 120, 4),
            caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
            concept_embeds=torch.zeros(1, 1),
        )
    with pytest.raises(ValueError, match="aspect_ratio"):
        processor(
            labels=["cat"],
            caption_embeds=torch.zeros(1, 120, 4),
            caption_padding_mask=torch.ones(1, 120, dtype=torch.bool),
            concept_embeds=torch.zeros(1, 1, 4),
            aspect_ratio=[1.0, 2.0],
        )


def test_processor_empty_prompt_uses_null_caption_embeddings() -> None:
    processor = LayouSynProcessor(max_in_len=1, max_y_len=3, y_in_channels=4)
    encoded = processor(labels=["cat"], concept_embeds=torch.zeros(1, 1, 768))
    assert encoded["caption_embeds"].shape == (1, 3, 4)
    try:
        processor(prompt="cat", labels=["cat"], concept_embeds=torch.zeros(1, 1, 768))
    except ValueError as exc:
        assert "caption_embeds are required" in str(exc)
    else:
        raise AssertionError("prompt without caption_embeds should fail")
