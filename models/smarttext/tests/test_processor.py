import torch
from PIL import Image, ImageFont
from typing import cast

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from smarttext import SmartTextConfig, SmartTextProcessor
from smarttext.candidate_generation import (
    SmartTextCandidate,
    SmartTextLine,
    candidate_to_reference_json,
)


def _candidate(index=0):
    return SmartTextCandidate(
        index=index,
        bbox_ltrb_px=(10, 10, 30, 30),
        lines=(SmartTextLine("A", 10, (10, 10, 20, 20)),),
    )


def test_processor_requires_image_and_prompt():
    processor = SmartTextProcessor(config=SmartTextConfig())

    encoded = processor(
        Image.new("RGB", (32, 32), "white"),
        prompt="A",
        font=ImageFont.load_default(),
    )

    assert encoded["prompts"] == ["A"]
    assert encoded["basnet_pixel_values"].shape[1] == 3


def test_processor_save_load_round_trip(tmp_path):
    processor = SmartTextProcessor(config=SmartTextConfig(candi_res=2))

    processor.save_pretrained(tmp_path)
    loaded = SmartTextProcessor.from_pretrained(tmp_path)

    assert loaded.config.candi_res == 2
    assert loaded.id2label == {0: "text"}


def test_processor_save_load_round_trip_from_subfolder(tmp_path):
    processor = SmartTextProcessor(config=SmartTextConfig(candi_res=2))
    processor.save_pretrained(tmp_path / "processor")

    loaded = SmartTextProcessor.from_pretrained(tmp_path, subfolder="processor")

    assert loaded.config.candi_res == 2


def test_processor_rejects_missing_payloads_and_bad_batch():
    processor = SmartTextProcessor(config=SmartTextConfig())

    for kwargs in ({}, {"images": Image.new("RGB", (32, 32), "white")}):
        try:
            processor(**kwargs)  # ty: ignore[invalid-argument-type]
        except ValueError as exc:
            assert "requires" in str(exc)
        else:
            raise AssertionError("missing SmartText payload should fail")

    try:
        processor(
            [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "white")],
            prompt=["A"],
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("prompt batch mismatch should fail")

    try:
        processor(Image.new("RGB", (32, 32), "white"), prompt="A", return_tensors="np")  # ty: ignore[invalid-argument-type]
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("return_tensors='np' should fail")


def test_processor_accepts_content_tensor_and_sequence_payloads():
    processor = SmartTextProcessor(config=SmartTextConfig())
    image_tensor = torch.zeros(3, 16, 16)

    encoded = processor(
        content={"image": image_tensor, "text": "A", "saliency": torch.zeros(16, 16)}
    )

    assert encoded["prompts"] == ["A"]
    assert encoded["saliency"].shape == torch.Size([16, 16])

    encoded_batch = processor(
        [Image.new("RGB", (16, 16), "white"), Image.new("RGB", (16, 16), "black")],
        prompt=("A", "B"),
    )

    assert encoded_batch["prompts"] == ["A", "B"]


def test_processor_decodes_candidate_payloads_and_rejects_prompt_type():
    processor = SmartTextProcessor(config=SmartTextConfig())
    candidate = candidate_to_reference_json(_candidate())

    flat = processor(
        Image.new("RGB", (32, 32), "white"),
        prompt="A",
        candidate_boxes=candidate,
    )
    nested = processor(
        Image.new("RGB", (32, 32), "white"),
        prompt="A",
        candidate_boxes=[candidate],
    )
    empty = processor(
        Image.new("RGB", (32, 32), "white"),
        prompt="A",
        candidate_boxes=[],
    )

    assert flat["candidate_boxes"][0].bbox_ltrb_px == _candidate().bbox_ltrb_px
    assert nested["candidate_boxes"][0].bbox_ltrb_px == _candidate().bbox_ltrb_px
    assert empty["candidate_boxes"] == []

    try:
        processor(Image.new("RGB", (32, 32), "white"), prompt=object())  # ty: ignore[invalid-argument-type]
    except TypeError as exc:
        assert "prompt/text" in str(exc)
    else:
        raise AssertionError("bad prompt type should fail")


def test_processor_decode_returns_normalized_schema():
    processor = SmartTextProcessor(config=SmartTextConfig(candi_res=1))

    output = cast(
        LayoutGenerationOutput,
        processor.decode(
            candidates=[_candidate()],
            scores=torch.tensor([1.0]),
            image_size=(100, 100),
        ),
    )

    assert_layout_output_schema(output)
    assert output.bbox.max() <= 1.0
    assert output.id2label == {0: "text"}


def test_processor_decode_dict_and_text_lines():
    processor = SmartTextProcessor(config=SmartTextConfig())

    output = processor.decode(
        candidates=[_candidate()],
        scores=torch.tensor([1.0]),
        image_size=(100, 100),
        output_type="dict",
        return_text_lines=True,
        score_normalization="raw",
    )

    assert cast(torch.Tensor, output["bbox"]).shape == torch.Size([1, 1, 4])
    assert cast(torch.Tensor, output["scores"]).item() == 1.0


def test_processor_decode_rejects_invalid_options():
    processor = SmartTextProcessor(config=SmartTextConfig())

    try:
        processor.decode(candidates=[], scores=torch.tensor([]), image_size=(100, 100))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty candidates should fail")

    try:
        processor.decode(
            candidates=[_candidate()],
            scores=torch.tensor([1.0]),
            image_size=(100, 100),
            score_normalization="bad",  # ty: ignore[invalid-argument-type]
        )
    except ValueError as exc:
        assert "score_normalization" in str(exc)
    else:
        raise AssertionError("bad score mode should fail")

    try:
        processor.decode(
            candidates=[_candidate()],
            scores=torch.tensor([1.0]),
            image_size=(100, 100),
            output_type="tuple",  # ty: ignore[invalid-argument-type]
        )
    except ValueError as exc:
        assert "output_type" in str(exc)
    else:
        raise AssertionError("bad output type should fail")
