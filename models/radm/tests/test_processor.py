import pytest
import torch
from PIL import Image
import tempfile
from typing import Literal, cast

from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from radm import RADMConfig, RADMProcessor


def test_processor_encodes_content_and_text_features() -> None:
    config = RADMConfig(image_size=32, text_feature_dim=4)
    processor = RADMProcessor(config=config)
    encoded = processor(
        content={"image": Image.new("RGB", (16, 16))},
        text_features=torch.ones(1, 2, 4),
        text_mask=torch.tensor([[[True], [False]]]),
    )
    assert encoded["text_features"].shape == (1, 2, 4)
    assert encoded["text_mask"].tolist() == [[[True], [False]]]


def test_processor_rejects_unsupported_condition() -> None:
    processor = RADMProcessor(config=RADMConfig())
    with pytest.raises(NotImplementedError, match="unconditional"):
        processor.validate_condition("unconditional")


def test_processor_decode_schema_and_dict() -> None:
    processor = RADMProcessor(config=RADMConfig(num_proposals=2))
    boxes = torch.tensor([[[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.5, 0.5]]])
    logits = torch.ones(1, 2, 5)
    out = cast(
        LayoutGenerationOutput,
        processor.decode(
            boxes_xyxy=boxes,
            logits=logits,
            class_threshold=0.1,
            nms_threshold=0.5,
            return_intermediates=True,
        ),
    )
    assert_layout_output_schema(out, batch_size=1)
    as_dict = processor.decode(
        boxes_xyxy=boxes,
        logits=logits,
        class_threshold=0.1,
        nms_threshold=0.5,
        output_type="dict",
    )
    assert "bbox" in as_dict


def test_processor_save_load_errors_and_text_validation() -> None:
    config = RADMConfig(image_size=32, text_feature_dim=4)
    processor = RADMProcessor(config=config)
    with tempfile.TemporaryDirectory() as tmp:
        processor.save_pretrained(tmp)
        loaded = RADMProcessor.from_pretrained(tmp)
    assert loaded.id2label == processor.id2label

    with pytest.raises(ValueError, match="images"):
        processor()
    with pytest.raises(ValueError, match="last dimension"):
        processor(Image.new("RGB", (16, 16)), text_features=torch.zeros(1, 1, 3))
    encoded = processor(
        Image.new("RGB", (16, 16)),
        text_features=torch.zeros(1, 2, 4),
        text_mask=torch.tensor([[True, False]]),
    )
    assert encoded["text_mask"].shape == (1, 2, 1)
    with pytest.raises(ValueError, match="output_type"):
        processor.decode(
            boxes_xyxy=torch.zeros(1, 1, 4),
            logits=torch.zeros(1, 1, 5),
            class_threshold=2.0,
            nms_threshold=0.5,
            output_type=cast(Literal["dataclass", "dict"], "bad"),
        )
