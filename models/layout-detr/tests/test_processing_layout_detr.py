import pytest
from PIL import Image
from typing import Literal, cast

from layout_detr import LayoutDetrConfig, LayoutDetrProcessor


def test_processor_maps_labels_and_pads_to_nine():
    processor = LayoutDetrProcessor(config=LayoutDetrConfig(max_text_length=8))
    batch = processor(
        images=Image.new("RGB", (32, 32), "white"),
        texts=["Sale", "Shop"],
        labels=["header", "button"],
    )

    assert tuple(batch["bbox_labels"].shape) == (1, 9)
    assert batch["bbox_labels"][0, :2].tolist() == [0, 5]
    assert batch["layout_mask"][0].tolist() == [True, True, *([False] * 7)]
    assert tuple(batch["input_ids"].shape) == (1, 9, 8)


def test_processor_rejects_invalid_label_and_prompt_only():
    processor = LayoutDetrProcessor(config=LayoutDetrConfig())

    with pytest.raises(ValueError, match="Unknown Ad Banner label"):
        processor(
            images=Image.new("RGB", (32, 32), "white"),
            texts=["Sale"],
            labels=["missing"],
        )
    with pytest.raises(ValueError, match="per-element texts"):
        processor(
            images=Image.new("RGB", (32, 32), "white"),
            prompt="Sale",
            labels=["header"],
        )


def test_processor_save_load_round_trip(tmp_path):
    processor = LayoutDetrProcessor(config=LayoutDetrConfig(background_size=16))
    processor.save_pretrained(tmp_path)

    loaded = LayoutDetrProcessor.from_pretrained(tmp_path)

    assert loaded.config.background_size == 16
    assert loaded.label2id["logo"] == 7


def test_processor_content_batch_masks_and_validation():
    processor = LayoutDetrProcessor(config=LayoutDetrConfig(max_text_length=4))
    image = Image.new("RGB", (32, 32), "white")

    batch = processor(
        content={
            "image": image,
            "texts": [["Sale"], ["Shop"]],
            "labels": [["header"], ["button"]],
        },
        batch_size=2,
        mask=[[True], [False]],
    )

    assert tuple(batch["pixel_values"].shape) == (2, 3, 256, 256)
    assert batch["layout_mask"][1].any().item() is False
    with pytest.raises(NotImplementedError):
        processor(
            images=image, texts=["Sale"], labels=["header"], condition_type="text"
        )
    with pytest.raises(ValueError):
        processor(images=[image, image], texts=["Sale"], labels=["header"])
    with pytest.raises(ValueError):
        processor(images=image, texts=["Sale"] * 10, labels=["header"] * 10)
    with pytest.raises(ValueError):
        processor(
            images=image,
            texts=["Sale"],
            labels=["header"],
            return_tensors=cast(Literal["pt"], "np"),
        )
