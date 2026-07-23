import pytest
import torch
from PIL import Image

from laygen.common.testing import assert_layout_output_schema
from layout_detr import (
    LayoutDetrConfig,
    LayoutDetrForConditionalGeneration,
    LayoutDetrPipeline,
    LayoutDetrProcessor,
)


def _pipeline():
    config = LayoutDetrConfig(
        background_size=16,
        hidden_dim=32,
        bert_f_dim=32,
        max_text_length=8,
        text_vocab_size=128,
    )
    return LayoutDetrPipeline(
        model=LayoutDetrForConditionalGeneration(config),
        processor=LayoutDetrProcessor(config=config),
        config=config,
    )


def test_pipeline_content_image_smoke_and_dict_output():
    pipe = _pipeline()

    output = pipe(
        Image.new("RGB", (32, 32), "white"),
        texts=["Sale", "Shop"],
        labels=["header", "button"],
        seed=0,
        return_intermediates=True,
    )
    assert_layout_output_schema(output)
    assert "latents" in output.intermediates

    as_dict = pipe(
        Image.new("RGB", (32, 32), "white"),
        texts=["Sale"],
        labels=["header"],
        output_type="dict",
    )
    assert {"bbox", "labels", "mask", "id2label"}.issubset(as_dict)


def test_generator_wins_over_seed_and_save_load(tmp_path):
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32), "white")
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    out1 = pipe(image, texts=["Sale"], labels=["header"], seed=0, generator=g1)
    out2 = pipe(image, texts=["Sale"], labels=["header"], seed=999, generator=g2)

    assert torch.allclose(out1.bbox, out2.bbox)

    pipe.save_pretrained(tmp_path)
    loaded = LayoutDetrPipeline.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, LayoutDetrPipeline)


def test_pipeline_rejects_unsupported_modes_and_options():
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32), "white")

    with pytest.raises(NotImplementedError):
        pipe(image, texts=["Sale"], labels=["header"], condition_type="text")
    with pytest.raises(ValueError, match="existing bbox"):
        pipe(image, texts=["Sale"], labels=["header"], bbox=torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="single forward"):
        pipe(image, texts=["Sale"], labels=["header"], num_inference_steps=2)
    with pytest.raises(ValueError, match="num_elements"):
        pipe(image, texts=["Sale"], labels=["header"], num_elements=1)
    with pytest.raises(ValueError, match="xywh"):
        pipe(image, texts=["Sale"], labels=["header"], box_format="ltrb")
    with pytest.raises(ValueError, match="normalized"):
        pipe(image, texts=["Sale"], labels=["header"], normalized=False)
