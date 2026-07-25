from typing import cast

import pytest
import torch
from transformers.tokenization_utils_base import BatchEncoding

from laygen.pipelines import LayoutGenerationPipeline

from layoutvae import LayoutVAEConfig, LayoutVAEModel, LayoutVAEPipeline
from layoutvae.pipeline_layoutvae import (
    _load_model_component,
    _load_processor_component,
)


def _pipe():
    return LayoutVAEPipeline(LayoutVAEModel(LayoutVAEConfig()).eval())


def test_pipeline_subclasses_layout_base():
    assert isinstance(_pipe(), LayoutGenerationPipeline)


def test_pipeline_label_generation_and_dict_output():
    pipe = _pipe()
    out = pipe(
        labels=[["text", "figure"]],
        class_counts=torch.tensor([[7.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        output_type="dict",
        return_intermediates=True,
    )
    assert out["bbox"].shape == (1, 9, 4)
    assert out["labels"].shape == (1, 9)
    assert out["mask"].sum().item() == 2
    assert "raw_ltwh" in out["intermediates"]


def test_pipeline_preprocess_forward_and_postprocess():
    pipe = _pipe()
    assert pipe._sanitize_parameters(labels=["text"])[1]["labels"] == ["text"]
    encoded = pipe.preprocess(
        ["text"], class_counts=torch.tensor([[8.0, 1, 0, 0, 0, 0]])
    )
    assert isinstance(encoded, BatchEncoding)
    out = pipe._forward(dict(encoded), output_type="dict")
    assert out["bbox"].shape == (1, 9, 4)
    assert pipe.postprocess(out) is out
    with pytest.raises(ValueError, match="labels are required"):
        pipe.preprocess()
    with pytest.raises(ValueError, match="Unsupported generation kwargs"):
        pipe._forward({"label_set": torch.zeros(1, 6), "unknown": True})


def test_pipeline_component_loaders_and_device(tmp_path):
    pipe = LayoutVAEPipeline(LayoutVAEModel(LayoutVAEConfig()), device=-1)
    pipe.save_pretrained(tmp_path)
    assert _load_model_component(tmp_path) is not None
    assert _load_processor_component(tmp_path) is not None
    model_dir = tmp_path / "model"
    processor_dir = tmp_path / "processor"
    pipe.model.save_pretrained(model_dir)
    pipe.processor.save_pretrained(processor_dir)
    assert _load_model_component(tmp_path, subfolder="model") is not None
    assert _load_processor_component(tmp_path, subfolder="processor") is not None


def test_pipeline_rejects_unsupported_condition():
    with pytest.raises(ValueError, match="Unsupported condition_type"):
        _pipe()(labels=["text"], condition_type="unconditional")


def test_pipeline_default_labels_and_bad_batch_size():
    out = _pipe()(batch_size=2, class_counts=torch.tensor([[8.0, 1, 0, 0, 0, 0]] * 2))
    assert out["bbox"].shape == (2, 9, 4)
    with pytest.raises(ValueError, match="batch_size"):
        _pipe()(labels=None, batch_size=0)


def test_pipeline_save_load(tmp_path):
    pipe = _pipe()
    pipe.save_pretrained(tmp_path)
    loaded = LayoutVAEPipeline.from_pretrained(tmp_path)
    out = loaded(
        labels=["text"],
        class_counts=torch.tensor([[8.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert cast(torch.Tensor, out["mask"]).sum().item() == 1
