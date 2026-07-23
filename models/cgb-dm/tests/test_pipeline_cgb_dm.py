from typing import cast

import pytest
import torch

from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from cgb_dm import CGBDMPipeline, CGBDMProcessor, CGBDMScheduler, CGBDMTransformerModel
from cgb_dm.pipeline_cgb_dm import normalize_condition_type


def tiny_pipe() -> CGBDMPipeline:
    model = CGBDMTransformerModel(
        num_labels=4,
        max_seq_length=2,
        image_size=(32, 32),
        dim_model=16,
        n_head=2,
        feature_dim=32,
        num_layers=1,
        num_train_timesteps=10,
    )
    return CGBDMPipeline(
        model=model,
        scheduler=CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=1),
        processor=CGBDMProcessor(max_seq_length=2, image_size=(32, 32)),
    )


def test_condition_aliases_and_unconditional_rejection():
    assert str(normalize_condition_type("uncond")) == "content_image"
    with pytest.raises(ValueError, match="requires image/content"):
        normalize_condition_type("unconditional")


def test_pipeline_content_image_seed_reproducible():
    pipe = tiny_pipe()
    pixel_values = torch.zeros(1, 4, 32, 32)

    first = cast(
        LayoutGenerationOutput,
        pipe(pixel_values=pixel_values, seed=1, num_inference_steps=1),
    )
    second = cast(
        LayoutGenerationOutput,
        pipe(pixel_values=pixel_values, seed=1, num_inference_steps=1),
    )

    assert first.bbox.shape == (1, 2, 4)
    assert torch.allclose(first.bbox, second.bbox)


def test_pipeline_conditioning_dict_output_and_save_pretrained(tmp_path):
    pipe = tiny_pipe()
    output = pipe(
        pixel_values=torch.zeros(1, 4, 32, 32),
        condition_type="label_size",
        bbox=[[[0.5, 0.5, 0.2, 0.2]]],
        labels=[[1]],
        seed=2,
        num_inference_steps=1,
        output_type="dict",
        return_intermediates=True,
    )
    output = cast(dict[str, object], output)
    bbox = cast(torch.Tensor, output["bbox"])
    trajectory = cast(list[torch.Tensor], output["trajectory"])
    assert bbox.shape == (1, 2, 4)
    assert trajectory[0].shape == (1, 2, 8)

    pipe.save_pretrained(tmp_path)
    loaded = CGBDMPipeline.from_pretrained(tmp_path, local_files_only=True)
    restored = cast(
        LayoutGenerationOutput,
        loaded(pixel_values=torch.zeros(1, 4, 32, 32), num_inference_steps=1, seed=3),
    )
    assert restored.bbox.shape == (1, 2, 4)


def test_pipeline_requires_conditioning_tensors():
    pipe = tiny_pipe()
    with pytest.raises(ValueError, match="bbox and labels are required"):
        pipe(pixel_values=torch.zeros(1, 4, 32, 32), condition_type="label")


def test_pipeline_content_container_and_error_paths():
    pipe = tiny_pipe()
    output = pipe(
        content={"image": torch.zeros(3, 32, 32), "saliency": torch.ones(32, 32)},
        num_inference_steps=1,
        seed=4,
    )
    assert isinstance(output, LayoutGenerationOutput)

    with pytest.raises(ValueError, match="Unsupported output_type"):
        pipe(pixel_values=torch.zeros(1, 4, 32, 32), output_type="bad")
    with pytest.raises(ValueError, match="Unsupported CGB-DM condition_type"):
        pipe(pixel_values=torch.zeros(1, 4, 32, 32), condition_type="text")
