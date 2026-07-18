import tempfile
from typing import cast

import pytest
import torch

from coarse_to_fine import (
    CoarseToFineConfig,
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)
from coarse_to_fine.conversion import strip_module_prefix
from laygen.modeling_outputs import LayoutGenerationOutput


def tiny_config() -> CoarseToFineConfig:
    return CoarseToFineConfig(
        dataset="publaynet",
        max_num_elements=2,
        discrete_x_grid=8,
        discrete_y_grid=8,
        d_model=16,
        d_z=16,
        n_layers=1,
        n_layers_decoder=1,
        n_heads=4,
        dim_feedforward=32,
        dropout=0.0,
    )


def test_pipeline_reproducible_with_generator():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )
    pipe = CoarseToFinePipeline(model=model, processor=processor)

    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    out_a = cast(LayoutGenerationOutput, pipe(batch_size=2, generator=gen_a))
    out_b = cast(LayoutGenerationOutput, pipe(batch_size=2, generator=gen_b))

    torch.testing.assert_close(out_a.bbox, out_b.bbox)
    torch.testing.assert_close(out_a.labels, out_b.labels)
    torch.testing.assert_close(out_a.mask, out_b.mask)


def test_generator_wins_over_seed():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )
    pipe = CoarseToFinePipeline(model=model, processor=processor)

    gen_a = torch.Generator().manual_seed(12)
    gen_b = torch.Generator().manual_seed(12)
    out_a = cast(LayoutGenerationOutput, pipe(batch_size=1, seed=0, generator=gen_a))
    out_b = cast(LayoutGenerationOutput, pipe(batch_size=1, seed=999, generator=gen_b))

    torch.testing.assert_close(out_a.bbox, out_b.bbox)


def test_unsupported_conditions_raise_explicitly():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )
    pipe = CoarseToFinePipeline(model=model, processor=processor)

    with pytest.raises(NotImplementedError):
        pipe(condition_type="label")


def test_model_does_not_publish_generate_layout():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()

    assert not hasattr(model, "generate_layout")


def test_forward_teacher_forcing_shapes():
    config = tiny_config()
    model = CoarseToFineForLayoutGeneration(config).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )
    labels = cast(torch.LongTensor, torch.tensor([[0, 1]]))
    bbox = cast(
        torch.FloatTensor,
        torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
    )
    mask = cast(torch.BoolTensor, torch.ones((1, 2), dtype=torch.bool))
    batch = processor.build_hierarchy_batch(labels, bbox, mask)

    out = model(
        labels=batch["labels"],
        bbox=batch["bbox"],
        mask=batch["mask"],
        group_bounding_box=batch["group_bounding_box"],
        label_in_one_group=batch["label_in_one_group"],
        group_mask=batch["group_mask"],
        grouped_bbox=batch["grouped_bbox"],
        grouped_labels=batch["grouped_labels"],
        grouped_mask=batch["grouped_mask"],
    )

    assert out["group_bounding_box_logits"].shape[-2:] == (
        4,
        config.bbox_vocab_size,
    )
    assert out["grouped_label_logits"].shape[-1] == config.element_label_size


def test_pipeline_and_dict_output():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )
    pipe = CoarseToFinePipeline(model=model, processor=processor)

    out = pipe(batch_size=1, seed=0, output_type="dict", return_intermediates=True)

    assert {"bbox", "labels", "mask", "id2label"}.issubset(out)
    assert pipe.postprocess(out) is out
    assert pipe.preprocess(None) == {}
    with pytest.raises(NotImplementedError):
        pipe._forward({})


def test_strip_module_prefix():
    state = {"module.foo": torch.tensor([1]), "bar": torch.tensor([2])}

    out = strip_module_prefix(state)

    assert set(out) == {"foo", "bar"}


def test_save_pretrained_from_pretrained_smoke():
    model = CoarseToFineForLayoutGeneration(tiny_config()).eval()
    processor = CoarseToFineProcessor(
        dataset="publaynet", x_grid=8, y_grid=8, max_num_elements=2
    )

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        loaded_model = CoarseToFineForLayoutGeneration.from_pretrained(tmp)
        loaded_processor = CoarseToFineProcessor.from_pretrained(tmp)
        pipe = CoarseToFinePipeline(model=loaded_model, processor=loaded_processor)
        out = cast(LayoutGenerationOutput, pipe(batch_size=1, seed=0))

    assert out.bbox.shape[0] == 1
    assert loaded_processor.id2label[0] == "text"
