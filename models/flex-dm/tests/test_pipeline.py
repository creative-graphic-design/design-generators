"""Flex-DM pipeline tests."""

import torch

from laygen.common.testing import assert_layout_output_schema

from flex_dm.testing import tiny_pipeline


def test_pipeline_completion_schema_and_dict_output() -> None:
    """Completion returns the common schema and supports dict output."""
    pipe = tiny_pipeline()
    output = pipe(
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        labels=torch.tensor([[1]]),
        mask=torch.tensor([[True]]),
        feature_group="pos",
        seed=0,
        return_intermediates=True,
    )

    assert_layout_output_schema(output, batch_size=1)
    as_dict = pipe(
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        labels=torch.tensor([[1]]),
        mask=torch.tensor([[True]]),
        feature_group="pos",
        output_type="dict",
    )
    assert set(as_dict) >= {"bbox", "labels", "mask", "id2label"}


def test_pipeline_rejects_unsupported_conditions() -> None:
    """Unsupported canonical modes raise explicit errors."""
    pipe = tiny_pipeline()
    try:
        pipe(condition_type="label")
    except NotImplementedError as exc:
        assert 'feature_group="type"' in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")

    for condition in ("unconditional", "label_size", "text"):
        try:
            pipe(condition_type=condition)
        except NotImplementedError as exc:
            assert condition.split("_")[0] in str(exc) or "input document" in str(exc)
        else:
            raise AssertionError("expected NotImplementedError")


def test_pipeline_refinement_content_image_and_iterative() -> None:
    """Pipeline covers refinement, content_image, and iterative decoding."""
    pipe = tiny_pipeline()
    kwargs = {
        "bbox": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        "labels": torch.tensor([[1]]),
        "mask": torch.tensor([[True]]),
        "return_intermediates": True,
    }
    refinement = pipe(condition_type="refinement", feature_group="pos", **kwargs)
    content = pipe(condition_type="content_image", feature_group="img", **kwargs)
    iterative = pipe(feature_group="type", num_inference_steps=2, **kwargs)

    assert "refinement_input" in refinement.intermediates
    assert content.bbox.shape[-1] == 4
    assert iterative.labels.shape == (1, 1)


def test_pipeline_generator_takes_precedence_over_seed() -> None:
    """Changing seed does not matter when the generator state is fixed."""
    pipe = tiny_pipeline()
    kwargs = {
        "bbox": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        "labels": torch.tensor([[1]]),
        "mask": torch.tensor([[True]]),
        "feature_group": "pos",
    }

    first = pipe(generator=torch.Generator().manual_seed(123), seed=1, **kwargs)
    second = pipe(generator=torch.Generator().manual_seed(123), seed=999, **kwargs)

    assert torch.equal(first.bbox, second.bbox)
    assert torch.equal(first.labels, second.labels)


def test_pipeline_save_pretrained_round_trip(tmp_path) -> None:
    """Pipeline root save/load uses standard component specs."""
    pipe = tiny_pipeline()
    pipe.save_pretrained(tmp_path)

    loaded = type(pipe).from_pretrained(tmp_path, local_files_only=True)
    output = loaded(
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
        labels=torch.tensor([[1]]),
        mask=torch.tensor([[True]]),
        feature_group="pos",
    )

    assert_layout_output_schema(output, batch_size=1)
