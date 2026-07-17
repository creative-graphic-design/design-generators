import torch
import pytest

from laygen.common.bbox import BoxFormat
from layout_flow import ConditionType, LayoutFlowConfig, LayoutFlowProcessor
from layout_flow.configuration_layout_flow import InitialDistributionName
from layout_flow.processing_layout_flow import normalize_condition_type


def test_processor_converts_ltwh_to_xywh_and_pads() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=3)
    )
    out = processor(
        bbox=[[[0.1, 0.2, 0.4, 0.6]]],
        labels=[[2]],
        mask=[[True]],
        box_format="ltwh",
    )
    assert out["bbox"].shape == (1, 3, 4)
    assert torch.allclose(out["bbox"][0, 0], torch.tensor([0.3, 0.5, 0.4, 0.6]))
    assert out["mask"].tolist() == [[True, False, False]]


def test_condition_masks_match_layout_flow_semantics() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=2)
    )
    mask = torch.tensor([[True, True]])
    assert processor.make_condition_mask("unconditional", mask=mask).sum().item() == 14
    label = processor.make_condition_mask("c", mask=mask)
    assert label[:, :, :4].all()
    assert not label[:, :, 4:].any()
    size = processor.make_condition_mask("cwh", mask=mask)
    assert size[:, :, :2].all()
    assert not size[:, :, 2:].any()


def test_processor_handles_aliases_errors_and_denormalized_boxes() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=3)
    )

    assert normalize_condition_type(ConditionType.label) is ConditionType.label
    with pytest.raises(ValueError, match="Unknown condition_type"):
        normalize_condition_type("bad")

    out = processor(
        bbox=torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
        labels=torch.tensor([1]),
        num_elements=1,
        box_format=BoxFormat.ltrb,
        normalized=False,
        canvas_size=(100, 100),
    )
    assert torch.allclose(out["bbox"][0, 0], torch.tensor([0.25, 0.25, 0.5, 0.5]))
    assert out["labels"].tolist() == [[1, 0, 0]]

    with pytest.raises(ValueError, match="canvas_size"):
        processor(
            bbox=[[[0.0, 0.0, 50.0, 50.0]]],
            box_format="ltrb",
            normalized=False,
        )


def test_processor_completion_and_postprocess_branches() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(dataset_name="publaynet", max_length=4)
    )
    mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    cond = processor.make_condition_mask(
        "completion", mask=mask, generator=torch.Generator().manual_seed(0)
    )
    assert cond[0].all()
    assert (cond[1] == 0).any()

    state = processor.preprocess_state(
        processor.model_state(
            torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
            torch.tensor([[2]]),
        )
    )
    post = processor.postprocess(
        state,
        mask=torch.tensor([[True]]),
        box_format="ltwh",
        normalized=True,
    )
    assert post["bbox"].shape == (1, 1, 4)
    denorm = processor.postprocess(
        state,
        mask=torch.tensor([[True]]),
        normalized=False,
        canvas_size=(100, 200),
    )
    assert torch.allclose(denorm["bbox"][0, 0], torch.tensor([50.0, 100.0, 25.0, 50.0]))
    with pytest.raises(ValueError, match="canvas_size"):
        processor.postprocess(state, mask=torch.tensor([[True]]), normalized=False)


def test_processor_default_inputs_truncate_and_non_scaled_distribution() -> None:
    processor = LayoutFlowProcessor(
        LayoutFlowConfig(
            dataset_name="publaynet",
            max_length=2,
            distribution=InitialDistributionName.gmm,
        )
    )
    out = processor(batch_size=2, num_elements=[1, 3])
    assert out["mask"].tolist() == [[True, False], [True, True]]
    long = processor(
        bbox=torch.zeros(1, 3, 4),
        labels=torch.tensor([[1, 2, 3]]),
        mask=torch.tensor([[True, True, True]]),
    )
    assert long["bbox"].shape == (1, 2, 4)
    state = torch.ones(1, 1, processor.config.sample_dim)
    assert torch.equal(processor.preprocess_state(state), state)
