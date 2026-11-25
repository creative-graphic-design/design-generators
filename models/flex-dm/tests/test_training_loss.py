"""CPU checks for Flex-DM's mask-aware training objective."""

import math

import pytest
import torch
from torch import nn

from flex_dm.configuration_flex_dm import FlexDmColumnSpec, FlexDmConfig
from flex_dm.modeling_flex_dm import FlexDmForMaskedDocumentModeling


def _column(
    kind: str,
    width: int,
    *,
    sequence: bool = True,
    condition: dict[str, object] | None = None,
) -> FlexDmColumnSpec:
    column: FlexDmColumnSpec = {
        "type": kind,  # type: ignore[typeddict-item]
        "input_dim": width if kind == "categorical" else None,
        "shape": (width,) if kind == "numerical" else (1,),
        "is_sequence": sequence,
        "primary_label": None,
    }
    if condition is not None:
        column["loss_condition"] = condition  # type: ignore[typeddict-item]
    return column


def _model(dataset_name: str = "crello") -> FlexDmForMaskedDocumentModeling:
    columns: dict[str, FlexDmColumnSpec] = {
        "length": _column("categorical", 3, sequence=False),
        "type": _column("categorical", 2),
        "color": _column(
            "categorical",
            3,
            condition={"key": "type", "mask": (True, False)},
        ),
        "embedding": _column(
            "numerical",
            2,
            condition={"key": "type", "mask": (True, False)},
        ),
    }
    return FlexDmForMaskedDocumentModeling(
        FlexDmConfig(
            dataset_name=dataset_name,
            input_columns=columns,
            attribute_groups={"type": ("type",), "pos": ("color",)},
            max_seq_length=3,
            latent_dim=8,
            num_blocks=0,
            l2=0.01,
        )
    )


def test_training_loss_masks_conditions_padding_and_adds_l2() -> None:
    model = _model()
    labels = {
        "length": torch.tensor([[0], [1]]),
        "type": torch.tensor([[[0], [1], [0]], [[1], [0], [0]]]),
        "color": torch.zeros((2, 3, 1), dtype=torch.long),
        "embedding": torch.zeros((2, 3, 2)),
    }
    masks = {
        key: torch.ones((2, 3), dtype=torch.bool) for key in labels if key != "length"
    }
    logits = {
        "type": torch.zeros((2, 3, 1, 2)),
        "color": torch.zeros((2, 3, 1, 3)),
        "embedding": torch.ones((2, 3, 2)),
    }

    components = model._compute_loss(logits, labels, masks, None)

    torch.testing.assert_close(components["type"], torch.tensor(1.5 * math.log(2)))
    torch.testing.assert_close(components["color"], torch.tensor(math.log(3)))
    torch.testing.assert_close(components["embedding"], torch.tensor(2.0))
    expected_l2 = 0.01 * sum(
        parameter.square().sum()
        for module in model.modules()
        if isinstance(module, nn.Linear | nn.Embedding)
        for parameter in module.parameters(recurse=False)
    )
    torch.testing.assert_close(model._compute_l2_loss(), expected_l2)

    output = model(
        inputs=labels,
        labels=labels,
        masks=masks,
        task_ids=torch.tensor([0, 0]),
    )
    assert output.loss_components is not None
    assert output.l2_loss is not None and output.loss is not None
    torch.testing.assert_close(
        output.loss,
        torch.stack((*output.loss_components.values(), output.l2_loss)).sum(),
    )


def test_rico_position_task_sorts_targets_and_predictions() -> None:
    columns = {
        "length": _column("categorical", 2, sequence=False),
        **{
            key: _column("categorical", 2)
            for key in ("type", "left", "top", "width", "height", "icon")
        },
    }
    model = FlexDmForMaskedDocumentModeling(
        FlexDmConfig(
            dataset_name="rico",
            input_columns=columns,
            attribute_groups={
                "type": ("type",),
                "pos": ("left", "top", "width", "height"),
                "attr": ("icon",),
            },
            latent_dim=8,
            num_blocks=0,
            l2=0,
        )
    )
    labels = {
        "length": torch.tensor([[1]]),
        "type": torch.tensor([[[1], [0]]]),
        **{
            key: torch.zeros((1, 2, 1), dtype=torch.long)
            for key in ("left", "top", "width", "height", "icon")
        },
    }
    logits = {
        key: torch.zeros((1, 2, 1, 2))
        for key in ("type", "left", "top", "width", "height", "icon")
    }
    logits["type"] = torch.tensor([[[[10.0, 0.0]], [[0.0, 10.0]]]])
    masks = {key: torch.zeros((1, 2), dtype=torch.bool) for key in logits}
    masks["type"][:] = True

    sorted_loss = model._compute_loss(logits, labels, masks, torch.tensor([3]))["type"]
    unsorted_loss = model._compute_loss(logits, labels, masks, None)["type"]

    assert sorted_loss < 1e-4
    assert unsorted_loss > 5.0


def test_categorical_loss_matches_keras_probability_clipping() -> None:
    model = _model()
    labels = {
        "length": torch.tensor([[0]]),
        "type": torch.tensor([[[0]]]),
    }
    logits = {"type": torch.tensor([[[[0.0, 100.0]]]])}
    masks = {"type": torch.ones((1, 1), dtype=torch.bool)}

    loss = model._compute_loss(logits, labels, masks, None)["type"]

    torch.testing.assert_close(loss, torch.tensor(-math.log(1e-7)))


def test_labels_require_training_masks() -> None:
    model = _model()
    inputs = {
        "length": torch.zeros((1, 1), dtype=torch.long),
        "type": torch.zeros((1, 1, 1), dtype=torch.long),
        "color": torch.zeros((1, 1, 1), dtype=torch.long),
        "embedding": torch.zeros((1, 1, 2)),
    }

    with pytest.raises(ValueError, match="masks are required"):
        model(inputs=inputs, labels=inputs)
