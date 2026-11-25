"""CPU regressions for the Flex-DM Lightning input/target boundary."""

import pytest
import torch

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


def _training_module(monkeypatch, *, dataset: str, columns, groups, task_id: int):
    pytest.importorskip("lightning")
    pytest.importorskip("traingen")
    from flex_dm.training import lightning_module as training_module

    config = FlexDmConfig(
        dataset_name=dataset,
        input_columns=columns,
        attribute_groups=groups,
        max_seq_length=2,
        latent_dim=8,
        num_blocks=0,
        masking_method="type" if dataset == "crello" else "pos",
        l2=0,
    )
    model = FlexDmForMaskedDocumentModeling(config)

    def load_model(cls, model_name_or_path: str):
        assert model_name_or_path == "tiny"
        return model

    monkeypatch.setattr(
        training_module.FlexDmForMaskedDocumentModeling,
        "from_pretrained",
        classmethod(load_model),
    )
    monkeypatch.setattr(
        training_module,
        "sample_training_tasks",
        lambda *args, **kwargs: torch.full((1,), task_id),
    )
    return training_module.FlexDmTrainingModule("tiny")


def test_crello_condition_uses_original_type_labels_not_mask_tokens(
    monkeypatch,
) -> None:
    """Masked and unused type tokens are model inputs, never loss conditions."""
    columns = {
        "length": _column("categorical", 3, sequence=False),
        "type": _column("categorical", 6),
        "color": _column(
            "numerical",
            2,
            condition={"key": "type", "mask": (True, True, True, True, True, True)},
        ),
    }
    module = _training_module(
        monkeypatch,
        dataset="crello",
        columns=columns,
        groups={"type": ("type",)},
        task_id=2,
    )
    batch = {
        "length": torch.tensor([[0]]),
        "type": torch.tensor([[[1], [2]]]),
        "color": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    }

    inputs, labels, masks, task_ids = module.prepare_batch(batch)
    condition_loss = module.model._compute_loss(
        {"color": torch.zeros((1, 2, 2))},
        labels,
        {"color": masks["color"]},
        task_ids,
    )
    output = module.model(inputs=inputs, labels=labels, masks=masks, task_ids=task_ids)

    assert condition_loss["color"].isfinite()
    assert output.loss is not None
    assert torch.equal(labels["type"], batch["type"])
    assert inputs["type"].squeeze(-1).tolist() == [[6, 7]]
    assert "task" not in labels
    assert "task" in inputs


def test_rico_sort_keeps_auxiliary_task_out_of_targets(monkeypatch) -> None:
    """The auxiliary task label belongs to model inputs, not RICO targets."""
    columns = {
        "length": _column("categorical", 3, sequence=False),
        **{
            key: _column("categorical", 16 if key == "type" else 64)
            for key in ("type", "left", "top", "width", "height", "icon")
        },
    }
    module = _training_module(
        monkeypatch,
        dataset="rico",
        columns=columns,
        groups={
            "type": ("type",),
            "pos": ("left", "top", "width", "height"),
            "attr": ("icon",),
        },
        task_id=3,
    )
    batch = {
        "length": torch.tensor([[1]]),
        **{
            key: torch.tensor([[[1], [2]]])
            for key in ("type", "left", "top", "width", "height", "icon")
        },
    }

    inputs, labels, masks, task_ids = module.prepare_batch(batch)
    output = module.model(inputs=inputs, labels=labels, masks=masks, task_ids=task_ids)

    assert output.loss is not None
    assert "task" not in labels
    assert "task" in inputs
