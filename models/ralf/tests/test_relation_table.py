"""Tests for the standalone relation-table compatibility loader."""

from __future__ import annotations

import enum
import pickle
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pytest
import torch

from ralf.modeling_ralf import (
    RalfRelationElement,
    RalfRelationLocation,
    RalfRelationSize,
)
from ralf.training.lightning_module import _load_relationship_table


@contextmanager
def _fake_global_enums(
    definitions: dict[str, dict[str, int]],
) -> Iterator[dict[str, type[enum.IntEnum]]]:
    module_name = "image2layout.train.helpers.relationships"
    module_names = [
        "image2layout",
        "image2layout.train",
        "image2layout.train.helpers",
        module_name,
    ]
    previous = {name: sys.modules.get(name) for name in module_names}
    module = types.ModuleType(module_name)
    classes: dict[str, type[enum.IntEnum]] = {}
    for name, members in definitions.items():
        enum_type = enum.IntEnum("enum_type", members)
        enum_type.__name__ = name
        enum_type.__module__ = module_name
        enum_type.__qualname__ = name
        setattr(module, name, enum_type)
        classes[name] = enum_type
    for name in module_names[:-1]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules[module_name] = module
    try:
        yield classes
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _save_synthetic_table(
    path: Path,
    definitions: dict[str, dict[str, int]],
    payload_factory: Callable[[dict[str, type[enum.IntEnum]]], object],
) -> None:
    with _fake_global_enums(definitions) as classes:
        payload = payload_factory(classes)
        torch.save(payload, path)


def _enum_member(
    classes: dict[str, type[enum.IntEnum]], enum_name: str, member_name: str
) -> enum.IntEnum:
    return getattr(classes[enum_name], member_name)


def test_relation_table_maps_the_three_pinned_globals(tmp_path: Path) -> None:
    path = tmp_path / "relationships.pt"
    definitions = {
        "RelElement": {"A": 10},
        "RelLoc": {"LEFT": 5},
        "RelSize": {"LARGER": 3},
    }

    _save_synthetic_table(
        path,
        definitions,
        lambda classes: {
            "sample": [
                [
                    _enum_member(classes, "RelElement", "A"),
                    _enum_member(classes, "RelLoc", "LEFT"),
                    _enum_member(classes, "RelSize", "LARGER"),
                ]
            ]
        },
    )

    table = _load_relationship_table(path)

    assert table["sample"] == [
        [RalfRelationElement.A, RalfRelationLocation.LEFT, RalfRelationSize.LARGER]
    ]


def test_relation_table_refuses_unknown_globals(tmp_path: Path) -> None:
    path = tmp_path / "relationships.pt"
    _save_synthetic_table(
        path,
        {"Unexpected": {"VALUE": 1}},
        lambda classes: {"sample": [[_enum_member(classes, "Unexpected", "VALUE")]]},
    )

    with pytest.raises(pickle.UnpicklingError, match="unsupported global"):
        _load_relationship_table(path)
