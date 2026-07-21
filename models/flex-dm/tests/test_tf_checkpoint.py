"""Flex-DM TensorFlow checkpoint helper tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flex_dm import tf_checkpoint


class _FakeTrain:
    def list_variables(
        self, checkpoint_prefix: str
    ) -> list[tuple[str, tuple[int, ...]]]:
        assert checkpoint_prefix == "best.ckpt"
        return [("model/a", (2, 3)), ("model/b", (1,))]

    def load_variable(self, checkpoint_prefix: str, name: str) -> object:
        assert checkpoint_prefix == "best.ckpt"
        values = {
            "model/a": np.ones((2, 3), dtype=np.float32),
            "model/b": [1.0],
        }
        return values[name]


def test_list_and_load_tf_checkpoint_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint helpers delegate to TensorFlow train APIs."""
    fake_tf = SimpleNamespace(train=_FakeTrain(), __version__="2.15.1")

    monkeypatch.setattr(tf_checkpoint, "import_module", lambda name: fake_tf)

    assert tf_checkpoint.list_tf_checkpoint_variables("best.ckpt") == [
        ("model/a", (2, 3)),
        ("model/b", (1,)),
    ]
    variables = tf_checkpoint.load_tf_checkpoint_variables("best.ckpt")
    assert set(variables) == {"model/a", "model/b"}
    assert variables["model/a"].shape == (2, 3)
    assert variables["model/b"].tolist() == [1.0]
    assert tf_checkpoint.tensorflow_version() == "2.15.1"


def test_tensorflow_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing TensorFlow produces the optional-dependency error."""

    def raise_import_error(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(tf_checkpoint, "import_module", raise_import_error)

    with pytest.raises(ImportError, match="TensorFlow is required"):
        tf_checkpoint.list_tf_checkpoint_variables("best.ckpt")

    assert tf_checkpoint.tensorflow_version() == "not-installed"
