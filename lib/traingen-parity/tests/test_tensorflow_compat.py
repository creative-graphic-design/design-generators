import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
from traingen_parity.tensorflow_compat import (
    install_assert_all_finite_compat,
    install_keras_preprocessing_compat,
    normalize_lookup_kwargs,
)


class _FakeLookup:
    def __init__(self, *args: str, **kwargs: str | None) -> None:
        self.args = args
        self.kwargs = kwargs

    def vocabulary_size(self) -> int:
        return 3


class _FakeDiscretization:
    pass


def test_tensorflow_compat_imports_neither_torch_nor_tensorflow() -> None:
    code = textwrap.dedent(
        """
        import sys
        import traingen_parity.tensorflow_compat

        assert "torch" not in sys.modules
        assert "tensorflow" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_normalize_lookup_kwargs_renames_mask_value() -> None:
    kwargs = {"mask_value": None, "num_oov_indices": 1}

    assert normalize_lookup_kwargs(kwargs) == {"num_oov_indices": 1, "mask_token": None}
    assert normalize_lookup_kwargs({"vocabulary": ["a"]}) == {"vocabulary": ["a"]}
    assert "mask_value" in kwargs


def test_install_keras_preprocessing_compat_registers_lookup_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = SimpleNamespace(
        StringLookup=_FakeLookup,
        IntegerLookup=_FakeLookup,
        Discretization=_FakeDiscretization,
    )
    tensorflow = ModuleType("tensorflow")
    vars(tensorflow)["keras"] = SimpleNamespace(layers=layers)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.delitem(
        sys.modules, "tensorflow.keras.layers.experimental", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "tensorflow.keras.layers.experimental.preprocessing", raising=False
    )

    install_keras_preprocessing_compat()

    preprocessing = sys.modules["tensorflow.keras.layers.experimental.preprocessing"]
    experimental = sys.modules["tensorflow.keras.layers.experimental"]
    string_lookup = preprocessing.StringLookup("a", mask_value="[MASK]")
    integer_lookup = preprocessing.IntegerLookup(mask_value=None)

    assert experimental.preprocessing is preprocessing
    assert preprocessing.Discretization is _FakeDiscretization
    assert string_lookup.args == ("a",)
    assert string_lookup.kwargs == {"mask_token": "[MASK]"}
    assert integer_lookup.kwargs == {"mask_token": None}
    assert string_lookup.vocab_size() == 3
    assert integer_lookup.vocab_size() == 3


def test_install_assert_all_finite_compat_preserves_non_finite_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf = pytest.importorskip("tensorflow")
    original_assert_all_finite = tf.debugging.assert_all_finite
    monkeypatch.setattr(tf.debugging, "assert_all_finite", original_assert_all_finite)

    with pytest.raises(TypeError):
        tf.debugging.assert_all_finite(tf.constant([1.0]))

    install_assert_all_finite_compat()

    finite = tf.debugging.assert_all_finite(tf.constant([1.0]))
    assert finite.numpy().tolist() == [1.0]
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.debugging.assert_all_finite(tf.constant([float("nan")]))
