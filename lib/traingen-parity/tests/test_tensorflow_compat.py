import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import traingen_parity.tensorflow_compat as tensorflow_compat
from traingen_parity.tensorflow_compat import (
    install_assert_all_finite_compat,
    install_keras_preprocessing_compat,
    install_sparse_categorical_accuracy_compat,
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


class _FakeShape:
    def __init__(self, dimensions: tuple[int, ...]) -> None:
        self.dimensions = dimensions
        self.rank = len(dimensions)

    def __getitem__(self, index: int) -> int:
        return self.dimensions[index]


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)
        self.shape = _FakeShape(self.values.shape)
        self.dtype = self.values.dtype


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


def test_install_sparse_categorical_accuracy_compat_with_fake_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def convert(value: object) -> _FakeTensor:
        return value if isinstance(value, _FakeTensor) else _FakeTensor(value)

    def squeeze(value: _FakeTensor, axis: int) -> _FakeTensor:
        return _FakeTensor(np.squeeze(value.values, axis=axis))

    def argmax(value: _FakeTensor, axis: int) -> _FakeTensor:
        return _FakeTensor(np.argmax(value.values, axis=axis))

    def cast(value: _FakeTensor, dtype: np.typing.DTypeLike) -> _FakeTensor:
        return _FakeTensor(value.values.astype(dtype))

    def equal(left: _FakeTensor, right: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(np.equal(left.values, right.values))

    def public_accuracy(y_true: _FakeTensor, y_pred: _FakeTensor) -> _FakeTensor:
        y_true = convert(y_true)
        y_pred = convert(y_pred)
        if y_true.shape.rank == y_pred.shape.rank and y_true.shape[-1] == 1:
            y_true = squeeze(y_true, axis=-1)
        matches = equal(y_true, argmax(y_pred, axis=-1))
        if matches.shape.rank > 1 and matches.shape[-1] == 1:
            matches = squeeze(matches, axis=-1)

        return cast(matches, np.float32)

    tensorflow = ModuleType("tensorflow")
    metrics = SimpleNamespace(sparse_categorical_accuracy=public_accuracy)
    vars(tensorflow).update(
        convert_to_tensor=convert,
        squeeze=squeeze,
        math=SimpleNamespace(argmax=argmax),
        cast=cast,
        equal=equal,
        keras=SimpleNamespace(
            metrics=metrics,
            backend=SimpleNamespace(floatx=lambda: np.float32),
        ),
    )
    monkeypatch.setattr(tensorflow_compat, "import_module", lambda _: tensorflow)

    mismatched_true = _FakeTensor([[1], [0]])
    mismatched_pred = _FakeTensor([[[0.1, 0.9]], [[0.8, 0.2]]])
    expected_mismatched = metrics.sparse_categorical_accuracy(
        mismatched_true, mismatched_pred
    )
    install_sparse_categorical_accuracy_compat()
    shim = metrics.sparse_categorical_accuracy
    actual_mismatched = shim(mismatched_true, mismatched_pred)

    assert actual_mismatched.shape.dimensions == (2, 1)
    assert np.array_equal(
        actual_mismatched.values.reshape(-1), expected_mismatched.values.reshape(-1)
    )

    equal_rank_true = _FakeTensor([[1], [0]])
    equal_rank_pred = _FakeTensor([[0.1, 0.9], [0.8, 0.2]])
    expected_equal_rank = public_accuracy(equal_rank_true, equal_rank_pred)
    actual_equal_rank = shim(equal_rank_true, equal_rank_pred)
    assert actual_equal_rank.shape.dimensions == expected_equal_rank.shape.dimensions
    assert np.array_equal(actual_equal_rank.values, expected_equal_rank.values)

    install_sparse_categorical_accuracy_compat()
    assert metrics.sparse_categorical_accuracy is shim


def test_install_sparse_categorical_accuracy_compat_preserves_tensorflow_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf = pytest.importorskip("tensorflow")
    original = tf.keras.metrics.sparse_categorical_accuracy
    monkeypatch.setattr(tf.keras.metrics, "sparse_categorical_accuracy", original)
    mismatched_true = tf.constant([[1], [0]], dtype=tf.int32)
    mismatched_pred = tf.constant([[[0.1, 0.9]], [[0.8, 0.2]]])
    expected_mismatched = original(mismatched_true, mismatched_pred)

    install_sparse_categorical_accuracy_compat()
    actual_mismatched = tf.keras.metrics.sparse_categorical_accuracy(
        mismatched_true, mismatched_pred
    )
    assert actual_mismatched.shape.as_list() == [2, 1]
    assert expected_mismatched.shape.as_list() == [2]
    assert np.array_equal(
        actual_mismatched.numpy(), expected_mismatched.numpy()[:, None]
    )

    equal_rank_true = tf.constant([[1], [0]], dtype=tf.int32)
    equal_rank_pred = tf.constant([[0.1, 0.9], [0.8, 0.2]])
    expected_equal_rank = original(equal_rank_true, equal_rank_pred)
    actual_equal_rank = tf.keras.metrics.sparse_categorical_accuracy(
        equal_rank_true, equal_rank_pred
    )
    assert actual_equal_rank.shape == expected_equal_rank.shape
    assert np.array_equal(actual_equal_rank.numpy(), expected_equal_rank.numpy())
