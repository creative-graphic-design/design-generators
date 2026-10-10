from pathlib import Path

import pytest

from traingen_parity.tensorflow_compat import (
    install_sparse_categorical_accuracy_compat,
)

pytestmark = pytest.mark.vendor_parity


def test_vendor_vector_metrics_concat_crello_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf = pytest.importorskip("tensorflow")
    original = tf.keras.metrics.sparse_categorical_accuracy
    monkeypatch.setattr(tf.keras.metrics, "sparse_categorical_accuracy", original)
    install_sparse_categorical_accuracy_compat()

    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / "vendor" / "canvas-vae" / "src" / "canvas-vae"
    monkeypatch.syspath_prepend(str(vendor_root))
    from canvasvae.models.metrics import VectorMetricLayer

    def column(
        field_type: str,
        input_dim: int | None,
        is_sequence: bool,
        shape: tuple[int, ...],
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "type": field_type,
            "shape": shape,
            "is_sequence": is_sequence,
        }
        if input_dim is not None:
            result["input_dim"] = input_dim

        return result

    columns = {
        "length": column("categorical", 50, False, (1,)),
        "group": column("categorical", 2, False, (1,)),
        "format": column("categorical", 2, False, (1,)),
        "canvas_width": column("categorical", 2, False, (1,)),
        "canvas_height": column("categorical", 2, False, (1,)),
        "category": column("categorical", 2, False, (1,)),
        "type": column("categorical", 2, True, (1,)),
        "left": column("categorical", 64, True, (1,)),
        "top": column("categorical", 64, True, (1,)),
        "width": column("categorical", 64, True, (1,)),
        "height": column("categorical", 64, True, (1,)),
        "opacity": column("categorical", 8, True, (1,)),
        "color": column("categorical", 16, True, (3,)),
        "image_embedding": column("numerical", None, True, (256,)),
    }
    y_true = {
        "length": tf.constant([[1]], dtype=tf.int32),
        "group": tf.constant([[0]], dtype=tf.int32),
        "format": tf.constant([[0]], dtype=tf.int32),
        "canvas_width": tf.constant([[0]], dtype=tf.int32),
        "canvas_height": tf.constant([[0]], dtype=tf.int32),
        "category": tf.constant([[0]], dtype=tf.int32),
        "type": tf.constant([[[0]]], dtype=tf.int32),
        "left": tf.constant([[[0]]], dtype=tf.int32),
        "top": tf.constant([[[0]]], dtype=tf.int32),
        "width": tf.constant([[[0]]], dtype=tf.int32),
        "height": tf.constant([[[0]]], dtype=tf.int32),
        "opacity": tf.constant([[[0]]], dtype=tf.int32),
        "color": tf.constant([[[0, 0, 0]]], dtype=tf.int32),
        "image_embedding": tf.ones((1, 1, 256), dtype=tf.float32),
    }
    y_pred = {
        "length": tf.zeros((1, 1, 50), dtype=tf.float32),
        "group": tf.zeros((1, 1, 2), dtype=tf.float32),
        "format": tf.zeros((1, 1, 2), dtype=tf.float32),
        "canvas_width": tf.zeros((1, 1, 2), dtype=tf.float32),
        "canvas_height": tf.zeros((1, 1, 2), dtype=tf.float32),
        "category": tf.zeros((1, 1, 2), dtype=tf.float32),
        "type": tf.zeros((1, 1, 1, 2), dtype=tf.float32),
        "left": tf.zeros((1, 1, 1, 64), dtype=tf.float32),
        "top": tf.zeros((1, 1, 1, 64), dtype=tf.float32),
        "width": tf.zeros((1, 1, 1, 64), dtype=tf.float32),
        "height": tf.zeros((1, 1, 1, 64), dtype=tf.float32),
        "opacity": tf.zeros((1, 1, 1, 8), dtype=tf.float32),
        "color": tf.zeros((1, 1, 3, 16), dtype=tf.float32),
        "image_embedding": tf.ones((1, 1, 256), dtype=tf.float32),
    }

    with tf.device("/CPU:0"):
        metrics = VectorMetricLayer(columns)((y_true, y_pred), training=False)

    assert metrics["group"].shape.as_list() == [1, 1]
    assert metrics["color"].shape.as_list() == [1, 3]
    assert metrics["total"].shape.as_list() == [1, 1]
    assert tf.reduce_all(tf.math.is_finite(metrics["total"]))
