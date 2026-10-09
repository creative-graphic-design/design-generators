from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import tensorflow as tf

import numpy as np
import pytest
import torch

from pixel_vae import (
    PixelVAEConfig,
    PixelVAEEncoder,
    PixelVAEModel,
    PixelVAEImageProcessor,
)
from pixel_vae.conversion import (
    convert_tensorflow_variables,
    extract_tensorflow_variable_map,
    extract_tensorflow_weights,
    tensorflow_arrays_to_torch,
    tensorflow_state_sha256,
)
from pixel_vae.image_processing_pixel_vae import decode_pixelvae_png
from pixel_vae.testing import (
    JSONValue,
    assert_within_limits,
    float32_difference_summary,
    max_absolute_difference,
    next_trace_convolution,
    run_report_only_diagnostic,
)
from laygen.common.vendor import vendor_root

pytestmark = pytest.mark.vendor_parity

SOURCE_BYTES = 2_989_732_284
SOURCE_SHA256 = "f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e"
SOURCE_COMMIT = "bc1e2072ba3a253f1b099e8b0c604f6051e787da"
PARITY_DIR = Path(os.environ.get("PIXELVAE_PARITY_DIR", ".cache/pixel-vae/parity"))
TFRECORD_DIR = Path(
    os.environ.get("CRELLO_V1_TFRECORD_DIR", ".cache/pixel-vae/crello-v1-tfrecords")
)
AUDIT_PATH = Path(
    os.environ.get("CRELLO_V1_AUDIT_PATH", ".cache/pixel-vae/crello-v1-dimensions.json")
)
TRAINING_TYPES = {b"imageElement", b"maskElement", b"svgElement"}
EXPECTED_DOCUMENTS = {"train": 18_768, "val": 2_315, "test": 2_278}
DIAGNOSTIC_TRAIN_PAIR_COUNT = 8
CALIBRATION_SELECTION_RULE = (
    "deduplicated canonical train PNGs in UTF-8 document_id/source element order; "
    "repeat r uses document-stage indexes [2(r-1), 2r) and filtered training-stage "
    "indexes [6(r-1), 6r); each S4 type uses unique index (r-1) mod n, recording "
    "its unique count and whether it wrapped; distinct S4 IDs are required only "
    "when that type has at least three unique PNGs"
)


@dataclass(frozen=True)
class ImageExample:
    image_id: str
    document_id: str
    element_index: int
    element_type: bytes
    image_bytes: bytes


@dataclass(frozen=True)
class ElementTypeSelection:
    example: ImageExample
    unique_png_index: int
    unique_png_count: int
    wrapped: bool


@dataclass(frozen=True)
class ParityInputSelection:
    document_images: list[ImageExample]
    training_images: list[ImageExample]
    type_selections: list[ElementTypeSelection]
    diagnostic_pairs: list[tuple[str, list[ImageExample]]]
    record: dict[str, JSONValue]


def test_pixel_vae_cpu_stages(tmp_path: Path) -> None:
    """Compare stages while keeping diagnostic-only failures non-blocking."""
    if os.environ.get("PIXELVAE_PARITY_MODE", "heldout") == "diagnostic":
        run_report_only_diagnostic(
            lambda: _run_pixel_vae_cpu_stages(tmp_path),
            report_path=PARITY_DIR / "diagnostic" / "s1-diagnostic.json",
            metadata={
                "mode": "diagnostic",
                "diagnostic_only": True,
                "limits_applied": False,
                "source_commit": SOURCE_COMMIT,
                "input_image_ids": {},
            },
        )
        return

    _run_pixel_vae_cpu_stages(tmp_path)


def _run_pixel_vae_cpu_stages(tmp_path: Path) -> None:
    """Compare static state, forward, updates, traces, and source data on CPU."""
    mode = os.environ.get("PIXELVAE_PARITY_MODE", "heldout")
    repeat = int(os.environ.get("PIXELVAE_PARITY_REPEAT", "0"))
    required = os.environ.get("PARITY_REQUIRE") == "1"
    if mode not in {"calibration", "heldout", "diagnostic", "plan"}:
        raise ValueError(
            "PIXELVAE_PARITY_MODE must be calibration, heldout, diagnostic, or plan"
        )
    if mode == "calibration" and repeat not in {1, 2, 3}:
        raise ValueError("PIXELVAE_PARITY_REPEAT must be 1, 2, or 3")

    missing = [path for path in (TFRECORD_DIR, AUDIT_PATH) if not path.exists()]
    if missing:
        message = f"private Crello v1 parity assets are missing: {missing}"
        if required:
            raise FileNotFoundError(message)
        pytest.skip(message)

    if mode == "plan":
        _write_complete_input_plan()
        return

    selection = _select_parity_inputs(mode, repeat=repeat)
    document_images = selection.document_images
    training_images = selection.training_images
    calibration_selection = (
        {
            key: value
            for key, value in selection.record.items()
            if key not in {"mode", "repeat"}
        }
        if mode == "calibration"
        else None
    )
    diagnostic_pairs = selection.diagnostic_pairs
    _write_input_selection(selection)

    import tensorflow as tf

    vendor = vendor_root("canvas-vae", marker="src/pixel-vae/pixelvae/model.py")
    sys.path.insert(0, str(vendor / "src" / "pixel-vae"))
    from pixelvae.model import PixelVAE, reconstruction_loss_fn
    from traingen.optim import KerasAdam

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    diagnostic = mode == "diagnostic"
    calibration = mode == "calibration"
    diagnostic_report: dict[str, JSONValue] | None = None
    diagnostic_path = PARITY_DIR / "diagnostic" / "s1-diagnostic.json"
    if diagnostic:
        input_image_ids = {
            pair_name: [example.image_id for example in examples]
            for pair_name, examples in diagnostic_pairs
        }
        diagnostic_report = {
            "mode": "diagnostic",
            "status": "running",
            "diagnostic_only": True,
            "limits_applied": False,
            "seed": 0,
            "source_commit": SOURCE_COMMIT,
            "input_image_ids": input_image_ids,
        }
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(
            json.dumps(diagnostic_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    reference = PixelVAE(latent_dim=256, kl=100.0, l2=1e-6)
    source_weights = extract_tensorflow_weights(reference)
    encoder_weights = {
        key: value
        for key, value in source_weights.items()
        if not key.startswith("decoder/")
    }
    state_digest = tensorflow_state_sha256(encoder_weights)
    model = PixelVAEModel(PixelVAEConfig())
    conversion = convert_tensorflow_variables(source_weights, model)
    model.eval()

    # Require every source state key to map to an equal converted value.
    transformed_weights, mapping_report = tensorflow_arrays_to_torch(
        source_weights, model, strict=True
    )
    target_state = model.state_dict()
    if mapping_report.assigned_tensors != conversion.assigned_tensors:
        raise AssertionError("strict conversion and state-map tensor counts differ")
    if not all(
        torch.equal(target_state[key], value)
        for key, value in transformed_weights.items()
    ):
        raise AssertionError(
            "converted model state differs from transformed TensorFlow arrays"
        )
    if model.config.to_dict()["latent_dim"] != 256 or model.config.kl_weight != 100.0:
        raise AssertionError(
            "PixelVAE production configuration differs from the source"
        )

    # Compare image decoding, posterior statistics, logits, and objective.
    processor = PixelVAEImageProcessor()
    if diagnostic:
        if diagnostic_report is None:
            raise RuntimeError("diagnostic inputs were not recorded before S1")
        diagnostic_report["encoder_state_sha256"] = state_digest
        diagnostic_path.write_text(
            json.dumps(diagnostic_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _run_s1_diagnostic(
            reference,
            model,
            processor,
            diagnostic_pairs,
            state_digest=state_digest,
            source_commit=SOURCE_COMMIT,
        )
        return

    s1_examples = document_images[:2]
    tensorflow_images = _tensorflow_rgba(s1_examples)
    processed = processor([example.image_bytes for example in s1_examples])
    package_rgba = np.stack(
        [decode_pixelvae_png(example.image_bytes) for example in s1_examples]
    )
    if not np.array_equal(package_rgba, tensorflow_images.numpy()):
        raise AssertionError("Pillow RGBA pixels differ from TensorFlow PNG decode")
    tensorflow_inputs = tf.image.convert_image_dtype(
        tensorflow_images, tf.float32
    ).numpy()
    if not np.array_equal(
        processed.pixel_values.numpy(), tensorflow_inputs.transpose(0, 3, 1, 2)
    ):
        raise AssertionError("float32 preprocessing differs from TensorFlow conversion")

    source_means, source_log_variances = reference.encoder(
        tensorflow_images, training=False
    )
    source_logits = reference(tensorflow_images, training=False)
    source_reconstruction_loss = reconstruction_loss_fn(
        tf.bitwise.right_shift(tensorflow_images, 4), source_logits
    )
    source_kl_loss = -0.5 * (
        1
        + source_log_variances
        - tf.square(source_means)
        - tf.exp(source_log_variances)
    )
    source_kl_loss = tf.reduce_mean(source_kl_loss)
    source_total_loss = tf.add_n(reference.losses)
    with torch.inference_mode():
        output = model(processed.pixel_values, sample=False)
    assert output.posterior_mean is not None
    assert output.posterior_log_variance is not None
    assert output.logits is not None
    assert output.reconstruction_loss is not None
    assert output.kl_divergence is not None
    assert output.loss is not None
    s1_output_pairs = {
        "s1.posterior_mean.max_abs": (
            source_means.numpy(),
            output.posterior_mean.numpy(),
        ),
        "s1.posterior_log_variance.max_abs": (
            source_log_variances.numpy(),
            output.posterior_log_variance.numpy(),
        ),
        "s1.decoder_logits.max_abs": (source_logits.numpy(), output.logits.numpy()),
        "s1.reconstruction_loss.abs": (
            np.asarray(source_reconstruction_loss.numpy()),
            np.asarray(output.reconstruction_loss.numpy()),
        ),
        "s1.kl_loss.abs": (
            np.asarray(source_kl_loss.numpy()),
            np.asarray(output.kl_divergence.numpy()),
        ),
        "s1.total_loss.abs": (
            np.asarray(source_total_loss.numpy()),
            np.asarray(output.loss.numpy()),
        ),
    }
    s1_output_comparisons = {
        name: float32_difference_summary(reference_values, package_values)
        for name, (reference_values, package_values) in s1_output_pairs.items()
    }
    metrics = {
        name: comparison["max_abs_diff"]
        for name, comparison in s1_output_comparisons.items()
    }

    # Compare gradients, Keras Adam slots, and one updated parameter state.
    initial_source_weights = reference.get_weights()
    initial_target_state = copy.deepcopy(model.state_dict())
    s2_images = _tensorflow_rgba(training_images[:2])
    s2_inputs = processor(
        [example.image_bytes for example in training_images[:2]]
    ).pixel_values
    s2_epsilon = np.random.default_rng(206).standard_normal((2, 256)).astype(np.float32)
    source_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
    source_loss, source_gradients, _ = _tensorflow_train_step(
        reference, source_optimizer, s2_images, s2_epsilon
    )
    model.train()
    model.zero_grad(set_to_none=True)
    package_output = model(
        s2_inputs,
        sample=True,
        epsilon=torch.from_numpy(s2_epsilon),
    )
    assert package_output.loss is not None
    package_output.loss.backward()
    package_optimizer = KerasAdam(model.parameters(), lr=1e-4, eps=1e-7)
    source_variable_map = extract_tensorflow_variable_map(reference)
    source_grad_arrays = _source_gradient_arrays(
        reference, source_variable_map, source_gradients
    )
    mapped_gradients, _ = tensorflow_arrays_to_torch(
        source_grad_arrays, model, strict=False
    )
    package_gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if set(mapped_gradients) != set(package_gradients):
        raise AssertionError(
            "TensorFlow and PyTorch trainable-gradient key sets differ"
        )
    metrics["s2.loss.abs"] = max_absolute_difference(
        np.asarray(source_loss.numpy()),
        np.asarray(package_output.loss.detach().numpy()),
    )
    metrics["s2.gradients.max_abs"] = max(
        max_absolute_difference(
            mapped_gradients[key].numpy(), package_gradients[key].numpy()
        )
        for key in mapped_gradients
    )
    source_optimizer.apply_gradients(
        zip(source_gradients, reference.trainable_variables, strict=True)
    )
    package_optimizer.step()
    metrics["s2.adam_first_moment.max_abs"] = _compare_optimizer_slots(
        source_optimizer,
        package_optimizer,
        reference,
        source_variable_map,
        model,
        slot_name="_momentums",
        package_slot="exp_avg",
    )
    metrics["s2.adam_second_moment.max_abs"] = _compare_optimizer_slots(
        source_optimizer,
        package_optimizer,
        reference,
        source_variable_map,
        model,
        slot_name="_velocities",
        package_slot="exp_avg_sq",
    )
    metrics["s2.parameters.max_abs"] = _compare_trainable_parameters(reference, model)
    metrics["s2.batch_norm_state.max_abs"] = _compare_batch_norm_state(reference, model)

    # Compare three deterministic batches from a reset untrained state.
    reference.set_weights(initial_source_weights)
    model.load_state_dict(initial_target_state)
    source_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
    package_optimizer = KerasAdam(model.parameters(), lr=1e-4, eps=1e-7)
    model.train()
    trace_loss_differences: list[float] = []
    trace_mean_differences: list[float] = []
    trace_parameter_differences: list[float] = []
    trace_batch_norm_differences: list[float] = []
    for step in range(3):
        batch = training_images[step * 2 : (step + 1) * 2]
        tf_batch = _tensorflow_rgba(batch)
        pt_batch = processor([example.image_bytes for example in batch]).pixel_values
        epsilon = (
            np.random.default_rng(310 + step)
            .standard_normal((2, 256))
            .astype(np.float32)
        )
        step_loss, step_gradients, source_step_mean = _tensorflow_train_step(
            reference, source_optimizer, tf_batch, epsilon
        )
        package_optimizer.zero_grad(set_to_none=True)
        step_output = model(pt_batch, sample=True, epsilon=torch.from_numpy(epsilon))
        assert step_output.loss is not None and step_output.posterior_mean is not None
        step_output.loss.backward()
        trace_loss_differences.append(
            max_absolute_difference(
                np.asarray(step_loss.numpy()),
                np.asarray(step_output.loss.detach().numpy()),
            )
        )
        trace_mean_differences.append(
            max_absolute_difference(
                source_step_mean.numpy(), step_output.posterior_mean.detach().numpy()
            )
        )
        source_optimizer.apply_gradients(
            zip(step_gradients, reference.trainable_variables, strict=True)
        )
        package_optimizer.step()
        trace_parameter_differences.append(
            _compare_trainable_parameters(reference, model)
        )
        trace_batch_norm_differences.append(_compare_batch_norm_state(reference, model))
    metrics["s3.step_losses.max_abs"] = max(trace_loss_differences)
    metrics["s3.posterior_means.max_abs"] = max(trace_mean_differences)
    metrics["s3.parameters.max_abs"] = max(trace_parameter_differences)
    metrics["s3.batch_norm_state.max_abs"] = max(trace_batch_norm_differences)

    # Audit every document element type and verify encoder serialization.
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("source_bytes") != SOURCE_BYTES
        or audit.get("source_sha256") != SOURCE_SHA256
    ):
        raise AssertionError(
            "Crello v1 audit does not match the pinned private archive"
        )
    if audit.get("document_counts") != EXPECTED_DOCUMENTS:
        raise AssertionError(
            f"unexpected v1 document counts: {audit.get('document_counts')}"
        )
    non_256 = audit.get("non_256_examples", {})
    invalid_pngs = audit.get("invalid_png_counts", {})
    if non_256 or invalid_pngs:
        raise AssertionError(
            "Crello v1 contains non-256x256 or invalid PNGs by element type: "
            f"non_256_examples={non_256}, invalid_png_counts={invalid_pngs}"
        )
    type_selections = selection.type_selections
    type_examples = [selected.example for selected in type_selections]
    if {example.element_type.decode("utf-8") for example in type_examples} != set(
        audit["dimensions_by_element_type"]
    ):
        raise AssertionError(
            "the audit did not select a document PNG for every source element type"
        )
    type_tensorflow_images = _tensorflow_rgba(type_examples)
    type_processed = processor([example.image_bytes for example in type_examples])
    type_package_rgba = np.stack(
        [decode_pixelvae_png(example.image_bytes) for example in type_examples]
    )
    if not np.array_equal(type_package_rgba, type_tensorflow_images.numpy()):
        raise AssertionError("RGBA decode differs for a document element type")
    type_tensorflow_inputs = tf.image.convert_image_dtype(
        type_tensorflow_images, tf.float32
    ).numpy()
    if not np.array_equal(
        type_processed.pixel_values.numpy(),
        type_tensorflow_inputs.transpose(0, 3, 1, 2),
    ):
        raise AssertionError("preprocessing differs for a document element type")

    reference.set_weights(initial_source_weights)
    model.load_state_dict(initial_target_state)
    model.eval()
    source_type_means, _ = reference.encoder(type_tensorflow_images, training=False)
    with torch.inference_mode():
        original_embeddings = model.encode(type_processed.pixel_values)
    metrics["s4.posterior_means.max_abs"] = max_absolute_difference(
        source_type_means.numpy(), original_embeddings.numpy()
    )
    model.save_pretrained(tmp_path / "full-model")
    model.encoder.save_pretrained(tmp_path / "encoder")
    encoder = PixelVAEEncoder.from_pretrained(tmp_path / "encoder").eval()
    encoder.save_pretrained(tmp_path / "encoder-roundtrip")
    restored_encoder = PixelVAEEncoder.from_pretrained(
        tmp_path / "encoder-roundtrip"
    ).eval()
    with torch.inference_mode():
        restored_embeddings = restored_encoder.encode(type_processed.pixel_values)
    if not torch.equal(original_embeddings, restored_embeddings):
        raise AssertionError(
            "posterior means changed across save_pretrained round trip"
        )

    result = {
        "mode": mode,
        "repeat": repeat if calibration else None,
        "process_id": os.getpid(),
        "seed": 0,
        "tensorflow_version": tf.__version__,
        "source_commit": SOURCE_COMMIT,
        "encoder_state_sha256": state_digest,
        "configuration": model.config.to_dict(),
        "s0_assigned_tensors": conversion.assigned_tensors,
        "s1_image_ids": [example.image_id for example in s1_examples],
        "calibration_selection": calibration_selection,
        "s2_image_ids": [example.image_id for example in training_images[:2]],
        "s3_image_ids": [example.image_id for example in training_images[:6]],
        "s4_element_types": [
            example.element_type.decode("utf-8") for example in type_examples
        ],
        "s4_image_ids": [example.image_id for example in type_examples],
        "s4_type_selections": _element_type_selection_records(type_selections),
        "s4_document_counts": audit["document_counts"],
        "s4_element_counts": audit["element_counts"],
        "s4_dimensions_by_element_type": audit["dimensions_by_element_type"],
        "s4_embedding_shape": list(original_embeddings.shape),
        "s4_embedding_dtype": str(original_embeddings.dtype),
        "metrics": metrics,
        "s1_output_comparisons": s1_output_comparisons,
    }
    output_path = (
        PARITY_DIR / "calibration" / f"repeat-{repeat}.json"
        if calibration
        else PARITY_DIR / "heldout.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limits: dict[str, float] = {}
    if not calibration:
        limits_path = PARITY_DIR / "calibration" / "limits.json"
        if not limits_path.is_file():
            if required:
                raise FileNotFoundError(f"frozen CPU limits are missing: {limits_path}")
            pytest.skip(f"frozen CPU limits are missing: {limits_path}")
        limits_record = json.loads(limits_path.read_text(encoding="utf-8"))
        if limits_record.get("encoder_state_sha256") != state_digest:
            raise AssertionError(
                "held-out encoder state differs from the calibrated state"
            )
        limits = limits_record["limits"]
        result["calibrated_limits"] = limits
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not calibration:
        assert_within_limits(metrics, limits)


def _run_s1_diagnostic(
    reference: tf.keras.Model,
    model: PixelVAEModel,
    processor: PixelVAEImageProcessor,
    image_pairs: list[tuple[str, list[ImageExample]]],
    *,
    state_digest: str,
    source_commit: str,
) -> None:
    import tensorflow as tf
    from pixelvae.model import reconstruction_loss_fn

    cnn = reference.encoder.cnn
    source_layer_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.ZeroPadding2D,
        tf.keras.layers.ReLU,
        tf.keras.layers.Add,
        tf.keras.layers.GlobalAveragePooling2D,
    )
    source_layers = [
        layer
        for layer in cnn.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    unsupported_layers = [
        layer.name
        for layer in source_layers
        if not isinstance(layer, source_layer_types)
    ]
    if unsupported_layers:
        raise ValueError(
            f"unmapped TensorFlow MobileNetV2 layers: {unsupported_layers}"
        )
    source_trace_model = tf.keras.Model(
        inputs=cnn.input,
        outputs=[layer.output for layer in source_layers],
    )
    backbone = model.encoder.backbone
    target_modules: dict[str, torch.nn.Module] = {}
    padded_layer_targets: dict[str, str] = {}
    added_layer_targets: dict[str, str] = {}
    for index, layer in enumerate(source_layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            target_modules[layer.name] = backbone
            continue

        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            target_modules[layer.name] = backbone.layers[layer.name]
            continue

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            target_modules[layer.name] = backbone.layers[layer.name]
            continue

        next_convolution_name = next_trace_convolution(
            layer.name,
            type(layer).__name__,
            [
                (candidate.name, type(candidate).__name__)
                for candidate in source_layers[index + 1 :]
            ],
        )
        if next_convolution_name is None:
            continue

        if isinstance(layer, tf.keras.layers.ZeroPadding2D):
            padded_layer_targets[layer.name] = next_convolution_name
        elif isinstance(layer, tf.keras.layers.Add):
            added_layer_targets[layer.name] = next_convolution_name
    target_modules["z_mean"] = model.encoder.z_mean
    target_modules["z_log_sigma"] = model.encoder.z_log_variance
    layer_names = [layer.name for layer in source_layers] + [
        "z_mean",
        "z_log_sigma",
    ]

    pair_records = []
    for pair_name, examples in image_pairs:
        if len(examples) != 2 or len({example.image_id for example in examples}) != 2:
            raise ValueError(
                f"diagnostic pair must contain two unique IDs: {pair_name}"
            )

        tensorflow_images = _tensorflow_rgba(examples)
        package_rgba = np.stack(
            [decode_pixelvae_png(example.image_bytes) for example in examples]
        )
        tensorflow_inputs = tf.image.convert_image_dtype(tensorflow_images, tf.float32)
        processed = processor([example.image_bytes for example in examples])
        tensorflow_preprocessed = tensorflow_inputs.numpy().transpose(0, 3, 1, 2)
        decode_matches = bool(np.array_equal(package_rgba, tensorflow_images.numpy()))
        preprocessing_matches = bool(
            np.array_equal(processed.pixel_values.numpy(), tensorflow_preprocessed)
        )

        source_trace_values = source_trace_model(tensorflow_inputs, training=False)
        if not isinstance(source_trace_values, (list, tuple)):
            source_trace_values = [source_trace_values]
        source_layer_values = {
            layer.name: np.asarray(value.numpy())
            for layer, value in zip(source_layers, source_trace_values, strict=True)
        }
        source_global_average = next(
            layer
            for layer in source_layers
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
        )
        source_features = tf.convert_to_tensor(
            source_layer_values[source_global_average.name]
        )
        source_means, source_log_variances = reference.encoder.head(source_features)
        source_latents = reference.sampling(
            (source_means, source_log_variances), training=False
        )
        source_logits = reference.decoder(source_latents, training=False)
        source_reconstruction_loss = reconstruction_loss_fn(
            tf.bitwise.right_shift(tensorflow_images, 4), source_logits
        )
        source_kl_loss = -0.5 * (
            1
            + source_log_variances
            - tf.square(source_means)
            - tf.exp(source_log_variances)
        )
        source_kl_loss = tf.reduce_mean(source_kl_loss)
        source_total_loss = tf.add_n(reference.losses)
        source_layer_values["z_mean"] = np.asarray(source_means.numpy())
        source_layer_values["z_log_sigma"] = np.asarray(source_log_variances.numpy())

        package_layer_values: dict[str, np.ndarray] = {}
        package_preconvolution_values: dict[str, np.ndarray] = {}

        def channels_last(values: torch.Tensor) -> np.ndarray:
            array = values.detach().cpu().numpy()
            if array.ndim == 4:
                array = array.transpose(0, 2, 3, 1)
            return np.asarray(array)

        def capture_layer(name: str):
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                package_layer_values[name] = channels_last(output)

            return hook

        def capture_padded_layer(name: str):
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
            ) -> None:
                package_layer_values[name] = channels_last(inputs[0])

            return hook

        def capture_relu_layer(name: str):
            def capture(inputs: torch.Tensor) -> torch.Tensor:
                output = original_relu6(inputs)
                package_layer_values[name] = channels_last(output)
                return output

            return capture

        handles = [
            module.register_forward_hook(capture_layer(name))
            for name, module in target_modules.items()
        ]
        handles.extend(
            backbone.layers[target_name].register_forward_pre_hook(
                capture_padded_layer(source_name)
            )
            for source_name, target_name in padded_layer_targets.items()
        )
        relu_names = [
            layer.name
            for layer in source_layers
            if isinstance(layer, tf.keras.layers.ReLU)
        ]
        import pixel_vae.modeling_pixel_vae as modeling_pixel_vae

        original_relu6 = modeling_pixel_vae.F.relu6
        original_apply_conv = backbone._apply_conv
        relu_capture = iter(relu_names)

        def capture_relu6(inputs: torch.Tensor) -> torch.Tensor:
            return capture_relu_layer(next(relu_capture))(inputs)

        def capture_preconvolution(name: str, inputs: torch.Tensor) -> torch.Tensor:
            package_preconvolution_values[name] = channels_last(inputs)
            return original_apply_conv(name, inputs)

        try:
            with (
                patch.object(
                    modeling_pixel_vae.F,
                    "relu6",
                    side_effect=capture_relu6,
                ),
                patch.object(
                    backbone, "_apply_conv", side_effect=capture_preconvolution
                ),
                torch.inference_mode(),
            ):
                output = model(processed.pixel_values, sample=False)
        finally:
            for handle in handles:
                handle.remove()

        for layer_name, convolution_name in added_layer_targets.items():
            package_layer_values[layer_name] = package_preconvolution_values[
                convolution_name
            ]

        if (
            output.posterior_mean is None
            or output.posterior_log_variance is None
            or output.logits is None
            or output.reconstruction_loss is None
            or output.kl_divergence is None
            or output.loss is None
        ):
            raise ValueError("PixelVAE diagnostic forward omitted an S1 output")

        output_pairs = {
            "s1.posterior_mean.max_abs": (
                np.asarray(source_means.numpy()),
                output.posterior_mean.numpy(),
            ),
            "s1.posterior_log_variance.max_abs": (
                np.asarray(source_log_variances.numpy()),
                output.posterior_log_variance.numpy(),
            ),
            "s1.decoder_logits.max_abs": (
                np.asarray(source_logits.numpy()),
                output.logits.numpy(),
            ),
            "s1.reconstruction_loss.abs": (
                np.asarray(source_reconstruction_loss.numpy()),
                np.asarray(output.reconstruction_loss.numpy()),
            ),
            "s1.kl_loss.abs": (
                np.asarray(source_kl_loss.numpy()),
                np.asarray(output.kl_divergence.numpy()),
            ),
            "s1.total_loss.abs": (
                np.asarray(source_total_loss.numpy()),
                np.asarray(output.loss.numpy()),
            ),
        }
        output_comparisons = {
            name: float32_difference_summary(reference_values, package_values)
            for name, (reference_values, package_values) in output_pairs.items()
        }
        layer_comparisons = []
        first_differing_layer = None
        for name in layer_names:
            comparison = float32_difference_summary(
                source_layer_values[name], package_layer_values[name]
            )
            if first_differing_layer is None and comparison["max_abs_diff"] > 0:
                first_differing_layer = name
            layer_comparisons.append({"layer": name, **comparison})
        metrics = {
            name: comparison["max_abs_diff"]
            for name, comparison in output_comparisons.items()
        }
        pair_records.append(
            {
                "pair_id": pair_name,
                "image_ids": [example.image_id for example in examples],
                "decode_matches_tensorflow": decode_matches,
                "preprocessing_matches_tensorflow": preprocessing_matches,
                "metrics": metrics,
                "output_comparisons": output_comparisons,
                "first_differing_layer": first_differing_layer,
                "layer_comparisons": layer_comparisons,
            }
        )

    train_ids = [
        image_id
        for pair_name, examples in image_pairs
        if pair_name != "heldout"
        for image_id in (example.image_id for example in examples)
    ]
    if len(train_ids) != DIAGNOSTIC_TRAIN_PAIR_COUNT * 2 or len(set(train_ids)) != len(
        train_ids
    ):
        raise ValueError("diagnostic train pairs must use distinct image IDs")

    report = {
        "mode": "diagnostic",
        "status": "complete",
        "diagnostic_only": True,
        "limits_applied": False,
        "seed": 0,
        "process_id": os.getpid(),
        "tensorflow_version": tf.__version__,
        "torch_version": torch.__version__,
        "source_commit": source_commit,
        "encoder_state_sha256": state_digest,
        "configuration": model.config.to_dict(),
        "comparison_dtype": "float32",
        "relative_difference_definition": "max_abs_diff / reference_max_abs",
        "ulp_definition": "ordered IEEE-754 binary32 integer distance at max_abs_diff index",
        "train_pair_count": DIAGNOSTIC_TRAIN_PAIR_COUNT,
        "input_image_ids": {
            pair_name: [example.image_id for example in examples]
            for pair_name, examples in image_pairs
        },
        "pairs": pair_records,
    }
    output_path = PARITY_DIR / "diagnostic" / "s1-diagnostic.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"DIAGNOSTIC_REPORT={output_path}")

    failed_exact_checks = [
        record["pair_id"]
        for record in pair_records
        if not record["decode_matches_tensorflow"]
        or not record["preprocessing_matches_tensorflow"]
    ]
    if failed_exact_checks:
        raise AssertionError(
            "diagnostic PNG decode or preprocessing differs for pairs: "
            f"{failed_exact_checks}"
        )


def _select_parity_inputs(mode: str, *, repeat: int = 0) -> ParityInputSelection:
    if mode == "calibration":
        if repeat not in {1, 2, 3}:
            raise ValueError("calibration repeat must be 1, 2, or 3")
        document_pool = _select_images("train", 6, filter_training_types=False)
        training_pool = _select_images("train", 18, filter_training_types=True)
        document_indices = list(range((repeat - 1) * 2, repeat * 2))
        training_indices = list(range((repeat - 1) * 6, repeat * 6))
        _require_image_count(document_pool, 6, split="train", stage="calibration S1")
        _require_image_count(
            training_pool, 18, split="train", stage="calibration S2/S3"
        )
        document_images = [document_pool[index] for index in document_indices]
        training_images = [training_pool[index] for index in training_indices]
        type_selections = _select_one_per_element_type("train", occurrence=repeat - 1)
        _require_audit_type_coverage(type_selections)
        type_examples = [selected.example for selected in type_selections]
        record: dict[str, JSONValue] = {
            "mode": mode,
            "repeat": repeat,
            "rule": CALIBRATION_SELECTION_RULE,
            "document_png_indices": document_indices,
            "training_png_indices": training_indices,
            "s1_image_ids": [example.image_id for example in document_images],
            "training_image_ids": [example.image_id for example in training_images],
            "s4_element_type_occurrence": repeat - 1,
            "s4_image_ids": [example.image_id for example in type_examples],
            "s4_type_selections": _element_type_selection_records(type_selections),
        }
        return ParityInputSelection(
            document_images, training_images, type_selections, [], record
        )

    if mode == "heldout":
        document_images = _select_images("test", 6, filter_training_types=False)
        training_images = _select_images("test", 6, filter_training_types=True)
        _require_image_count(document_images, 6, split="test", stage="held-out S1")
        _require_image_count(training_images, 6, split="test", stage="held-out S2/S3")
        type_selections = _select_one_per_element_type("test")
        _require_audit_type_coverage(type_selections)
        type_examples = [selected.example for selected in type_selections]
        record: dict[str, JSONValue] = {
            "mode": mode,
            "selection_rule": "first canonical test PNGs by document and source element order",
            "s1_image_ids": [example.image_id for example in document_images[:2]],
            "s2_image_ids": [example.image_id for example in training_images[:2]],
            "s3_image_ids": [example.image_id for example in training_images[:6]],
            "s4_image_ids": [example.image_id for example in type_examples],
            "s4_type_selections": _element_type_selection_records(type_selections),
        }
        return ParityInputSelection(
            document_images, training_images, type_selections, [], record
        )

    if mode == "diagnostic":
        heldout_documents = _select_images("test", 6, filter_training_types=False)
        train_documents = _select_images(
            "train", DIAGNOSTIC_TRAIN_PAIR_COUNT * 2, filter_training_types=False
        )
        training_images = _select_images("train", 6, filter_training_types=True)
        _require_image_count(
            heldout_documents, 6, split="test", stage="diagnostic held-out S1"
        )
        _require_image_count(
            train_documents,
            DIAGNOSTIC_TRAIN_PAIR_COUNT * 2,
            split="train",
            stage="diagnostic train S1",
        )
        _require_image_count(
            training_images, 6, split="train", stage="diagnostic training"
        )
        pairs = [
            ("heldout", heldout_documents[:2]),
            *[
                (
                    f"train-{pair_index + 1:02d}",
                    train_documents[pair_index * 2 : pair_index * 2 + 2],
                )
                for pair_index in range(DIAGNOSTIC_TRAIN_PAIR_COUNT)
            ],
        ]
        record: dict[str, JSONValue] = {
            "mode": mode,
            "diagnostic_only": True,
            "input_image_ids": {
                pair_name: [example.image_id for example in examples]
                for pair_name, examples in pairs
            },
        }
        return ParityInputSelection(train_documents, training_images, [], pairs, record)

    raise ValueError(f"no input-selection rule exists for mode {mode!r}")


def _require_image_count(
    examples: list[ImageExample], count: int, *, split: str, stage: str
) -> None:
    if len(examples) < count:
        raise ValueError(
            f"{split} {stage} requires {count} unique PNGs; found {len(examples)}"
        )


def _require_audit_type_coverage(selections: list[ElementTypeSelection]) -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    selected_types = {
        selection.example.element_type.decode("utf-8") for selection in selections
    }
    expected_types = set(audit["dimensions_by_element_type"])
    if selected_types != expected_types:
        raise ValueError(
            "input selection does not cover every audited element type: "
            f"selected={sorted(selected_types)}, expected={sorted(expected_types)}"
        )


def _element_type_selection_records(
    selections: list[ElementTypeSelection],
) -> list[dict[str, JSONValue]]:
    return [
        {
            "element_type": selection.example.element_type.decode("utf-8"),
            "image_id": selection.example.image_id,
            "unique_png_index": selection.unique_png_index,
            "unique_png_count": selection.unique_png_count,
            "wrapped": selection.wrapped,
        }
        for selection in selections
    ]


def _write_complete_input_plan() -> None:
    selections = []
    errors = []
    for mode, repeat in (
        ("diagnostic", 0),
        ("calibration", 1),
        ("calibration", 2),
        ("calibration", 3),
        ("heldout", 0),
    ):
        try:
            selections.append(_select_parity_inputs(mode, repeat=repeat))
        except Exception as error:
            errors.append({"mode": mode, "repeat": repeat or None, "error": str(error)})

    for selection in selections:
        _write_input_selection(selection)
    plan = {
        "mode": "plan",
        "selection_files": [
            str(_selection_path(selection)) for selection in selections
        ],
        "selection_errors": errors,
    }
    plan_path = PARITY_DIR / "input-selection-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    _write_selection_record(plan_path, plan)
    print(f"INPUT_SELECTION_PLAN={plan_path}")
    if errors:
        raise ValueError(
            "input-selection plan found invalid stages: "
            + json.dumps(errors, sort_keys=True)
        )


def _selection_path(selection: ParityInputSelection) -> Path:
    if selection.record["mode"] == "calibration":
        return (
            PARITY_DIR
            / "calibration"
            / f"repeat-{selection.record['repeat']}.inputs.json"
        )
    if selection.record["mode"] == "heldout":
        return PARITY_DIR / "heldout.inputs.json"
    return PARITY_DIR / "diagnostic" / "inputs.json"


def _write_input_selection(selection: ParityInputSelection) -> None:
    _write_selection_record(_selection_path(selection), selection.record)


def _write_selection_record(path: Path, record: Mapping[str, object]) -> None:
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to replace input selection: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def _select_images(
    split: str,
    count: int,
    *,
    filter_training_types: bool,
) -> list[ImageExample]:
    import tensorflow as tf

    records = list(_record_index(split))
    examples: list[ImageExample] = []
    seen: set[str] = set()
    for document_key, document_id, path, payload_offset, payload_length in sorted(
        records
    ):
        del document_key
        with path.open("rb") as stream:
            stream.seek(payload_offset)
            serialized = stream.read(payload_length)
        sequence = tf.train.SequenceExample.FromString(serialized)
        types = sequence.feature_lists.feature_list["type"].feature
        images = sequence.feature_lists.feature_list["image_bytes"].feature
        for index, (element_type, image) in enumerate(zip(types, images, strict=True)):
            kind = element_type.bytes_list.value[0]
            if filter_training_types and kind not in TRAINING_TYPES:
                continue
            png_bytes = image.bytes_list.value[0]
            image_id = f"crello-v1/{split}/{hashlib.sha256(png_bytes).hexdigest()}"
            if image_id in seen:
                continue
            seen.add(image_id)
            examples.append(ImageExample(image_id, document_id, index, kind, png_bytes))
            if len(examples) == count:
                return examples
    return examples


def _select_one_per_element_type(
    split: str, *, occurrence: int = 0
) -> list[ElementTypeSelection]:
    if occurrence < 0:
        raise ValueError("element-type occurrence must be nonnegative")

    import tensorflow as tf

    examples: dict[bytes, list[ImageExample]] = {}
    seen: dict[bytes, set[str]] = {}
    records = sorted(_record_index(split))
    for _document_key, document_id, path, payload_offset, payload_length in records:
        with path.open("rb") as stream:
            stream.seek(payload_offset)
            serialized = stream.read(payload_length)
        sequence = tf.train.SequenceExample.FromString(serialized)
        types = sequence.feature_lists.feature_list["type"].feature
        images = sequence.feature_lists.feature_list["image_bytes"].feature
        for index, (element_type, image) in enumerate(zip(types, images, strict=True)):
            kind = element_type.bytes_list.value[0]
            png_bytes = image.bytes_list.value[0]
            image_id = f"crello-v1/{split}/{hashlib.sha256(png_bytes).hexdigest()}"
            seen.setdefault(kind, set())
            examples.setdefault(kind, [])
            if image_id in seen[kind]:
                continue
            seen[kind].add(image_id)
            if len(examples[kind]) <= occurrence:
                examples[kind].append(
                    ImageExample(image_id, document_id, index, kind, png_bytes)
                )

    if not seen:
        raise ValueError(
            f"split {split} has no unique PNGs for an element-type selection"
        )
    return [
        ElementTypeSelection(
            example=examples[element_type][occurrence % len(image_ids)],
            unique_png_index=occurrence % len(image_ids),
            unique_png_count=len(image_ids),
            wrapped=occurrence >= len(image_ids),
        )
        for element_type, image_ids in seen.items()
    ]


def _record_index(split: str) -> Iterator[tuple[bytes, str, Path, int, int]]:
    import tensorflow as tf

    for path in sorted(TFRECORD_DIR.rglob(f"{split}-*.tfrecord")):
        with path.open("rb") as stream:
            while length_bytes := stream.read(8):
                if len(length_bytes) != 8:
                    raise ValueError(f"truncated TFRecord length in {path}")
                payload_length = struct.unpack("<Q", length_bytes)[0]
                if len(stream.read(4)) != 4:
                    raise ValueError(f"truncated TFRecord length checksum in {path}")
                payload_offset = stream.tell()
                payload = stream.read(payload_length)
                if len(payload) != payload_length or len(stream.read(4)) != 4:
                    raise ValueError(f"truncated TFRecord payload in {path}")
                sequence = tf.train.SequenceExample.FromString(payload)
                document_id = (
                    sequence.context.feature["id"].bytes_list.value[0].decode("utf-8")
                )
                sort_key = f"crello-v1/{split}/{document_id}".encode("utf-8")
                yield sort_key, document_id, path, payload_offset, payload_length


def test_selection_plans_cover_calibration_repeats_and_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_archive(tmp_path, monkeypatch)

    all_images = _select_images("train", 100, filter_training_types=False)
    training_images = _select_images("train", 100, filter_training_types=True)
    assert len(all_images) == 34
    assert len(training_images) == 20
    assert all(example.element_type in TRAINING_TYPES for example in training_images)
    assert len({example.image_id for example in all_images}) == len(all_images)

    calibration = [
        _select_parity_inputs("calibration", repeat=repeat) for repeat in (1, 2, 3)
    ]
    assert [selection.record["document_png_indices"] for selection in calibration] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert [selection.record["training_png_indices"] for selection in calibration] == [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
    ]
    for field in ("s1_image_ids", "training_image_ids"):
        repeated_ids = [
            set(_record_string_ids(selection.record, field))
            for selection in calibration
        ]
        assert all(
            first.isdisjoint(second)
            for index, first in enumerate(repeated_ids)
            for second in repeated_ids[index + 1 :]
        )
    s4_by_type = [
        {
            selected.example.element_type.decode("utf-8"): selected
            for selected in selection.type_selections
        }
        for selection in calibration
    ]
    assert all(set(selection) == set(s4_by_type[0]) for selection in s4_by_type[1:])
    for element_type in s4_by_type[0]:
        selected = [selection[element_type] for selection in s4_by_type]
        unique_counts = {item.unique_png_count for item in selected}
        assert len(unique_counts) == 1
        unique_count = selected[0].unique_png_count
        if unique_count >= 3:
            assert len({item.example.image_id for item in selected}) == 3
        else:
            assert [item.unique_png_index for item in selected] == [
                repeat % unique_count for repeat in range(3)
            ]
            assert [item.wrapped for item in selected] == [
                repeat >= unique_count for repeat in range(3)
            ]

    heldout = _select_parity_inputs("heldout")
    assert len(heldout.document_images) == 6
    assert len(heldout.training_images) == 6
    assert len(heldout.type_selections) == 5
    assert heldout.record["s1_image_ids"] == [
        example.image_id for example in heldout.document_images[:2]
    ]
    assert heldout.record["s2_image_ids"] == [
        example.image_id for example in heldout.training_images[:2]
    ]
    assert heldout.record["s3_image_ids"] == [
        example.image_id for example in heldout.training_images[:6]
    ]

    diagnostic = _select_parity_inputs("diagnostic")
    assert len(diagnostic.diagnostic_pairs) == DIAGNOSTIC_TRAIN_PAIR_COUNT + 1
    assert len(diagnostic.diagnostic_pairs[0][1]) == 2
    assert all(len(pair[1]) == 2 for pair in diagnostic.diagnostic_pairs[1:])
    train_pair_ids = [
        {example.image_id for example in pair}
        for _pair_name, pair in diagnostic.diagnostic_pairs[1:]
    ]
    assert all(
        first.isdisjoint(second)
        for index, first in enumerate(train_pair_ids)
        for second in train_pair_ids[index + 1 :]
    )


def test_selection_plan_reports_short_lists_and_sparse_element_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_root = tmp_path / "short-tfrecords"
    short_root.mkdir()
    _write_synthetic_split(
        short_root,
        "test",
        [
            ("test-doc-0", [(b"imageElement", b"same-png")]),
            ("test-doc-1", [(b"imageElement", b"same-png")]),
        ],
    )
    monkeypatch.setattr(sys.modules[__name__], "TFRECORD_DIR", short_root)
    assert len(_select_images("test", 6, filter_training_types=True)) == 1
    with pytest.raises(ValueError, match="held-out S1 requires 6 unique PNGs; found 1"):
        _select_parity_inputs("heldout")

    calibration_short_root = tmp_path / "calibration-short-tfrecords"
    calibration_short_root.mkdir()
    _write_synthetic_split(
        calibration_short_root,
        "train",
        [("train-doc-0", [(b"imageElement", b"one-png")])],
    )
    monkeypatch.setattr(sys.modules[__name__], "TFRECORD_DIR", calibration_short_root)
    with pytest.raises(
        ValueError, match="calibration S1 requires 6 unique PNGs; found 1"
    ):
        _select_parity_inputs("calibration", repeat=1)

    sparse_root = tmp_path / "sparse-tfrecords"
    sparse_root.mkdir()
    _write_synthetic_split(
        sparse_root,
        "train",
        [
            ("train-doc-0", [(b"imageElement", b"same-png")]),
            ("train-doc-1", [(b"imageElement", b"same-png")]),
        ],
    )
    monkeypatch.setattr(sys.modules[__name__], "TFRECORD_DIR", sparse_root)
    wrapped = _select_one_per_element_type("train", occurrence=1)
    assert len(wrapped) == 1
    assert wrapped[0].example.element_type == b"imageElement"
    assert wrapped[0].unique_png_index == 0
    assert wrapped[0].unique_png_count == 1
    assert wrapped[0].wrapped


def test_plan_mode_writes_all_input_selections_without_model_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_archive(tmp_path, monkeypatch)
    monkeypatch.setenv("PIXELVAE_PARITY_MODE", "plan")
    monkeypatch.setenv("PARITY_REQUIRE", "1")

    _run_pixel_vae_cpu_stages(tmp_path / "model-output")

    output_dir = tmp_path / "parity"
    expected = [
        output_dir / "diagnostic" / "inputs.json",
        output_dir / "calibration" / "repeat-1.inputs.json",
        output_dir / "calibration" / "repeat-2.inputs.json",
        output_dir / "calibration" / "repeat-3.inputs.json",
        output_dir / "heldout.inputs.json",
        output_dir / "input-selection-plan.json",
    ]
    assert all(path.is_file() for path in expected)
    assert not (tmp_path / "model-output" / "full-model").exists()
    assert not (output_dir / "calibration" / "repeat-1.json").exists()
    assert (
        json.loads(expected[-1].read_text(encoding="utf-8"))["selection_errors"] == []
    )
    repeat_1 = json.loads(expected[1].read_text(encoding="utf-8"))
    repeat_2 = json.loads(expected[2].read_text(encoding="utf-8"))
    repeat_3 = json.loads(expected[3].read_text(encoding="utf-8"))
    text_selections = [
        next(
            item
            for item in repeat["s4_type_selections"]
            if item["element_type"] == "textElement"
        )
        for repeat in (repeat_1, repeat_2, repeat_3)
    ]
    assert [item["unique_png_count"] for item in text_selections] == [1, 1, 1]
    assert [item["unique_png_index"] for item in text_selections] == [0, 0, 0]
    assert [item["wrapped"] for item in text_selections] == [False, True, True]
    assert len({item["image_id"] for item in text_selections}) == 1


def _install_synthetic_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record_dir = tmp_path / "tfrecords"
    record_dir.mkdir()
    for split, count in (("train", 7), ("test", 6)):
        _write_synthetic_split(record_dir, split, _synthetic_documents(split, count))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "dimensions_by_element_type": {
                    element_type.decode("utf-8"): [256, 256]
                    for element_type in (
                        b"coloredBackground",
                        b"imageElement",
                        b"maskElement",
                        b"svgElement",
                        b"textElement",
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "TFRECORD_DIR", record_dir)
    monkeypatch.setattr(sys.modules[__name__], "AUDIT_PATH", audit_path)
    monkeypatch.setattr(sys.modules[__name__], "PARITY_DIR", tmp_path / "parity")


def _record_string_ids(record: Mapping[str, JSONValue], key: str) -> list[str]:
    value = record[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"selection record {key!r} is not a string list")
    return [item for item in value if isinstance(item, str)]


def _synthetic_documents(
    split: str, count: int
) -> list[tuple[str, list[tuple[bytes, bytes]]]]:
    element_types = (
        b"coloredBackground",
        b"imageElement",
        b"maskElement",
        b"svgElement",
        b"textElement",
    )
    documents = []
    for document_index in range(count):
        document_id = f"{split}-doc-{document_index:03d}"
        elements = []
        for element_type in element_types:
            source_document = document_index
            if split == "train" and element_type == b"textElement":
                source_document = 0
            elif (
                split == "train"
                and document_index == 1
                and element_type == b"imageElement"
            ):
                source_document = 0
            image_bytes = (
                f"{split}/{source_document}/{element_type.decode('utf-8')}".encode()
            )
            elements.append((element_type, image_bytes))
        documents.append((document_id, elements))
    return documents


def _write_synthetic_split(
    root: Path,
    split: str,
    documents: list[tuple[str, list[tuple[bytes, bytes]]]],
) -> None:
    import tensorflow as tf

    path = root / f"{split}-00000.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        for document_id, elements in documents:
            sequence = tf.train.SequenceExample()
            sequence.context.feature["id"].bytes_list.value.append(
                document_id.encode("utf-8")
            )
            for element_type, image_bytes in elements:
                sequence.feature_lists.feature_list[
                    "type"
                ].feature.add().bytes_list.value.append(element_type)
                sequence.feature_lists.feature_list[
                    "image_bytes"
                ].feature.add().bytes_list.value.append(image_bytes)
            writer.write(sequence.SerializeToString())


def _tensorflow_rgba(examples: list[ImageExample]) -> tf.Tensor:
    import tensorflow as tf

    decoded = [
        tf.io.decode_png(example.image_bytes, channels=4) for example in examples
    ]
    return tf.stack(decoded)


def _tensorflow_train_step(
    reference: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    epsilon: np.ndarray,
) -> tuple[tf.Tensor, list[tf.Tensor], tf.Tensor]:
    import tensorflow as tf

    original_normal = tf.random.normal
    original_head_call = reference.encoder.head.call
    source_means: list[tf.Tensor] = []

    def record_head(inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = original_head_call(inputs)
        source_means.append(result[0])
        return result

    def fixed_normal(
        shape: tf.TensorShape | tuple[int, ...],
        mean: float = 0.0,
        stddev: float = 1.0,
        dtype: tf.dtypes.DType = tf.float32,
        seed: int | None = None,
        name: str | None = None,
    ) -> tf.Tensor:
        del shape, mean, stddev, seed, name
        return tf.convert_to_tensor(epsilon, dtype=dtype)

    tf.random.normal = fixed_normal
    reference.encoder.head.call = record_head
    try:
        with tf.GradientTape() as tape:
            reference(images, training=True)
            loss = tf.add_n(reference.losses)
        gradients = tape.gradient(loss, reference.trainable_variables)
    finally:
        tf.random.normal = original_normal
        reference.encoder.head.call = original_head_call
    if any(gradient is None for gradient in gradients):
        raise AssertionError("TensorFlow produced a missing trainable gradient")
    if len(source_means) != 1:
        raise AssertionError(
            "TensorFlow training call did not expose one posterior mean"
        )
    return loss, gradients, source_means[0]


def _source_gradient_arrays(
    reference: tf.keras.Model,
    variable_map: dict[str, tf.Variable],
    gradients: list[tf.Tensor],
) -> dict[str, np.ndarray]:
    key_by_identity = {id(variable): key for key, variable in variable_map.items()}
    output = {}
    for variable, gradient in zip(
        reference.trainable_variables, gradients, strict=True
    ):
        key = key_by_identity.get(id(variable))
        if key is None:
            raise AssertionError(
                f"trainable variable has no converter key: {variable.name}"
            )
        if gradient is None:
            raise AssertionError(f"trainable variable has no gradient: {key}")
        output[key] = np.asarray(gradient.numpy())
    return output


def _compare_optimizer_slots(
    source_optimizer: tf.keras.optimizers.Optimizer,
    package_optimizer: torch.optim.Optimizer,
    reference: tf.keras.Model,
    variable_map: dict[str, tf.Variable],
    model: PixelVAEModel,
    *,
    slot_name: str,
    package_slot: str,
) -> float:
    variables = reference.trainable_variables
    source_slots = getattr(source_optimizer, slot_name)
    if len(source_slots) != len(variables):
        raise AssertionError(f"TensorFlow optimizer slot count differs for {slot_name}")
    key_by_identity = {id(variable): key for key, variable in variable_map.items()}
    arrays = {
        key_by_identity[id(variable)]: np.asarray(slot.numpy())
        for variable, slot in zip(variables, source_slots, strict=True)
    }
    mapped, _ = tensorflow_arrays_to_torch(arrays, model, strict=False)
    parameters = dict(model.named_parameters())
    differences = []
    for name, tensor in mapped.items():
        if name not in parameters:
            continue
        slot = package_optimizer.state[parameters[name]][package_slot]
        differences.append(
            max_absolute_difference(tensor.numpy(), slot.detach().cpu().numpy())
        )
    if not differences:
        raise AssertionError(f"no optimizer slots were compared for {slot_name}")
    return max(differences)


def _compare_trainable_parameters(
    reference: tf.keras.Model, model: PixelVAEModel
) -> float:
    source_weights = extract_tensorflow_weights(reference)
    mapped, _ = tensorflow_arrays_to_torch(source_weights, model, strict=False)
    package_parameters = dict(model.named_parameters())
    differences = [
        max_absolute_difference(
            value.numpy(), package_parameters[name].detach().cpu().numpy()
        )
        for name, value in mapped.items()
        if name in package_parameters
    ]
    if len(differences) != len(package_parameters):
        raise AssertionError("not all trainable parameters were compared")
    return max(differences)


def _compare_batch_norm_state(reference: tf.keras.Model, model: PixelVAEModel) -> float:
    source_weights = extract_tensorflow_weights(reference)
    mapped, _ = tensorflow_arrays_to_torch(source_weights, model, strict=False)
    state = model.state_dict()
    differences = [
        max_absolute_difference(value.numpy(), state[name].detach().cpu().numpy())
        for name, value in mapped.items()
        if name.endswith(("running_mean", "running_var"))
    ]
    if len(differences) != 116:
        raise AssertionError(
            f"expected 116 mapped BatchNorm statistics, got {len(differences)}"
        )
    return max(differences)
