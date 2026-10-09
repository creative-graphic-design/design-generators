from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from pixel_vae.testing import assert_within_limits, max_absolute_difference
from laygen.common.vendor import vendor_root

pytestmark = pytest.mark.vendor_parity

SOURCE_BYTES = 2_989_732_284
SOURCE_SHA256 = "f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e"
PARITY_DIR = Path(os.environ.get("PIXELVAE_PARITY_DIR", ".cache/pixel-vae/parity"))
TFRECORD_DIR = Path(
    os.environ.get("CRELLO_V1_TFRECORD_DIR", ".cache/pixel-vae/crello-v1-tfrecords")
)
AUDIT_PATH = Path(
    os.environ.get("CRELLO_V1_AUDIT_PATH", ".cache/pixel-vae/crello-v1-dimensions.json")
)
TRAINING_TYPES = {b"imageElement", b"maskElement", b"svgElement"}
EXPECTED_DOCUMENTS = {"train": 18_768, "val": 2_315, "test": 2_278}


@dataclass(frozen=True)
class ImageExample:
    image_id: str
    document_id: str
    element_index: int
    element_type: bytes
    image_bytes: bytes


def test_pixel_vae_cpu_stages(tmp_path: Path) -> None:
    """Compare static state, forward, updates, traces, and source data on CPU."""
    mode = os.environ.get("PIXELVAE_PARITY_MODE", "heldout")
    repeat = int(os.environ.get("PIXELVAE_PARITY_REPEAT", "0"))
    required = os.environ.get("PARITY_REQUIRE") == "1"
    if mode not in {"calibration", "heldout"}:
        raise ValueError("PIXELVAE_PARITY_MODE must be calibration or heldout")
    if mode == "calibration" and repeat not in {1, 2, 3}:
        raise ValueError("PIXELVAE_PARITY_REPEAT must be 1, 2, or 3")

    missing = [path for path in (TFRECORD_DIR, AUDIT_PATH) if not path.exists()]
    if missing:
        message = f"private Crello v1 parity assets are missing: {missing}"
        if required:
            raise FileNotFoundError(message)
        pytest.skip(message)

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

    train_document_images = _select_images("train", 6, filter_training_types=False)
    train_training_images = _select_images("train", 6, filter_training_types=True)
    test_document_images = _select_images("test", 6, filter_training_types=False)
    test_training_images = _select_images("test", 6, filter_training_types=True)
    calibration = mode == "calibration"
    document_images = train_document_images if calibration else test_document_images
    training_images = train_training_images if calibration else test_training_images
    if min(map(len, (document_images, training_images))) < 6:
        raise AssertionError(
            "canonical v1 source stream did not provide six unique images"
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
    s1_examples = document_images[:2]
    tensorflow_images = _tensorflow_rgba(s1_examples)
    processor = PixelVAEImageProcessor()
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
    metrics = {
        "s1.posterior_mean.max_abs": max_absolute_difference(
            source_means.numpy(), output.posterior_mean.numpy()
        ),
        "s1.posterior_log_variance.max_abs": max_absolute_difference(
            source_log_variances.numpy(), output.posterior_log_variance.numpy()
        ),
        "s1.decoder_logits.max_abs": max_absolute_difference(
            source_logits.numpy(), output.logits.numpy()
        ),
        "s1.reconstruction_loss.abs": max_absolute_difference(
            np.asarray(source_reconstruction_loss.numpy()),
            np.asarray(output.reconstruction_loss.numpy()),
        ),
        "s1.kl_loss.abs": max_absolute_difference(
            np.asarray(source_kl_loss.numpy()), np.asarray(output.kl_divergence.numpy())
        ),
        "s1.total_loss.abs": max_absolute_difference(
            np.asarray(source_total_loss.numpy()), np.asarray(output.loss.numpy())
        ),
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
    type_examples = _select_one_per_element_type("test")
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
        "source_commit": "bc1e2072ba3a253f1b099e8b0c604f6051e787da",
        "encoder_state_sha256": state_digest,
        "configuration": model.config.to_dict(),
        "s0_assigned_tensors": conversion.assigned_tensors,
        "s1_image_ids": [example.image_id for example in s1_examples],
        "s2_image_ids": [example.image_id for example in training_images[:2]],
        "s3_image_ids": [example.image_id for example in training_images[:6]],
        "s4_element_types": [
            example.element_type.decode("utf-8") for example in type_examples
        ],
        "s4_image_ids": [example.image_id for example in type_examples],
        "s4_document_counts": audit["document_counts"],
        "s4_element_counts": audit["element_counts"],
        "s4_dimensions_by_element_type": audit["dimensions_by_element_type"],
        "s4_embedding_shape": list(original_embeddings.shape),
        "s4_embedding_dtype": str(original_embeddings.dtype),
        "metrics": metrics,
    }
    output_path = (
        PARITY_DIR / "calibration" / f"repeat-{repeat}.json"
        if calibration
        else PARITY_DIR / "heldout.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        assert_within_limits(metrics, limits)
        result["calibrated_limits"] = limits
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def _select_one_per_element_type(split: str) -> list[ImageExample]:
    import tensorflow as tf

    examples: dict[bytes, ImageExample] = {}
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
            if kind not in examples:
                png_bytes = image.bytes_list.value[0]
                image_id = f"crello-v1/{split}/{hashlib.sha256(png_bytes).hexdigest()}"
                examples[kind] = ImageExample(
                    image_id, document_id, index, kind, png_bytes
                )

    return list(examples.values())


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
