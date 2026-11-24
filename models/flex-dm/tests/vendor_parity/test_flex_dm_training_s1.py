"""Compare the fixed-batch Flex-DM training forward and objective live."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np
import pytest
import torch
from flex_dm import FlexDmForMaskedDocumentModeling
from traingen_parity.compare import TensorTolerance, compare_step_trace
from traingen_parity.trace import build_step_trace

from .test_flex_dm_training_s0 import (
    DatasetName,
    _DATASETS,
    _PINNED_VENDOR,
    _assert_numerical_reference_mode,
    _assert_reference_device,
    _first_batch,
    _source_model_state,
    _to_torch_inputs,
)
from .training_reference import build_original, cache_root, original_args

_OUTPUT_MAX_REL_TO_MAX = 1e-5
_LOSS_MAX_REL_TO_MAX = 1e-6


def _capture_dropout_masks(original_model, tf) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for layer in original_model.submodules:
        if not isinstance(layer, tf.keras.layers.Dropout):
            continue

        original_call = layer.call

        def capture(inputs, training=None, *, call=original_call):
            output = call(inputs, training=training)
            if training:
                input_values = inputs.numpy()
                output_values = output.numpy()
                masks.append(np.logical_or(input_values == 0.0, output_values != 0.0))
            return output

        layer.call = capture

    return masks


def _replay_dropout_masks(package, masks: list[np.ndarray]) -> list[int]:
    draws = iter(masks)
    consumed = [0]

    def replay(values: torch.Tensor) -> torch.Tensor:
        mask = next(draws)
        assert tuple(values.shape) == mask.shape
        consumed[0] += 1
        return values * torch.as_tensor(mask, device=values.device) / 0.9

    for module in package.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0.1
            module.forward = replay

    return consumed


def _reference_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value.numpy(), device="cuda")


def _max_rel_to_max(comparison, reference: torch.Tensor) -> float:
    scale = max(float(reference.abs().max().item()), 1e-12)
    return comparison.max_abs_diff / scale


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", tuple(_DATASETS))
def test_s1_flex_dm_training_forward_and_loss_match_reference(
    dataset: DatasetName,
) -> None:
    """Compare all fine-tuning tasks on the seed-zero production-size batch."""
    import tensorflow as tf

    _assert_reference_device(tf)
    _assert_numerical_reference_mode(tf)
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    vendor_revision = subprocess.check_output(
        ["git", "-C", "vendor/flex-dm", "rev-parse", "HEAD"], text=True
    ).strip()
    assert vendor_revision == _PINNED_VENDOR

    args = original_args(dataset)
    batch_size = int(args["batch_size"])
    seed = int(args["seed"])
    assert batch_size == 256 and seed == 0
    assert float(args["l2"]) == 0.01
    assert float(args["dropout"]) == 0.1
    dataspec, original, _ = build_original(dataset, batch_size=batch_size)
    batch = _first_batch(dataspec, "train", batch_size=batch_size)
    assert int(batch["left"].shape[0]) == batch_size
    batch_sha256 = hashlib.sha256(
        b"".join(tf.io.serialize_tensor(batch[key]).numpy() for key in sorted(batch))
    ).hexdigest()

    package = (
        FlexDmForMaskedDocumentModeling.from_pretrained(
            cache_root() / "converted" / f"flex-dm-{dataset}"
        )
        .to("cuda")
        .train()
    )
    package.load_state_dict(_source_model_state(dataset), strict=True)
    assert package.config.dataset_name == dataset
    assert package.config.l2 == 0.01
    assert package.config.dropout == 0.1

    from mfp.models import mfp as vendor_mfp

    dropout_masks = _capture_dropout_masks(original.model, tf)
    task_names = tuple(str(args["masking_method"]).split("_"))
    results: dict[str, dict[str, float]] = {}
    dropout_shapes: dict[str, list[list[int]]] = {}
    for task_name in task_names:
        dropout_masks.clear()
        task_id = original.task_names.index(task_name)
        tf.random.set_seed(seed)
        tasks = tf.fill((batch_size,), task_id)
        targets, modified_inputs, masks = vendor_mfp.preprocess_for_train(
            batch,
            original.input_columns,
            tasks,
            is_autoreg=original.is_autoreg,
            input_dtype=original.input_dtype,
        )
        with tf.device("/GPU:0"):
            original_logits = original.model(modified_inputs, training=True)
            sort_flag = (
                tasks == original.task_names.index("pos") if original.sort_pos else None
            )
            for metric in original.loss_layer.metrics:
                metric.reset_state()
            original.loss_layer(
                (targets, original_logits, masks),
                training=False,
                sort_flag=sort_flag,
            )

        components = {
            metric.name.removesuffix("_loss"): _reference_tensor(metric.result())
            for metric in original.loss_layer.metrics
            if metric.name.endswith("_loss")
        }
        assert set(components) == set(package.config.valid_sequence_keys)
        regularizers = original.model.losses
        l2_loss = _reference_tensor(tf.add_n(regularizers))
        reference_tensors = {
            **{
                f"logits/{key}": _reference_tensor(value)
                for key, value in original_logits.items()
            },
            **{f"loss/{key}": value for key, value in components.items()},
            "l2_loss": l2_loss,
            "total_loss": torch.stack((*components.values(), l2_loss)).sum(),
        }

        package_inputs = _to_torch_inputs(
            modified_inputs, set(package.config.input_columns)
        )
        package_labels = _to_torch_inputs(targets, set(package.config.input_columns))
        package_masks = _to_torch_inputs(masks, set(package.config.input_columns))
        dropout_draws_consumed = _replay_dropout_masks(package, dropout_masks)
        with torch.inference_mode():
            package_output = package(
                inputs=package_inputs,
                labels=package_labels,
                masks=package_masks,
                task_ids=torch.full((batch_size,), task_id, device="cuda"),
            )
        assert dropout_draws_consumed[0] == len(dropout_masks) > 0
        dropout_shapes[task_name] = [list(mask.shape) for mask in dropout_masks]
        assert package_output.loss_components is not None
        assert package_output.l2_loss is not None and package_output.loss is not None
        package_tensors = {
            **{f"logits/{key}": value for key, value in package_output.logits.items()},
            **{
                f"loss/{key}": value
                for key, value in package_output.loss_components.items()
            },
            "l2_loss": package_output.l2_loss,
            "total_loss": package_output.loss,
        }
        assert set(package_tensors) == set(reference_tensors)

        output_names = tuple(
            name for name in reference_tensors if name.startswith("logits/")
        )
        tolerances = {
            name: TensorTolerance(
                atol=float(reference_tensors[name].abs().max().item())
                * (
                    _OUTPUT_MAX_REL_TO_MAX
                    if name in output_names
                    else _LOSS_MAX_REL_TO_MAX
                ),
                rtol=0.0,
            )
            for name in reference_tensors
        }
        report = compare_step_trace(
            build_step_trace("original", reference_tensors),
            build_step_trace("package", package_tensors),
            tolerances,
        )
        assert report.passed, [
            {
                "name": item.name,
                "max_abs_diff": item.max_abs_diff,
                "max_rel_diff": item.max_rel_diff,
                "message": item.message,
            }
            for item in report.comparisons
            if not item.passed
        ]
        output_differences = [
            _max_rel_to_max(item, reference_tensors[item.name])
            for item in report.comparisons
            if item.name in output_names
        ]
        loss_differences = [
            _max_rel_to_max(item, reference_tensors[item.name])
            for item in report.comparisons
            if item.name not in output_names
        ]
        assert max(output_differences, default=0.0) <= _OUTPUT_MAX_REL_TO_MAX
        assert max(loss_differences, default=0.0) <= _LOSS_MAX_REL_TO_MAX
        results[task_name] = {
            "output_max_rel_to_max": max(output_differences, default=0.0),
            "loss_max_rel_to_max": max(loss_differences, default=0.0),
        }

    print(
        json.dumps(
            {
                "dataset": dataset,
                "starting_checkpoint": "released Ours-EXP-FT best.ckpt loaded identically in both systems",
                "batch_size": batch_size,
                "train_stream": "first sorted train shard, source order",
                "batch_sha256": batch_sha256,
                "seed": seed,
                "tf32": "disabled",
                "dropout_forward_mode": "training; original masks replayed with configured probability 0.1",
                "dropout_mask_shapes": dropout_shapes,
                "tasks": results,
            },
            sort_keys=True,
        )
    )
