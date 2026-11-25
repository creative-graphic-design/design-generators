"""Execute the pinned Flex-DM training path and compare live behavior."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from flex_dm import FlexDmForMaskedDocumentModeling
from flex_dm.training import masking
from flex_dm.conversion import conversion_report, map_tensor_by_rule
from flex_dm.data_specs import attribute_groups_for_dataset, build_column_specs
from flex_dm.tf_checkpoint import load_tf_checkpoint_variables

from .training_reference import (
    _REPO_ROOT,
    DatasetName,
    build_original,
    cache_root,
    original_args,
    original_asset_root,
    original_variant_root,
)

_PINNED_VENDOR = "f2bcc9f8b8958d4072f11f259bc61242bfe99d7c"
_DATASETS: dict[DatasetName, dict[str, int | float]] = {
    "crello": {"parameters": 2_812_257, "tensors": 98, "atol": 1.1e-5},
    "rico": {"parameters": 2_296_679, "tensors": 88, "atol": 5.5e-6},
}
_METHODS = {"crello": "elem_pos_attr_img_txt", "rico": "elem_pos_attr"}


def _assert_reference_device(tf) -> None:
    assert tf.__version__ == "2.15.1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    assert len(tf.config.list_physical_devices("GPU")) == 1
    assert tf.config.list_logical_devices("GPU")


def _source_model_state(dataset: DatasetName) -> dict[str, torch.Tensor]:
    args = original_args(dataset)
    checkpoint = original_variant_root(dataset) / "checkpoints" / "best.ckpt"
    variables = load_tf_checkpoint_variables(checkpoint)
    source_keys = {
        key
        for key in variables
        if key.startswith("model/") and "/.OPTIMIZER_SLOT/" not in key
    }
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for source_name in sorted(source_keys):
        mapped = map_tensor_by_rule(source_name, variables[source_name])
        if mapped is not None:
            target_name, tensor = mapped
            if target_name in converted:
                raise AssertionError(f"duplicate source mapping for {target_name}")
            converted[target_name] = tensor
            consumed.add(source_name)
    report = conversion_report(
        converted=converted,
        target_keys=set(converted),
        source_keys=source_keys,
        consumed_source_keys=consumed,
    )
    assert report.unexpected_source_keys == ()
    assert len(converted) == _DATASETS[dataset]["tensors"]
    assert report.matched_parameter_count == _DATASETS[dataset]["parameters"]
    assert args["arch_type"] == "oneshot"
    return converted


def _first_batch(dataspec, split: str, *, batch_size: int = 1):
    import tensorflow as tf

    shards = sorted(
        tf.io.gfile.glob(os.path.join(dataspec._path, f"{split}-*.tfrecord"))
    )
    assert shards
    records = tf.data.TFRecordDataset(shards[0]).batch(batch_size)
    return next(iter(records.map(dataspec.parse_fn)))


def _to_torch_inputs(batch, keys: set[str]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(batch[key].numpy()).to("cuda") for key in keys}


def _assert_numerical_reference_mode(tf) -> None:
    tf.config.experimental.enable_tensor_float_32_execution(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not tf.config.experimental.tensor_float_32_execution_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def _trace_shape(shape, tf) -> tuple[int, ...]:
    if hasattr(shape, "numpy"):
        shape = shape.numpy()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(value) for value in tf.TensorShape(shape).as_list())


def _trace_tensor(operation: str, value, *, requested_shape, tf) -> dict[str, object]:
    array = value.numpy()
    return {
        "operation": operation,
        "requested_shape": list(_trace_shape(requested_shape, tf)),
        "result_shape": list(array.shape),
        "dtype": str(array.dtype),
        "device": value.device,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _write_trace(name: str, value: dict[str, object]) -> None:
    output_dir = os.environ.get("FLEX_DM_S0_TRACE_DIR")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )


def _vendor_runtime():
    from mfp.data.spec import get_attribute_groups
    from mfp.models import mfp as vendor_mfp

    return vendor_mfp, get_attribute_groups


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", tuple(_DATASETS))
def test_s0_flex_dm_training_topology_matches_reference(dataset: DatasetName) -> None:
    """Live vendor and package models map every parameter and agree on one GPU input."""
    import tensorflow as tf

    _assert_reference_device(tf)
    _assert_numerical_reference_mode(tf)
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    assert not os.environ.get("TF_DETERMINISTIC_OPS")
    vendor_revision = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT / "vendor" / "flex-dm"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert vendor_revision == _PINNED_VENDOR

    dataspec, original, _args = build_original(dataset, batch_size=1)
    package_dir = cache_root() / "converted" / f"flex-dm-{dataset}"
    package = FlexDmForMaskedDocumentModeling.from_pretrained(package_dir)
    state = _source_model_state(dataset)
    report = package.load_state_dict(state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    assert len(tuple(package.named_parameters())) == _DATASETS[dataset]["tensors"]
    assert (
        sum(parameter.numel() for parameter in package.parameters())
        == _DATASETS[dataset]["parameters"]
    )
    assert package.config.dataset_name == dataset

    batch = _first_batch(dataspec, "test")
    reference_inputs = {
        key: batch[key] for key in original.input_columns if key in batch
    }
    package_inputs = _to_torch_inputs(batch, set(package.config.input_columns))
    with tf.device("/GPU:0"):
        reference = original.model(reference_inputs, training=False)
    package = package.to("cuda").eval()
    with torch.inference_mode():
        actual = package(inputs=package_inputs).logits
    assert int(original.model.count_params()) == _DATASETS[dataset]["parameters"]
    max_abs = max(
        float(
            np.max(np.abs(reference[key].numpy() - actual[key].detach().cpu().numpy()))
        )
        for key in actual
    )
    assert max_abs <= _DATASETS[dataset]["atol"]


@pytest.mark.vendor_parity
def test_s0_original_training_branches_match_resolved_args(monkeypatch) -> None:
    """The original GPU preparation path records its executed branches and every RNG draw."""
    import tensorflow as tf

    _assert_reference_device(tf)
    assert tf.config.experimental.tensor_float_32_execution_enabled()
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") is None
    assert os.environ.get("TF_DETERMINISTIC_OPS") is None
    deterministic_enable_calls = 0
    raw_enable_determinism = tf.config.experimental.enable_op_determinism

    def trace_enable_determinism():
        nonlocal deterministic_enable_calls
        deterministic_enable_calls += 1
        return raw_enable_determinism()

    monkeypatch.setattr(
        tf.config.experimental,
        "enable_op_determinism",
        trace_enable_determinism,
    )
    tf.debugging.set_log_device_placement(True)
    profile_dir = os.environ.get("FLEX_DM_S0_PROFILE_DIR")
    if profile_dir:
        tf.profiler.experimental.start(profile_dir)
    policy = tf.keras.mixed_precision.global_policy()
    assert policy.compute_dtype == policy.variable_dtype == "float32"

    traces = []
    raw_uniform = tf.random.uniform
    raw_normal = tf.random.normal

    def traced_uniform(shape, *args, **kwargs):
        value = raw_uniform(shape, *args, **kwargs)
        traces.append(
            _trace_tensor("tf.random.uniform", value, requested_shape=shape, tf=tf)
        )
        return value

    def traced_normal(shape, *args, **kwargs):
        value = raw_normal(shape, *args, **kwargs)
        traces.append(
            _trace_tensor("tf.random.normal", value, requested_shape=shape, tf=tf)
        )
        return value

    raw_categorical = tf.random.categorical

    def traced_categorical(logits, num_samples, *args, **kwargs):
        value = raw_categorical(logits, num_samples, *args, **kwargs)
        traces.append(
            _trace_tensor(
                "tf.random.categorical",
                value,
                requested_shape=value.shape,
                tf=tf,
            )
        )
        return value

    monkeypatch.setattr(tf.random, "uniform", traced_uniform)
    monkeypatch.setattr(tf.random, "normal", traced_normal)
    monkeypatch.setattr(tf.random, "categorical", traced_categorical)
    vendor_mfp = None
    get_attribute_groups = None
    raw_random = raw_elem = raw_feat = None
    prep_calls = []
    token_calls = []
    candidate_records = []
    loss_sort_calls = []
    loss_flags = []
    column_names = {}
    shuffle_inputs = sort_inputs = None
    raw_apply_token = None
    vendor_masking = None

    for dataset in _DATASETS:
        exp_args = original_args(dataset)
        assert exp_args["dataset_name"] == dataset
        assert exp_args["num_epochs"] == 500
        assert exp_args["batch_size"] == 256
        assert exp_args["arch_type"] == "oneshot"
        assert exp_args["seq_type"] == "default"
        assert exp_args["input_dtype"] == "set"
        assert exp_args["masking_method"] == _METHODS[dataset]
        assert (
            exp_args["weights"]
            == {
                "crello": "tmp/jobs/crello/20230324084601-random/checkpoints/best.ckpt",
                "rico": "tmp/jobs/rico/20230327062033-random/checkpoints/best.ckpt",
            }[dataset]
        )

        dataspec, _exp_model, _ = build_original(
            dataset, batch_size=int(exp_args["batch_size"]), restore_checkpoint=False
        )
        if vendor_mfp is None:
            vendor_mfp, get_attribute_groups = _vendor_runtime()
            raw_random = vendor_mfp.random_masking
            raw_elem = vendor_mfp.elem_masking
            raw_feat = vendor_mfp.feat_masking
            shuffle_inputs = vendor_mfp.shuffle_inputs
            sort_inputs = vendor_mfp.sort_inputs
            from mfp.models import masking as vendor_masking

            raw_apply_token = vendor_masking.apply_token

            def traced_shuffle(*args, _call=shuffle_inputs, **kwargs):
                prep_calls.append("shuffle_inputs")
                return _call(*args, **kwargs)

            def traced_sort(*args, _call=sort_inputs, **kwargs):
                prep_calls.append("sort_inputs")
                return _call(*args, **kwargs)

            def traced_apply_token(
                input_, column, mask, token_type, _apply=raw_apply_token
            ):
                token_calls.append(
                    {
                        "column": column_names[id(column)],
                        "token_type": token_type,
                        "input_shape": list(input_.shape),
                        "mask_shape": list(mask.shape),
                    }
                )
                return _apply(input_, column, mask, token_type)

            monkeypatch.setattr(vendor_mfp, "shuffle_inputs", traced_shuffle)
            monkeypatch.setattr(vendor_mfp, "sort_inputs", traced_sort)
            monkeypatch.setattr(vendor_masking, "apply_token", traced_apply_token)
        batch = _first_batch(dataspec, "train", batch_size=int(exp_args["batch_size"]))
        assert int(batch["left"].shape[0]) == exp_args["batch_size"] == 256
        assert all(value.device.endswith("/device:CPU:0") for value in batch.values())
        input_columns = _exp_model.input_columns
        expected_groups = get_attribute_groups(input_columns.keys())
        branch_flags = {
            "is_autoreg": _exp_model.is_autoreg,
            "input_dtype": _exp_model.input_dtype,
            "sort_pos": _exp_model.sort_pos,
            "dataset": dataset,
            "train_batch_size": int(batch["left"].shape[0]),
            "configured_batch_size": exp_args["batch_size"],
            "train_stream_selection": "first sorted train shard, source order",
            "train_stream_batch_shapes": {
                key: list(value.shape) for key, value in batch.items()
            },
            "training_rng_seed": int(exp_args.get("seed", 0)),
            "train_stream_batch_sha256": hashlib.sha256(
                b"".join(
                    tf.io.serialize_tensor(batch[key]).numpy() for key in sorted(batch)
                )
            ).hexdigest(),
            "tf32_matmul_enabled_by_default": tf.config.experimental.tensor_float_32_execution_enabled(),
            "deterministic_ops_env": os.environ.get("TF_DETERMINISTIC_OPS"),
            "device": "/GPU:0",
        }
        assert branch_flags["is_autoreg"] is False
        assert branch_flags["input_dtype"] == "set"
        assert branch_flags["sort_pos"] is (dataset == "rico")
        last_trace = None

        def record_candidate(name, result, records=candidate_records):
            _modified, masks = result
            records.append(
                {
                    "candidate": name,
                    "keys": sorted(masks),
                    "mask_shapes": {
                        key: list(value.shape) for key, value in masks.items()
                    },
                    "mask_sha256": {
                        key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                        for key, value in masks.items()
                    },
                }
            )
            return result

        def traced_random(*args, _call=raw_random, _record=record_candidate, **kwargs):
            return _record("random", _call(*args, **kwargs))

        def traced_elem(*args, _call=raw_elem, _record=record_candidate, **kwargs):
            return _record("elem", _call(*args, **kwargs))

        def traced_feat(*args, _call=raw_feat, _record=record_candidate, **kwargs):
            group = tuple(args[3])
            return _record(f"feature:{','.join(group)}", _call(*args, **kwargs))

        monkeypatch.setattr(vendor_mfp, "random_masking", traced_random)
        monkeypatch.setattr(vendor_mfp, "elem_masking", traced_elem)
        monkeypatch.setattr(vendor_mfp, "feat_masking", traced_feat)
        for variant, method in (
            ("ours-imp", "random"),
            ("ours-exp-ft", _METHODS[dataset]),
        ):
            traces.clear()
            candidate_records.clear()
            token_calls.clear()
            prep_calls.clear()
            tf.random.set_seed(int(exp_args.get("seed", 0)))
            dataspec, model, _ = build_original(
                dataset,
                masking_method=method,
                batch_size=int(exp_args["batch_size"]),
                restore_checkpoint=False,
            )
            del dataspec
            column_names.clear()
            column_names.update(
                {id(value): key for key, value in model.input_columns.items()}
            )
            task_event = {
                "operation": "tfp.Categorical.sample",
                "requested_shape": [int(batch["left"].shape[0])],
                "task_names": list(model.task_names),
                "task_probabilities": model.task_cat_dist.probs_parameter()
                .numpy()
                .tolist(),
            }
            traces.append(task_event)
            tasks = model.task_cat_dist.sample(tf.shape(batch["left"])[0])
            task_values = tasks.numpy()
            task_event.update(
                {
                    "result_shape": list(tasks.shape),
                    "selected_tasks": task_values.tolist(),
                    "dtype": str(task_values.dtype),
                    "device": tasks.device,
                    "value_sha256": hashlib.sha256(task_values.tobytes()).hexdigest(),
                }
            )
            task_id = int(tasks[0].numpy())
            allowed = {"random"} if variant == "ours-imp" else set(method.split("_"))
            selected_names = {
                name
                for name in model.task_names
                if name == "random" and "random" in allowed or name in allowed
            }
            expected_probabilities = np.asarray(
                [name in selected_names for name in model.task_names], dtype=np.float32
            )
            expected_probabilities /= expected_probabilities.sum()
            np.testing.assert_allclose(
                task_event["task_probabilities"], expected_probabilities
            )
            assert model.task_names[task_id] in selected_names
            ordered_inputs, modified_inputs, masks = vendor_mfp.preprocess_for_train(
                batch,
                model.input_columns,
                tasks,
                is_autoreg=model.is_autoreg,
                input_dtype=model.input_dtype,
            )
            prepared_input_devices = {
                key: value.device for key, value in ordered_inputs.items()
            }
            assert all(
                device.endswith("/device:CPU:0")
                for device in prepared_input_devices.values()
            )
            assert all(
                value.device.endswith("/device:GPU:0")
                for value in modified_inputs.values()
            )
            assert all(
                value.device.endswith("/device:GPU:0") for value in masks.values()
            )
            assert all(draw["device"].endswith("/device:GPU:0") for draw in traces)
            assert prep_calls == []
            assert token_calls and {call["token_type"] for call in token_calls} >= {
                "unused",
                "masked",
                "random",
            }
            outputs = model.model(modified_inputs, training=True)
            expected_candidates = ["random", "elem"] + [
                f"feature:{','.join(group)}" for group in expected_groups.values()
            ]
            assert [
                item["candidate"] for item in candidate_records
            ] == expected_candidates
            branch_flags["selected_task"] = model.task_names[task_id]
            branch_flags["allowed_tasks"] = sorted(selected_names)
            branch_flags["unused_token_padding"] = any(
                call["token_type"] == "unused" for call in token_calls
            )
            branch_flags["shuffle_inputs"] = "shuffle_inputs" in prep_calls
            branch_flags["preparation_sort_inputs"] = "sort_inputs" in prep_calls
            branch_flags["candidate_order"] = expected_candidates
            branch_flags["selected_candidate"] = model.task_names[task_id]
            branch_flags["loss_sort_branch"] = (
                "rico_position" if model.sort_pos else "ordinary"
            )
            branch_flags["forward_output_shapes"] = {
                key: list(value.shape) for key, value in outputs.items()
            }
            branch_flags["forward_output_devices"] = {
                key: value.device for key, value in outputs.items()
            }
            assert all("GPU:0" in value.device for value in outputs.values())
            branch_flags["tf32_enabled"] = (
                tf.config.experimental.tensor_float_32_execution_enabled()
            )
            branch_flags["mixed_precision_policy"] = policy.name
            branch_flags["deterministic_ops_api_calls"] = deterministic_enable_calls
            assert branch_flags["tf32_enabled"] is True
            assert branch_flags["mixed_precision_policy"] == "float32"
            assert branch_flags["unused_token_padding"] is True
            assert branch_flags["shuffle_inputs"] is False
            assert branch_flags["preparation_sort_inputs"] is False
            branch_flags["token_calls"] = list(token_calls)
            branch_flags["batch_devices"] = {
                key: value.device for key, value in batch.items()
            }
            branch_flags["prepared_input_devices"] = prepared_input_devices
            branch_flags["modified_input_devices"] = {
                key: value.device for key, value in modified_inputs.items()
            }
            branch_flags["mask_devices"] = {
                key: value.device for key, value in masks.items()
            }
            branch_flags["mask_shapes"] = {
                key: list(value.shape) for key, value in masks.items()
            }
            last_trace = {
                "source_revision": _PINNED_VENDOR,
                "args_variant": "ours-exp-ft",
                "recipe": variant,
                "masking_method": method,
                "branches": branch_flags.copy(),
                "draws": list(traces),
                "candidates_before_tf_where": list(candidate_records),
            }
            _write_trace(f"branches-{dataset}-{variant}", last_trace)

        from mfp.models import metrics as vendor_metrics

        loss_sort_calls.clear()
        raw_sort = vendor_metrics.sort_inputs

        def traced_sort_inputs(*args, _calls=loss_sort_calls, _sort=raw_sort, **kwargs):
            _calls.append(True)
            return _sort(*args, **kwargs)

        loss_flags.clear()
        raw_loss_call = model.loss_layer.call

        def traced_loss_call(
            inputs,
            training=False,
            sort_flag=None,
            ignore_sort=None,
            _flags=loss_flags,
            _call=raw_loss_call,
        ):
            _flags.append(sort_flag)
            return _call(
                inputs,
                training=training,
                sort_flag=sort_flag,
                ignore_sort=ignore_sort,
            )

        monkeypatch.setattr(vendor_metrics, "sort_inputs", traced_sort_inputs)
        monkeypatch.setattr(model.loss_layer, "call", traced_loss_call)
        tf.random.set_seed(int(exp_args.get("seed", 0)))
        model(batch, training=True)
        if dataset == "rico":
            assert len(loss_sort_calls) == 2
            assert len(loss_flags) == 1 and tf.is_tensor(loss_flags[0])
        else:
            assert not loss_sort_calls
            assert loss_flags == [None]
        assert last_trace is not None
        assert deterministic_enable_calls == 0
        last_trace["branches"]["loss_sort_flag_is_tensor"] = tf.is_tensor(loss_flags[0])
        last_trace["branches"]["loss_sort_inputs_calls"] = len(loss_sort_calls)
        last_trace["branches"]["deterministic_enable_api_calls"] = (
            deterministic_enable_calls
        )
        _write_trace(f"branches-{dataset}-ours-exp-ft", last_trace)
        monkeypatch.setattr(vendor_metrics, "sort_inputs", raw_sort)
        monkeypatch.setattr(vendor_mfp, "random_masking", raw_random)
        monkeypatch.setattr(vendor_mfp, "elem_masking", raw_elem)
        monkeypatch.setattr(vendor_mfp, "feat_masking", raw_feat)
    if profile_dir:
        tf.profiler.experimental.stop()
    tf.debugging.set_log_device_placement(False)


def _shape_from_size(size) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)):
        return tuple(int(value) for value in size[0])
    return tuple(int(value) for value in size)


def _torch_inputs_from_batch(batch, keys: set[str]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(batch[key].numpy()) for key in keys}


def _assert_prepared_equal(reference, actual) -> None:
    assert set(reference) == set(actual)
    for key, value in reference.items():
        expected = torch.from_numpy(value.numpy())
        observed = actual[key].cpu()
        assert observed.shape == expected.shape
        torch.testing.assert_close(
            observed.to(expected.dtype), expected, rtol=0, atol=0
        )


@pytest.mark.vendor_parity
def test_s0_training_masking_replays_original_random_draws(monkeypatch) -> None:
    """The package port replays every original draw and candidate for both dataset schemas."""
    import tensorflow as tf

    for dataset, method in _METHODS.items():
        args = original_args(dataset)
        batch_size = int(args["batch_size"])
        assert batch_size == 256
        dataspec, original, _args = build_original(
            dataset,
            masking_method=method,
            batch_size=batch_size,
            restore_checkpoint=False,
        )
        from mfp.data.spec import get_attribute_groups
        from mfp.models import mfp as vendor_mfp

        batch = _first_batch(dataspec, "train", batch_size=batch_size)
        assert int(batch["left"].shape[0]) == batch_size
        seed = int(args.get("seed", 0))
        assert seed == 0
        task_names = original.task_names
        attribute_groups = get_attribute_groups(original.input_columns.keys())
        data_root = original_asset_root() / "data" / dataset
        if (data_root / dataset).is_dir():
            data_root /= dataset
        vocabulary = json.loads((data_root / "vocabulary.json").read_text())
        package_columns = build_column_specs(
            dataset_name=dataset, vocabulary=vocabulary
        )
        package_inputs = _torch_inputs_from_batch(batch, set(package_columns))
        package_groups = attribute_groups_for_dataset(dataset)
        assert [(key, tuple(value)) for key, value in package_groups.items()] == [
            (key, tuple(value)) for key, value in attribute_groups.items()
        ]
        tasks_to_check = ["random"] + [name for name in method.split("_")]

        for task_name in tasks_to_check:
            task_id = task_names.index(task_name)
            tasks = tf.fill([batch_size], task_id)
            original_draws: list[tuple[str, tuple[int, ...], np.ndarray]] = []
            raw_uniform = tf.random.uniform
            raw_normal = tf.random.normal

            def traced_uniform(
                shape, *args, _draws=original_draws, _uniform=raw_uniform, **kwargs
            ):
                value = _uniform(shape, *args, **kwargs)
                operation = "uniform_int" if value.dtype.is_integer else "uniform_float"
                _draws.append(
                    (operation, tuple(value.shape.as_list()), value.numpy().copy())
                )
                return value

            def traced_normal(
                shape, *args, _draws=original_draws, _normal=raw_normal, **kwargs
            ):
                value = _normal(shape, *args, **kwargs)
                _draws.append(
                    ("normal", tuple(value.shape.as_list()), value.numpy().copy())
                )
                return value

            monkeypatch.setattr(tf.random, "uniform", traced_uniform)
            monkeypatch.setattr(tf.random, "normal", traced_normal)
            tf.random.set_seed(seed)
            reference = vendor_mfp.preprocess_for_train(
                batch,
                original.input_columns,
                tasks,
                is_autoreg=original.is_autoreg,
                input_dtype=original.input_dtype,
            )
            monkeypatch.setattr(tf.random, "uniform", raw_uniform)
            monkeypatch.setattr(tf.random, "normal", raw_normal)

            package_calls: list[tuple[str, tuple[int, ...]]] = []
            draw_index = 0

            def replay(
                operation,
                size,
                *,
                device,
                dtype,
                _draws=original_draws,
                _calls=package_calls,
            ):
                nonlocal draw_index
                shape = _shape_from_size(size)
                expected_operation, expected_shape, values = _draws[draw_index]
                assert (operation, shape) == (expected_operation, expected_shape)
                draw_index += 1
                _calls.append((operation, shape))
                if operation == "normal":
                    # TensorFlow records stddev=0.1 output; the package scales randn itself.
                    values = values / 0.1
                return torch.as_tensor(values, device=device, dtype=dtype)

            def replay_rand(*size, device=None, dtype=None, generator=None):
                return replay("uniform_float", size, device=device, dtype=dtype)

            def replay_randint(
                low, high, size, *, device=None, dtype=None, generator=None
            ):
                return replay("uniform_int", (size,), device=device, dtype=dtype)

            def replay_randn(*size, device=None, dtype=None, generator=None):
                return replay("normal", size, device=device, dtype=dtype)

            with monkeypatch.context() as patch:
                patch.setattr(masking, "rand", replay_rand)
                patch.setattr(masking, "randint", replay_randint)
                patch.setattr(masking, "randn", replay_randn)
                actual = masking.prepare_training_batch(
                    package_inputs,
                    package_columns,
                    torch.full((batch_size,), task_id),
                    attribute_groups=package_groups,
                    is_autoreg=original.is_autoreg,
                    input_dtype=original.input_dtype,
                )

            assert draw_index == len(original_draws)
            assert package_calls == [
                (operation, shape) for operation, shape, _values in original_draws
            ]
            _assert_prepared_equal(
                {key: value for key, value in reference[0].items() if key in actual[0]},
                actual[0],
            )
            _assert_prepared_equal(reference[1], actual[1])
            _assert_prepared_equal(reference[2], actual[2])
            output_dir = os.environ.get("FLEX_DM_S0_TRACE_DIR")
            if output_dir:
                root = Path(output_dir)
                root.mkdir(parents=True, exist_ok=True)
                trace_name = f"replay-{dataset}-{task_name}"
                np.savez_compressed(
                    root / f"{trace_name}.npz",
                    **{
                        f"draw{index:03d}": item[2]
                        for index, item in enumerate(original_draws)
                    },
                )
                (root / f"{trace_name}.json").write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "task": task_name,
                            "batch_size": batch_size,
                            "train_stream_selection": "first sorted train shard, source order",
                            "seed": seed,
                            "draws": [
                                {"operation": item[0], "shape": list(item[1])}
                                for item in original_draws
                            ],
                            "draw_count": len(original_draws),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
