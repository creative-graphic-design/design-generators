"""Compare one original and package Flex-DM optimizer step."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
import torch


_DIRECT_LIMIT = 3.5e-4
_FLOAT64_LIMIT = 4.1e-3
_ZERO_GRADIENT_LIMIT = 1e-6
_LEARNING_RATE = 1e-4
_BATCH_HASHES = {
    "crello": "8e4863c9462193a48786ce29d8d23a6edf01995eaa912bc9ad524e10f03b7d6d",
    "rico": "54cb191cdfee1736465bf1c82dd15f1543a0fa784220ce284b49745f5d74abed",
}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.to(device=expected.device, dtype=torch.float64)
    expected = expected.double()
    difference = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-30)
    return float((difference / denominator).item())


def _reference_step(
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradient = gradient.double()
    first_moment = gradient * (1 - 0.9)
    second_moment = gradient.square() * (1 - 0.999)
    alpha = _LEARNING_RATE * (1 - 0.999**1) ** 0.5 / (1 - 0.9**1)
    update = -(first_moment * alpha) / (second_moment.sqrt() + 1e-7)
    return first_moment, second_moment, update


def _float64_model_trace(
    module,
    initial_state: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_ids: torch.Tensor,
    dropout_masks: list[np.ndarray],
) -> dict[str, dict[str, torch.Tensor]]:
    from flex_dm import FlexDmForMaskedDocumentModeling
    from traingen.optim import clip_gradients_by_norm
    from .test_flex_dm_training_s1 import _replay_dropout_masks

    model = FlexDmForMaskedDocumentModeling(module.model.config).to(
        device="cuda", dtype=torch.float64
    )
    model.load_state_dict(
        {
            key: value.to(device="cuda", dtype=torch.float64)
            for key, value in initial_state.items()
        }
    )
    model.train()
    consumed = _replay_dropout_masks(model, dropout_masks)
    double_inputs = {
        key: value.double() if value.is_floating_point() else value
        for key, value in inputs.items()
    }
    double_labels = {
        key: value.double() if value.is_floating_point() else value
        for key, value in labels.items()
    }
    output = model(
        inputs=double_inputs,
        labels=double_labels,
        masks=masks,
        task_ids=task_ids,
    )
    if output.loss is None:
        raise AssertionError("float64 arbitration requires a loss")

    output.loss.backward()
    raw = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    clip_gradients_by_norm(model.parameters(), max_norm=1.0)
    clipped = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert consumed[0] == len(dropout_masks)
    first: dict[str, torch.Tensor] = {}
    second: dict[str, torch.Tensor] = {}
    for name, gradient in clipped.items():
        first[name], second[name], _ = _reference_step(gradient)
    return {"raw": raw, "clipped": clipped, "m": first, "v": second}


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", ("crello", "rico"))
def test_s2_flex_dm_trainer_step_matches_reference(dataset: str) -> None:
    """Run one reference step and one shipped Lightning trainer step."""
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader
    from traingen.optim import KerasAdam
    from flex_dm.conversion import map_tensor_by_rule
    from flex_dm.data_specs import load_builtin_spec
    from flex_dm.training.datamodule import FlexDmDataModule
    from flex_dm.training.lightning_module import FlexDmTrainingModule

    assert os.environ.get("PARITY_REQUIRE") == "1"
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32

    reference_python = os.environ.get("FLEX_DM_REFERENCE_PYTHON")
    reference_dir = os.environ.get("FLEX_DM_S2_REFERENCE_DIR")
    asset_dir = os.environ.get("FLEX_DM_ORIGINAL_ASSET_DIR")
    if not reference_python or not reference_dir or not asset_dir:
        pytest.fail(
            "S2 requires reference Python, reference directory, and original assets"
        )

    saved_reference = os.environ.get("FLEX_DM_S2_REFERENCE_ARTIFACT")
    if saved_reference:
        artifact_path = Path(saved_reference)
        if not artifact_path.is_file():
            pytest.fail(f"S2 reference artifact does not exist: {artifact_path}")
    else:
        artifact_dir = Path(reference_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / (
            f"{dataset}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}.npz"
        )
        script = Path(__file__).with_name("capture_flex_dm_training_s2.py")
        reference_env = os.environ.copy()
        reference_env["CUDA_VISIBLE_DEVICES"] = "0"
        reference_env["NVIDIA_TF32_OVERRIDE"] = "0"
        subprocess.run(
            [
                reference_python,
                str(script),
                "--dataset",
                dataset,
                "--output",
                str(artifact_path),
            ],
            check=True,
            env=reference_env,
            text=True,
        )

    with np.load(artifact_path, allow_pickle=False) as saved:
        reference = {name: saved[name].copy() for name in saved.files}
    metadata = json.loads(str(reference.pop("metadata")))
    assert metadata["dataset"] == dataset
    assert metadata["batch_size"] == 256
    assert metadata["seed"] == 0
    assert metadata["iterations"] == 1
    assert metadata["learning_rate"] == pytest.approx(_LEARNING_RATE)
    assert metadata["batch_sha256"] == _BATCH_HASHES[dataset]
    assert metadata["betas"] == [0.9, 0.999]
    assert metadata["epsilon"] == pytest.approx(1e-7)
    assert metadata["clipnorm"] == pytest.approx(1.0)
    assert metadata["global_clipnorm"] is None

    original_data_dir = Path(asset_dir) / "data" / dataset
    if (original_data_dir / dataset).is_dir():
        original_data_dir /= dataset
    data = FlexDmDataModule(
        data_dir=str(original_data_dir),
        dataset_name=dataset,
        batch_size=256,
        shuffle_train=False,
    )
    data.setup("fit")
    package_batch = next(iter(data.train_dataloader()))
    expected_batch = {
        key.removeprefix("batch::"): value
        for key, value in reference.items()
        if key.startswith("batch::")
    }
    assert set(package_batch) <= set(expected_batch)
    assert set(expected_batch) - set(package_batch) == (
        {"id", "uuid"} if dataset == "crello" else set()
    )
    source_columns = load_builtin_spec(dataset)["columns"]
    lengths = package_batch["length"].squeeze(-1) + 1
    for key in package_batch:
        expected = torch.as_tensor(expected_batch[key], dtype=package_batch[key].dtype)
        if source_columns[key]["is_sequence"]:
            # The reference maps padded blanks through lookup layers; compare source tokens.
            valid = torch.arange(package_batch[key].shape[1]).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            torch.testing.assert_close(
                package_batch[key][valid], expected[valid], rtol=0, atol=0
            )
            continue

        torch.testing.assert_close(
            package_batch[key],
            expected,
            rtol=0,
            atol=0,
        )

    def tensors(
        prefix: str,
        *,
        dtype: torch.dtype | None = None,
        keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            key.removeprefix(f"{prefix}::"): torch.as_tensor(
                value, device="cuda", dtype=dtype
            )
            for key, value in reference.items()
            if key.startswith(f"{prefix}::")
            and (keys is None or key.removeprefix(f"{prefix}::") in keys)
        }

    module = FlexDmTrainingModule(
        str(Path(asset_dir).parents[0] / "converted" / f"flex-dm-{dataset}")
    )
    model_columns = set(module.model.config.input_columns)
    prepared_inputs = tensors("inputs", keys=model_columns)
    labels = tensors("labels", keys=model_columns)
    masks = tensors("masks", dtype=torch.bool, keys=model_columns)
    task_ids = torch.as_tensor(reference["tasks"], device="cuda", dtype=torch.long)
    dropout_masks = [
        reference[key]
        for key in sorted(
            (name for name in reference if name.startswith("dropout::")),
            key=lambda name: int(name.removeprefix("dropout::")),
        )
    ]
    assert module.automatic_optimization
    assert module.model.config.dataset_name == dataset
    module.prepare_batch = lambda _batch: (
        prepared_inputs,
        labels,
        masks,
        task_ids,
    )
    with pytest.raises(ValueError, match="per-parameter clip_norm"):
        module.configure_gradient_clipping(
            module.configure_optimizers(), gradient_clip_val=1.0
        )
    from .test_flex_dm_training_s1 import _replay_dropout_masks

    dropout_count = _replay_dropout_masks(module.model, dropout_masks)
    raw_gradients: dict[str, torch.Tensor] = {}
    clipped_gradients: dict[str, torch.Tensor] = {}
    clip_hook_calls = 0

    def capture_raw(optimizer) -> None:
        del optimizer
        raw_gradients.update(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in module.model.named_parameters()
                if parameter.grad is not None
            }
        )

    configure_clipping = module.configure_gradient_clipping

    def capture_clipped(
        optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        nonlocal clip_hook_calls
        configure_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        clip_hook_calls += 1
        clipped_gradients.update(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in module.model.named_parameters()
                if parameter.grad is not None
            }
        )

    module.on_before_optimizer_step = capture_raw
    module.configure_gradient_clipping = capture_clipped
    initial_state = {
        key: value.detach().clone() for key, value in module.model.state_dict().items()
    }
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=1,
        limit_train_batches=1,
        max_epochs=1,
        gradient_clip_val=0.0,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=DataLoader([package_batch], batch_size=None))
    assert trainer.global_step == 1
    assert clip_hook_calls == 1
    assert dropout_count[0] == len(dropout_masks)

    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, KerasAdam)
    assert optimizer.param_groups[0]["lr"] == _LEARNING_RATE
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-7
    assert all(int(state["step"]) == 1 for state in optimizer.state.values())

    reference_groups: dict[str, dict[str, torch.Tensor]] = {}
    for group in ("raw", "clipped", "m", "v", "after", "before"):
        values: dict[str, torch.Tensor] = {}
        for name, array in reference.items():
            prefix = f"{group}::"
            if not name.startswith(prefix):
                continue
            mapped = map_tensor_by_rule(name.removeprefix(prefix), array)
            if mapped is None:
                continue
            key, tensor = mapped
            if key in values:
                raise AssertionError(f"multiple original tensors map to {key}")
            values[key] = tensor.to(device="cuda")
        reference_groups[group] = values

    initial_parameters = {
        name: initial_state[name].detach().to(device="cuda")
        for name, _ in module.model.named_parameters()
    }
    assert set(initial_parameters) == set(reference_groups["before"])
    assert all(
        torch.equal(initial_parameters[name], reference_groups["before"][name])
        for name in initial_parameters
    ), "package and original must start from the same released checkpoint"

    package_gradients = {
        "raw": raw_gradients,
        "clipped": clipped_gradients,
        "m": {
            name: optimizer.state[parameter]["exp_avg"].detach().clone()
            for name, parameter in module.model.named_parameters()
            if parameter in optimizer.state
        },
        "v": {
            name: optimizer.state[parameter]["exp_avg_sq"].detach().clone()
            for name, parameter in module.model.named_parameters()
            if parameter in optimizer.state
        },
        "after": {
            name: parameter.detach().to(device="cuda")
            for name, parameter in module.model.named_parameters()
        },
    }
    package_gradients["before"] = {
        name: initial_state[name].to(device="cuda")
        for name in package_gradients["after"]
    }
    key_biases = {
        name
        for name in package_gradients["raw"]
        if name.endswith("attention.k_proj.bias")
    }
    assert key_biases

    # The reference precedent treats these softmax-invariant biases as exact zero gradients.
    zero_gradient_maxima = {
        "package": max(
            float(package_gradients["raw"][name].abs().max().item())
            for name in key_biases
        ),
        "reference": max(
            float(reference_groups["raw"][name].abs().max().item())
            for name in key_biases
        ),
    }
    assert max(zero_gradient_maxima.values()) <= _ZERO_GRADIENT_LIMIT

    for group in ("raw", "clipped", "m", "v"):
        assert set(package_gradients[group]) == set(reference_groups[group]), (
            group,
            sorted(set(package_gradients[group]) ^ set(reference_groups[group])),
        )

    direct: dict[str, dict[str, float]] = {}
    for group in ("raw", "clipped", "m", "v"):
        direct[group] = {
            name: _relative_l2(
                package_gradients[group][name], reference_groups[group][name]
            )
            for name in package_gradients[group]
            if name not in key_biases
        }
    direct_maxima = {
        group: max(values.values(), default=0.0) for group, values in direct.items()
    }

    exact: dict[str, dict[str, torch.Tensor]] | None = None
    if any(
        value > _DIRECT_LIMIT for values in direct.values() for value in values.values()
    ):
        exact = _float64_model_trace(
            module,
            initial_state,
            prepared_inputs,
            labels,
            masks,
            task_ids,
            dropout_masks,
        )

    arbitration_maxima: dict[str, float] = {}
    for group, values in direct.items():
        if max(values.values(), default=0.0) <= _DIRECT_LIMIT:
            continue
        if exact is None:
            raise AssertionError("float64 arbitration was not computed")
        group_errors = []
        for name, error in values.items():
            if error <= _DIRECT_LIMIT:
                continue
            float64_value = exact[group][name]
            group_errors.extend(
                (
                    _relative_l2(package_gradients[group][name], float64_value),
                    _relative_l2(reference_groups[group][name], float64_value),
                )
            )
        arbitration_maxima[group] = max(group_errors, default=0.0)

    update_errors: dict[str, dict[str, float]] = {"package": {}, "reference": {}}
    for name in package_gradients["clipped"]:
        _, _, package_update = _reference_step(package_gradients["clipped"][name])
        _, _, original_update = _reference_step(reference_groups["clipped"][name])
        update_errors["package"][name] = _relative_l2(
            package_gradients["after"][name] - package_gradients["before"][name],
            package_update,
        )
        update_errors["reference"][name] = _relative_l2(
            reference_groups["after"][name] - reference_groups["before"][name],
            original_update,
        )

    update_differences = {
        name: _relative_l2(
            package_gradients["after"][name] - package_gradients["before"][name],
            reference_groups["after"][name] - reference_groups["before"][name],
        )
        for name in package_gradients["after"]
        if name in reference_groups["after"]
    }
    largest_direct = {}
    for group, values in direct.items():
        name = max(values, key=values.__getitem__)
        package_norm = float(
            torch.linalg.vector_norm(package_gradients[group][name].double()).item()
        )
        reference_norm = float(
            torch.linalg.vector_norm(reference_groups[group][name].double()).item()
        )
        largest_direct[group] = {
            "parameter": name,
            "norm_rel": values[name],
            "package_norm": package_norm,
            "reference_norm": reference_norm,
        }

    maxima = {
        "loss": {
            "package": float(module.latest_step_trace["train_loss"].item()),
            "reference": float(metadata["metrics"]["loss"]),
        },
        "key_projection_bias_max_abs": zero_gradient_maxima,
        "direct_norm_rel": direct_maxima,
        "direct_failures": {
            group: [name for name, error in values.items() if error > _DIRECT_LIMIT]
            for group, values in direct.items()
        },
        "largest_direct": largest_direct,
        "float64_arbitration_norm_rel": arbitration_maxima,
        "own_float64_update_norm_rel": {
            name: max(values.values(), default=0.0)
            for name, values in update_errors.items()
        },
        "package_vs_reference_update_norm_rel_report_only": max(
            update_differences.values(), default=0.0
        ),
    }
    print(
        json.dumps(
            {"dataset": dataset, "artifact": str(artifact_path), "maxima": maxima},
            sort_keys=True,
        )
    )

    assert all(value <= _DIRECT_LIMIT for value in direct_maxima.values()) or all(
        value <= _FLOAT64_LIMIT for value in arbitration_maxima.values()
    ), maxima
    assert all(
        value <= _FLOAT64_LIMIT for value in update_errors["package"].values()
    ), maxima
    assert all(
        value <= _FLOAT64_LIMIT for value in update_errors["reference"].values()
    ), maxima
