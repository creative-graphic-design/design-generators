"""Unit tests for the fail-closed S2 evidence comparisons."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import run_training_stages as stages
from ralf import RalfForConditionalLayoutGeneration
from ralf.training.datamodule import RalfTrainingBatch
from run_training_stages import (
    _assert_optimizer_state_storage_independent,
    _compare_learning_rates,
    _compare_named_gradients,
    _compare_scheduler_outputs,
    _compare_state_dicts,
    _build_vendor_scheduler,
    _condition_type,
    _fresh_s3_run_root,
    _load_optimizer_state_without_hyperparameters,
    _natural_run_envelope,
    _recipe_epochs,
    _s3_child_import_gate,
    _s3_status,
    _s3_trace_status,
    _run_s3_fit,
    _s3,
    _s4,
    _state_sync_copy_integrity_record,
    main,
    RalfS3TraceCallback,
)


pytestmark = pytest.mark.vendor_parity


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="label RNG placement needs CUDA"
)
def test_label_condition_shuffle_order_matches_vendor_cpu_rng() -> None:
    """Match vendor label-condition element order and its CPU RNG source."""
    from training_reference import require_vendor

    require_vendor(Path(os.environ.get("RALF_CACHE_DIR", ".cache/ralf/cache")))
    from datasets import ClassLabel, Features, Sequence as DatasetSequence, Value
    from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from image2layout.train.helpers.task import get_condition
    from image2layout.train.models.layoutformerpp.task_preprocessor import (
        LabelPreprocessor,
    )
    from ralf.modeling_ralf import (
        RalfConditionalInputs,
        RalfTaskPreprocessor,
        RalfTokenizerView,
    )
    from ralf import RalfConfig, RalfLayoutTokenizer

    labels = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    bbox = torch.full((4, 5, 4), 0.10, dtype=torch.float32)
    bbox[1] = 0.20
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    raw = {
        "id": ["0", "1", "2", "3"],
        "image": torch.zeros(4, 3, 8, 8),
        "saliency": torch.zeros(4, 1, 8, 8),
        "label": labels,
        "center_x": bbox[..., 0],
        "center_y": bbox[..., 1],
        "width": bbox[..., 2],
        "height": bbox[..., 3],
        "mask": mask,
    }
    features = Features(
        {
            "id": Value("string"),
            "label": DatasetSequence(ClassLabel(names=[str(i) for i in range(5)])),
        }
    )
    vendor_tokenizer = LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=5,
        num_bin=8,
        var_order=["label", "width", "height", "center_x", "center_y"],
        special_tokens=["pad", "bos", "eos"],
    )
    vendor_condition, _ = get_condition(raw, "c", vendor_tokenizer)
    vendor_preprocessor = LabelPreprocessor(
        tokenizer=vendor_tokenizer,
        global_task_embedding=False,
    )
    torch.manual_seed(17)
    vendor_sequence = vendor_preprocessor(vendor_condition)["seq"]

    config = RalfConfig(
        dataset_name="cgl",
        id2label={i: str(i) for i in range(5)},
        max_seq_length=5,
        num_bin=8,
    )
    encoded = RalfLayoutTokenizer(config).encode_layout(
        labels=labels,
        bbox=bbox,
        mask=mask,
    )
    package_condition = RalfConditionalInputs(
        image=torch.zeros(4, 4, 8, 8, device="cuda"),
        retrieved={},
        seq=encoded["input_ids"].to("cuda"),
    )
    package_preprocessor = RalfTaskPreprocessor(
        tokenizer=RalfTokenizerView(config),
        task="c",
        global_task_embedding=False,
    )
    torch.manual_seed(17)
    package_sequence = package_preprocessor(package_condition)["seq"]

    assert torch.equal(package_sequence.cpu(), vendor_sequence)


def test_condition_type_maps_canonical_label_to_vendor_task() -> None:
    assert _condition_type("unconditional") == ("unconditional", "uncond")
    assert _condition_type("label") == ("label", "c")


def test_condition_type_rejects_unsupported_tasks() -> None:
    with pytest.raises(ValueError, match="unsupported RALF condition"):
        _condition_type("label_size")


def test_recipe_epochs_follow_pinned_vendor_overrides() -> None:
    assert _recipe_epochs("cgl", "unconditional") == 30
    assert _recipe_epochs("cgl", "label") == 50


def test_loss_pair_reseeds_each_stochastic_condition_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build vendor and package conditions from the same reseeded CPU state."""

    events: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    class StubModel:
        def train(self) -> None:
            return None

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            condition = kwargs["constraint_input_ids"]
            assert isinstance(condition, torch.Tensor)
            return SimpleNamespace(loss=torch.tensor(1.0), logits=condition.float())

    class StubModule:
        condition_type = "label"

        def __init__(self) -> None:
            self.model = cast(RalfForConditionalLayoutGeneration, StubModel())

        def _condition_kwargs(
            self, batch: RalfTrainingBatch
        ) -> dict[str, torch.Tensor]:
            del batch
            state = torch.get_rng_state().clone()
            condition = torch.randperm(4)
            events.append(("package", state, condition))
            return {"constraint_input_ids": condition}

    class StubVendor:
        def train(self) -> None:
            return None

        def train_loss(
            self, inputs: dict[str, object], targets: dict[str, object]
        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            del targets
            condition = inputs["seq"]
            assert isinstance(condition, torch.Tensor)
            return {"logits": condition.float()}, {"nll_loss": torch.tensor(1.0)}

    def stub_vendor_preprocess(
        model: torch.nn.Module, batch: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        del model, batch
        state = torch.get_rng_state().clone()
        condition = torch.randperm(4)
        events.append(("vendor", state, condition))
        return {"seq": condition}, {"seq": condition}

    monkeypatch.setattr(stages, "vendor_preprocess", stub_vendor_preprocess)
    monkeypatch.setattr(stages, "_move_batch", lambda batch, device: batch)
    monkeypatch.setattr(stages, "_vendor_move", lambda values, device: values)
    torch.manual_seed(999)
    batch = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "labels": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.bool),
        "pixel_values": torch.zeros(1, 1),
        "saliency": torch.zeros(1, 1),
        "retrieved": {},
    }
    typed_batch = cast(RalfTrainingBatch, batch)

    package_loss, vendor_loss, package_logits, vendor_logits = stages._loss_pair(
        StubModule(),
        cast(torch.nn.Module, StubVendor()),
        typed_batch,
        torch.device("cpu"),
        seed=17,
    )

    assert [event[0] for event in events] == ["package", "vendor"]
    assert torch.equal(events[0][1], events[1][1])
    assert torch.equal(events[0][2], events[1][2])
    assert torch.equal(package_loss, vendor_loss)
    assert torch.equal(package_logits, vendor_logits)


def _linear_pair() -> tuple[torch.nn.Linear, torch.nn.Linear]:
    package = torch.nn.Linear(2, 2)
    vendor = torch.nn.Linear(2, 2)
    vendor.load_state_dict(package.state_dict())
    return package, vendor


def test_named_gradient_comparison_requires_per_parameter_coverage() -> None:
    package, vendor = _linear_pair()
    inputs = torch.tensor([[1.0, 2.0]])
    package(inputs).sum().backward()
    vendor(inputs).sum().backward()

    result = _compare_named_gradients(package, vendor, "raw_gradients")

    assert result["first_divergence"] is None
    assert result["max_abs_diff"] == 0.0
    assert result["package"]["named_parameter_count"] == 2
    assert result["package"]["present_count"] == 2
    assert result["package"]["presence_digest"] == result["vendor"]["presence_digest"]


def test_named_gradient_comparison_reports_first_parameter_and_max_abs() -> None:
    package, vendor = _linear_pair()
    package.weight.grad = torch.ones_like(package.weight)
    package.bias.grad = torch.ones_like(package.bias)
    vendor.weight.grad = torch.zeros_like(vendor.weight)
    vendor.bias.grad = torch.ones_like(vendor.bias)

    with pytest.raises(
        RuntimeError, match=r"first divergence at raw_gradients\.weight"
    ) as exc_info:
        _compare_named_gradients(package, vendor, "raw_gradients")

    assert "aggregate max_abs_diff=1" in str(exc_info.value)


def test_learning_rate_comparison_reports_both_optimizer_values() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)

    result = _compare_learning_rates(package_optimizer, vendor_optimizer)

    assert result["first_divergence"] is None
    assert result["max_abs_diff"] == 0.0
    assert result["groups"] == [
        {"index": 0, "package": 1e-4, "vendor": 1e-4, "abs_diff": 0.0}
    ]


def test_state_comparison_can_record_natural_drift_without_hiding_it() -> None:
    package_state = {"weight": torch.ones(2)}
    vendor_state = {"weight": torch.zeros(2)}

    result = _compare_state_dicts(
        package_state,
        vendor_state,
        "natural.parameters",
        enforce=False,
    )

    assert result == {
        "first_divergence": "weight",
        "max_abs_diff": 1.0,
    }

    with pytest.raises(RuntimeError, match=r"first divergence at strict\.weight"):
        _compare_state_dicts(package_state, vendor_state, "strict")


def test_natural_run_envelope_reports_run_to_run_drift() -> None:
    # Recorded traces store the loss as one mapping but store each gradient
    # norm as a list with one entry per tracked optimizer.
    first = {
        "trajectory": [
            {
                "global_step": 1,
                "loss": {"package": 2.0, "vendor": 2.1},
                "raw_gradient_norm": [{"package": 3.0, "vendor": 3.1}],
                "clipped_gradient_norm": [{"package": 0.5, "vendor": 0.6}],
                "package_state_sha256": "a",
            }
        ]
    }
    second = {
        "trajectory": [
            {
                "global_step": 1,
                "loss": {"package": 2.25, "vendor": 2.2},
                "raw_gradient_norm": [{"package": 3.5, "vendor": 3.3}],
                "clipped_gradient_norm": [{"package": 0.75, "vendor": 0.8}],
                "package_state_sha256": "b",
            }
        ]
    }

    result = _natural_run_envelope(first, second)

    assert result["run_count"] == 2
    assert result["step_count"] == 1
    max_abs_diff = result["max_abs_diff"]
    assert max_abs_diff["loss"]["package"] == pytest.approx(0.25)
    assert max_abs_diff["loss"]["vendor"] == pytest.approx(0.1)
    assert max_abs_diff["raw_gradient_norm"]["package"] == pytest.approx(0.5)
    assert max_abs_diff["raw_gradient_norm"]["vendor"] == pytest.approx(0.2)
    assert max_abs_diff["clipped_gradient_norm"]["package"] == pytest.approx(0.25)
    assert max_abs_diff["clipped_gradient_norm"]["vendor"] == pytest.approx(0.2)
    assert result["first_package_state_hash_divergence_step"] == 1
    assert result["package_state_hashes_equal"] is False


def test_natural_run_envelope_rejects_optimizer_entry_count_mismatch() -> None:
    step = {
        "global_step": 1,
        "loss": {"package": 2.0, "vendor": 2.0},
        "raw_gradient_norm": [{"package": 3.0, "vendor": 3.0}],
        "clipped_gradient_norm": [{"package": 0.5, "vendor": 0.5}],
        "package_state_sha256": "a",
    }
    second_step = dict(step)
    second_step["raw_gradient_norm"] = [
        {"package": 3.0, "vendor": 3.0},
        {"package": 1.0, "vendor": 1.0},
    ]

    with pytest.raises(RuntimeError, match="raw_gradient_norm"):
        _natural_run_envelope({"trajectory": [step]}, {"trajectory": [second_step]})


def test_s3_uses_production_trainer_and_has_no_manual_scheduler_sentinel() -> None:
    source = inspect.getsource(_s3) + inspect.getsource(_run_s3_fit)

    assert "traingen fit" in source
    assert "RalfTrainingModule" in source
    assert "RalfDataModule" in source
    assert '"scheduler_last_epoch": 0' not in source
    assert "natural_run_to_run_envelope" in source
    assert "state_synchronized_lockstep" in source
    assert "state_sync_copy_integrity" in inspect.getsource(RalfS3TraceCallback)
    assert "--trainer.limit_train_batches=" in source
    assert "--trainer.limit_val_batches=" in source
    assert "train_limit" in source
    assert "validation_limit" in source


def test_s3_uses_condition_recipe_and_training_worker_settings() -> None:
    source = inspect.getsource(_run_s3_fit)

    assert "_recipe_epochs(args.dataset, args.condition)" in source
    assert "--data.init_args.num_workers=4" in source
    assert 'env["OMP_NUM_THREADS"] = "4"' in source


def test_s3_child_uses_repo_root_for_callback_imports() -> None:
    source = inspect.getsource(_run_s3_fit)

    assert "run_training_stages.RalfS3TraceCallback" in source
    assert "run_training_stages.RalfS3CSVLogger" in source
    assert 'callback_root = ROOT / "models" / "ralf"' in source
    assert 'env["PYTHONPATH"]' in source
    assert "str(callback_root)" in source
    assert "os.pathsep" in source
    assert "models.ralf.tests.vendor_parity" not in source
    assert "lib/laygen/src" not in source
    assert "lib/posgen/src" not in source
    assert "lib/traingen/src" not in source
    assert "lib/traingen-parity/src" not in source


def test_s3_child_import_gate_resolves_spawned_paths() -> None:
    _s3_child_import_gate()


def test_stage_evidence_records_runtime_allocator_environment() -> None:
    source = inspect.getsource(main)

    assert '"runtime_environment"' in source
    assert '"pytorch_cuda_alloc_conf"' in source


def test_s3_trace_records_peak_cuda_memory() -> None:
    source = inspect.getsource(RalfS3TraceCallback.on_fit_end)

    assert '"peak_memory_allocated_bytes"' in source
    assert "torch.cuda.max_memory_allocated()" in source


def test_s3_does_not_disable_deterministic_training() -> None:
    source = inspect.getsource(_s3) + inspect.getsource(_run_s3_fit)

    assert "--trainer.deterministic=warn" in source
    assert "--trainer.deterministic=true" not in source
    assert "--trainer.deterministic=false" not in source


def test_training_configs_use_lightning_warning_mode() -> None:
    root = Path(__file__).parents[4]
    for dataset in ("cgl", "pku"):
        source = (
            root / "models" / "ralf" / "configs" / "training" / f"{dataset}.yaml"
        ).read_text()
        assert "deterministic: warn" in source
        assert "deterministic: true" not in source


def test_s3_requires_effective_deterministic_warning_state() -> None:
    source = inspect.getsource(RalfS3TraceCallback.on_fit_start)

    assert "torch.are_deterministic_algorithms_enabled()" in source
    assert "torch.is_deterministic_algorithms_warn_only_enabled()" in source
    assert "lightning_trainer.deterministic" not in source


def test_s3_reference_backward_restores_package_rng_state() -> None:
    source = inspect.getsource(RalfS3TraceCallback.on_after_backward)

    assert "package_rng_after_backward" in source
    assert "restore_rng_state(package_rng_after_backward)" in source


def test_s3_optimizer_state_sync_separates_tensor_storage(tmp_path: Path) -> None:
    del tmp_path
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package(torch.ones(1, 2)).sum().backward()
    vendor(torch.ones(1, 2)).sum().backward()
    package_optimizer.step()
    vendor_optimizer.step()

    _load_optimizer_state_without_hyperparameters(vendor_optimizer, package_optimizer)
    result = _assert_optimizer_state_storage_independent(
        package_optimizer, vendor_optimizer, package, vendor, "S3.state_sync"
    )

    for package_parameter, vendor_parameter in zip(
        package_optimizer.state, vendor_optimizer.state, strict=True
    ):
        package_state = package_optimizer.state[package_parameter]
        vendor_state = vendor_optimizer.state[vendor_parameter]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert package_state[key] is not vendor_state[key]
            assert package_state[key].data_ptr() != vendor_state[key].data_ptr()
    assert result["checked_entries"] > 0
    assert result["shared_storage_entries"] == []


def test_optimizer_state_storage_check_fails_closed_on_shared_storage() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package(torch.ones(1, 2)).sum().backward()
    vendor(torch.ones(1, 2)).sum().backward()
    package_optimizer.step()
    vendor_optimizer.step()

    vendor_optimizer.state[vendor.weight]["exp_avg"] = package_optimizer.state[
        package.weight
    ]["exp_avg"]

    with pytest.raises(RuntimeError, match=r"weight\.exp_avg") as exc_info:
        _assert_optimizer_state_storage_independent(
            package_optimizer, vendor_optimizer, package, vendor, "S3.state_sync"
        )

    assert "optimizer_state_storage" in str(exc_info.value)


def test_optimizer_state_copy_keeps_each_system_hyperparameters() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=5e-4)
    package(torch.ones(1, 2)).sum().backward()
    package_optimizer.step()

    _load_optimizer_state_without_hyperparameters(vendor_optimizer, package_optimizer)

    assert vendor_optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert set(vendor_optimizer.state) == {vendor.weight, vendor.bias}
    assert torch.equal(
        vendor_optimizer.state[vendor.weight]["exp_avg"],
        package_optimizer.state[package.weight]["exp_avg"],
    )


def test_synchronized_layer_enforces_independently_produced_learning_rates() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=5e-4)

    with pytest.raises(RuntimeError, match="learning_rates"):
        _compare_learning_rates(package_optimizer, vendor_optimizer)

    assert vendor_optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_scheduler_output_comparison_reports_each_system_own_values() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        package_optimizer, milestones=[1], gamma=0.1
    )
    vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        vendor_optimizer, milestones=[2], gamma=0.1
    )

    result = _compare_scheduler_outputs(
        package_scheduler, vendor_scheduler, "S3.epoch[0]", enforce=False
    )

    assert result["first_divergence"] == "milestones"
    assert result["package_milestones"] == [1]
    assert result["vendor_milestones"] == [2]

    package_scheduler.step()
    vendor_scheduler.step()

    with pytest.raises(RuntimeError, match=r"S3\.epoch\[0\]\.scheduler_outputs"):
        _compare_scheduler_outputs(package_scheduler, vendor_scheduler, "S3.epoch[0]")


def test_scheduler_reference_uses_the_pinned_vendor_fractional_config() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        package_optimizer, milestones=[20], gamma=0.1
    )
    vendor_scheduler = _build_vendor_scheduler(vendor_optimizer, epochs=30)

    result = _compare_scheduler_outputs(
        package_scheduler, vendor_scheduler, "S3.vendor_scheduler", enforce=False
    )

    assert set(vendor_scheduler.milestones) == {21}
    assert result["first_divergence"] == "milestones"


def test_scheduler_output_comparison_accepts_two_agreeing_schedulers() -> None:
    package, vendor = _linear_pair()
    package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        torch.optim.AdamW(package.parameters(), lr=1e-4), milestones=[1], gamma=0.1
    )
    vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        torch.optim.AdamW(vendor.parameters(), lr=1e-4), milestones=[1], gamma=0.1
    )
    package_scheduler.step()
    vendor_scheduler.step()

    matched = _compare_scheduler_outputs(
        package_scheduler, vendor_scheduler, "S3.epoch[1]"
    )

    assert matched["first_divergence"] is None
    assert matched["max_abs_diff"] == 0.0
    assert matched["package_last_lrs"] == matched["vendor_last_lrs"]
    assert matched["package_last_lrs"] != [1e-4]
    assert matched["package_last_epoch"] == 1


def test_scheduler_comparison_records_epoch_offset_separately() -> None:
    package, vendor = _linear_pair()
    package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        torch.optim.AdamW(package.parameters(), lr=1e-4), milestones=[1], gamma=0.1
    )
    vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        torch.optim.AdamW(vendor.parameters(), lr=1e-4), milestones=[1], gamma=0.1
    )
    package_scheduler.last_epoch = 2
    vendor_scheduler.last_epoch = 1

    result = _compare_scheduler_outputs(
        package_scheduler, vendor_scheduler, "S3.epoch[2]", enforce=False
    )

    assert result["first_divergence"] == "last_epoch"
    assert result["epoch_offset"] == 1
    assert result["max_abs_diff"] == 0.0


def test_scheduler_group_mismatch_is_strict_json_safe() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(
        [
            {"params": [vendor.weight]},
            {"params": [vendor.bias]},
        ],
        lr=1e-4,
    )
    package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        package_optimizer, milestones=[1], gamma=0.1
    )
    vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        vendor_optimizer, milestones=[1], gamma=0.1
    )

    result = _compare_scheduler_outputs(
        package_scheduler, vendor_scheduler, "S3.group_count", enforce=False
    )

    assert result["first_divergence"] == "group_count"
    assert result["max_abs_diff"] is None
    json.dumps(result, allow_nan=False)


def test_s3_trace_status_uses_layered_vocabulary() -> None:
    assert _s3_trace_status("natural", None) == "PASS"
    assert _s3_trace_status("natural", {"location": "loss"}) == "LEFT_CONTRACT"
    assert _s3_trace_status("synchronized", None) == "PASS"
    assert _s3_trace_status("synchronized", {"location": "loss"}) == "FAIL"


def test_state_sync_record_names_read_as_copy_integrity() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package(torch.ones(1, 2)).sum().backward()
    package_optimizer.step()
    vendor.load_state_dict(package.state_dict())
    _load_optimizer_state_without_hyperparameters(vendor_optimizer, package_optimizer)

    record = _state_sync_copy_integrity_record(
        package_optimizer,
        vendor_optimizer,
        package,
        vendor,
        "S3.global_step[1]",
    )

    assert set(record) == {
        "parameters_copy_integrity",
        "optimizer_state_copy_integrity",
        "optimizer_state_storage_independence",
        "package_state_sha256_after_copy",
        "vendor_state_sha256_after_copy",
        "note",
    }
    assert "copy" in str(record["note"])
    assert not any(key.endswith("divergence") for key in record)


def test_s3_status_reports_both_evidence_layers() -> None:
    natural_pass = [{"first_divergence": None}, {"first_divergence": None}]
    natural_left = [
        {"first_divergence": None},
        {"first_divergence": {"location": "S3.epoch[0].batch[1].raw_gradients"}},
    ]
    synchronized_pass = {"first_divergence": None}
    synchronized_fail = {"first_divergence": {"location": "S3.global_step[1]"}}

    assert _s3_status(natural_pass, synchronized_pass) == {
        "natural": "PASS",
        "synchronized": "PASS",
        "verdict": "pass",
    }
    assert _s3_status(natural_left, synchronized_pass) == {
        "natural": "LEFT_CONTRACT",
        "synchronized": "PASS",
        "verdict": "bounded-pass",
    }
    assert _s3_status(natural_left, synchronized_fail) == {
        "natural": "LEFT_CONTRACT",
        "synchronized": "FAIL",
        "verdict": "fail",
    }
    assert _s3_status(natural_pass, None) == {
        "natural": "PASS",
        "synchronized": "NOT_RUN",
        "verdict": "pass",
    }
    assert _s3_status(natural_left, None) == {
        "natural": "LEFT_CONTRACT",
        "synchronized": "NOT_RUN",
        "verdict": "fail",
    }

    with pytest.raises(RuntimeError, match="natural"):
        _s3_status([], synchronized_pass)


def test_s3_emits_the_layered_status_structure() -> None:
    source = inspect.getsource(_s3)
    callback_source = inspect.getsource(RalfS3TraceCallback.on_fit_end)

    assert "_s3_status(natural_runs, synchronized_run)" in source
    assert '"status": synchronized_run["status"]' not in source
    assert '"RECORDED"' not in callback_source


def test_s3_run_preserves_prior_artifacts(tmp_path: Path) -> None:
    first = _fresh_s3_run_root(tmp_path)
    first.mkdir(parents=True)
    second = _fresh_s3_run_root(tmp_path)

    assert first != second
    assert first.name == "run-001"
    assert second.name == "run-002"


def test_s4_records_authoritative_train_validation_stream_evidence() -> None:
    source = inspect.getsource(_s4)

    for required_surface in (
        "train_dataloader()",
        "val_dataloader()",
        '"split_membership"',
        '"serialized_sha256"',
        '"package_stream_sha256"',
        '"vendor_stream_sha256"',
    ):
        assert required_surface in source


def test_s4_points_vendor_retrieval_cache_at_authoritative_cache() -> None:
    source = inspect.getsource(_s4)

    assert "image2layout.train.global_variables" in source
    assert "PRECOMPUTED_WEIGHT_DIR" in source


def test_s4_uses_existing_vendor_legacy_torch_load_mode() -> None:
    source = inspect.getsource(_s4)

    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in source


def test_s4_derives_order_from_the_production_batch_sampler() -> None:
    source = inspect.getsource(_s4)

    assert "batch_sampler" in source
    assert "torch.empty((), dtype=torch.int64).random_()" in source


def test_s4_checks_vendor_loader_order_by_authoritative_ids() -> None:
    source = inspect.getsource(_s4)

    assert 'vendor_batch["id"]' in source


def test_s4_restores_rng_between_lazy_loader_iterators() -> None:
    source = inspect.getsource(_s4)

    assert "package_loader_rng" in source
    assert "restore_rng_state(package_loader_rng)" in source


def test_s4_compares_vendor_collated_retrieval_batch() -> None:
    source = inspect.getsource(_s4)

    assert 'batch["retrieved"]' in source
