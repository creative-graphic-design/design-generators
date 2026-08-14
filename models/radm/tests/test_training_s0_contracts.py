from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest
import numpy as np
import torch

from radm import RADMConfig, RADMDenoiser
from radm.training.config import effective_radm_config
from radm.training.dataset import (
    RADMDataCollator,
    RADMTrainingExample,
    _apply_training_transforms,
)
from radm.training.optim import build_radm_optimizer, build_radm_scheduler
from radm.modeling_radm import RADMProposalHead, RADMDenoiserOutput
from radm.training.topology import (
    assert_radm_package_topology,
    assert_forward_parity,
    build_state_key_map,
    compare_state_dict_topology,
)


_VENDOR_DEPENDENCY_NAMES = ("detectron2", "fvcore", "iopath")
_PROHIBITED_RUNTIME_LANGUAGE = (
    "detectron2",
    "original head",
    "original implementation",
    "original cosine",
    "original sampler",
    "original loss",
    "original simota",
    "original full-model",
)


def _probe_training_namespace(*, block_lightning: bool) -> dict[str, bool]:
    """Return the training namespace state from an isolated interpreter."""
    repository_root = Path(__file__).parents[3]
    script = """
import importlib.util
import json
import sys

if BLOCK_LIGHTNING:
    sys.modules["lightning"] = None

import radm

root_before_training_import = "training" not in radm.__dict__
import radm.training as training

print(json.dumps({
    "lightning_available": importlib.util.find_spec("lightning") is not None,
    "root_before_training_import": root_before_training_import,
    "training_module_loaded": "radm.training.lightning_module" in sys.modules,
    "data_module_loaded": "radm.training.datamodule" in sys.modules,
    "has_training_module": hasattr(training, "RADMTrainingModule"),
    "has_data_module": hasattr(training, "RADMDataModule"),
}))
""".replace("BLOCK_LIGHTNING", repr(block_lightning), 1)
    environment = os.environ.copy()
    source_path = str(repository_root / "models/radm/src")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, environment.get("PYTHONPATH", "")) if path
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, bool], json.loads(completed.stdout))


def test_s0_training_namespace_respects_optional_lightning() -> None:
    """Import training classes only when the optional dependency is present."""
    repository_root = Path(__file__).parents[3]
    namespace_source = (
        repository_root / "models/radm/src/radm/training/__init__.py"
    ).read_text(encoding="utf-8")

    assert (
        namespace_source.count("from importlib.util import find_spec as _find_spec")
        == 1
    )
    assert namespace_source.count('if _find_spec("lightning") is not None:') == 1
    assert "__all__" not in namespace_source
    assert "__getattr__" not in namespace_source
    assert "globals(" not in namespace_source

    training_root = repository_root / "models/radm/src/radm/training"

    for source_path in sorted(training_root.glob("*.py")):
        source_text = source_path.read_text(encoding="utf-8")
        assert "__getattr__" not in source_text, source_path
        assert "laygen.common.import_utils" not in source_text, source_path
        assert "_build_training_module" not in source_text, source_path
        assert "_build_data_module" not in source_text, source_path
        assert "globals(" not in source_text, source_path

    for config_path in sorted(
        (repository_root / "models/radm/configs/training").glob("radm_*.yaml")
    ):
        config_text = config_path.read_text(encoding="utf-8")
        assert "radm.training.lightning_module.RADMTrainingModule" in config_text
        assert "radm.training.datamodule.RADMDataModule" in config_text
        assert "class_path: radm.training.RADM" not in config_text

    without_lightning = _probe_training_namespace(block_lightning=True)
    assert without_lightning == {
        "lightning_available": False,
        "root_before_training_import": True,
        "training_module_loaded": False,
        "data_module_loaded": False,
        "has_training_module": False,
        "has_data_module": False,
    }

    with_environment = _probe_training_namespace(block_lightning=False)
    assert with_environment["root_before_training_import"] is True
    if with_environment["lightning_available"]:
        assert with_environment["training_module_loaded"] is True
        assert with_environment["data_module_loaded"] is True
        assert with_environment["has_training_module"] is True
        assert with_environment["has_data_module"] is True
    else:
        assert with_environment["training_module_loaded"] is False
        assert with_environment["data_module_loaded"] is False
        assert with_environment["has_training_module"] is False
        assert with_environment["has_data_module"] is False


def test_s0_runtime_source_uses_package_neutral_language() -> None:
    """Keep reference-stack wording out of runtime and training source."""
    repository_root = Path(__file__).parents[3]
    source_root = repository_root / "models/radm/src/radm"
    violations: list[str] = []
    for source_path in sorted(source_root.rglob("*.py")):
        if source_path.name == "conversion.py":
            continue
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            lowered = line.casefold()
            for phrase in _PROHIBITED_RUNTIME_LANGUAGE:
                if phrase in lowered:
                    violations.append(
                        f"{source_path.relative_to(repository_root)}:{line_number}: "
                        f"{line.strip()}"
                    )
                    break
    assert not violations, "prohibited reference-stack wording: " + repr(violations)


def test_s0_optimizer_is_the_only_gradient_clipping_owner() -> None:
    """Disable Lightning clipping while retaining captured optimizer clipping."""
    repository_root = Path(__file__).parents[3]
    config_paths = sorted(
        (repository_root / "models/radm/configs/training").glob("radm_*.yaml")
    )
    assert config_paths
    for config_path in config_paths:
        config_text = config_path.read_text(encoding="utf-8")
        assert "gradient_clip_val: 0.0" in config_text, config_path
        assert "gradient_clip_algorithm" not in config_text, config_path
    optimizer_text = (
        repository_root / "models/radm/src/radm/training/optim.py"
    ).read_text(encoding="utf-8")
    assert "clip_grad_norm_" in optimizer_text


def test_s0_package_topology_rejects_unmatched_runtime_class_mapping() -> None:
    """Require the package config to preserve the live five-label vocabulary."""
    effective = replace(
        effective_radm_config(),
        class_id_to_label={
            **effective_radm_config().class_id_to_label,
            4: "different runtime label",
        },
    )
    config = RADMConfig(
        num_classes=effective.num_classes,
        num_proposals=effective.num_proposals,
        hidden_dim=effective.hidden_dim,
        text_feature_dim=effective.text_feature_dim,
        original_id2label=effective_radm_config().class_id_to_label,
    )
    model = RADMDenoiser(config=config)

    with pytest.raises(AssertionError, match="class mapping"):
        assert_radm_package_topology(model, effective)


def test_s0_forward_parity_checks_derived_diffusion_outputs() -> None:
    """Compare the complete package forward result, including derived fields."""
    package_output = RADMDenoiserOutput(
        logits=torch.zeros(1, 1, 1),
        boxes_xyxy=torch.zeros(1, 1, 4),
        pred_original_sample=torch.zeros(1, 1, 4),
        pred_noise=torch.zeros(1, 1, 4),
        auxiliary_logits=torch.zeros(1, 1, 1, 1),
        auxiliary_boxes_xyxy=torch.zeros(1, 1, 1, 4),
    )
    reference_output = {
        "auxiliary_logits": torch.zeros(1, 1, 1, 1),
        "auxiliary_boxes_xyxy": torch.zeros(1, 1, 1, 4),
        "pred_original_sample": torch.zeros(1, 1, 4),
        "pred_noise": torch.ones(1, 1, 4),
    }

    with pytest.raises(AssertionError, match="pred_noise"):
        assert_forward_parity(reference_output, package_output)


def test_s0_training_doc_records_accepted_topology_evidence() -> None:
    """Keep the durable S0 record aligned with the recorded evidence."""
    repository_root = Path(__file__).parents[3]
    training_doc = (repository_root / "models/radm/TRAINING.md").read_text(
        encoding="utf-8"
    )
    normalized_doc = training_doc.casefold()
    assert "text encoding fields are derived from the selected mapper" in normalized_doc
    assert "class-mapping" in normalized_doc
    assert "class-mapping and derived-output guards" in normalized_doc
    assert "test_s0_radm_topology.py" in normalized_doc
    assert "23 passed" in normalized_doc
    assert "413f87a45760ceac5635b6a08c8047f86478acf5" in normalized_doc
    assert "green" not in normalized_doc
    assert "reference runtime investigation" not in normalized_doc


def test_s0_real_topology_test_invokes_package_topology_guard() -> None:
    """Keep the real-model test on the package-side static topology guard."""
    repository_root = Path(__file__).parents[3]
    topology_test = (
        repository_root / "models/radm/tests/vendor_parity/test_s0_radm_topology.py"
    ).read_text(encoding="utf-8")
    assert "assert_radm_package_topology" in topology_test
    assert "assert_radm_package_topology(package, state.effective)" in topology_test


def test_s0_training_surface_excludes_vendor_dependencies_and_language() -> None:
    """Keep package-local training independent of the reference stack."""
    repository_root = Path(__file__).parents[3]
    package_metadata = tomllib.loads(
        (repository_root / "models/radm/pyproject.toml").read_text(encoding="utf-8")
    )
    training_dependencies = cast(
        list[str], package_metadata["project"]["optional-dependencies"]["training"]
    )
    dependency_leaks = [
        dependency
        for dependency in training_dependencies
        if any(name in dependency.casefold() for name in _VENDOR_DEPENDENCY_NAMES)
    ]

    source_root = repository_root / "models/radm/src/radm/training"
    source_leaks: list[str] = []
    for source_path in sorted(source_root.rglob("*.py")):
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if any(name in line.casefold() for name in _VENDOR_DEPENDENCY_NAMES):
                source_leaks.append(
                    f"{source_path.relative_to(repository_root)}:{line_number}: "
                    f"{line.strip()}"
                )

    assert not dependency_leaks, (
        f"vendor dependencies leaked into the training extra: {dependency_leaks}"
    )
    assert not source_leaks, (
        "vendor dependency/reference language leaked into package training source: "
        f"{source_leaks}"
    )


def test_s0_effective_config_exposes_four_class_five_label_discrepancy() -> None:
    effective = effective_radm_config()

    assert effective.num_classes == 4
    assert effective.vocabulary_size == 5
    assert effective.class_id_to_label == {
        0: "Logo",
        1: "文字",
        2: "衬底",
        3: "符号元素",
        4: "强调突出子部分文字",
    }
    assert effective.predicted_class_id_to_label == {
        0: "Logo",
        1: "文字",
        2: "衬底",
        3: "符号元素",
    }
    assert effective.predicted_class_id_to_label != effective.class_id_to_label


def test_s0_effective_config_captures_optimizer_scheduler_and_sampler_state() -> None:
    effective = effective_radm_config()

    assert effective.optimizer == "ADAMW"
    assert effective.learning_rate == 2.5e-5
    assert effective.weight_decay == 1.0e-4
    assert effective.betas == (0.9, 0.999)
    assert effective.eps == 1.0e-8
    assert effective.gradient_clip_norm == 1.0
    assert effective.warmup_iters == 1000
    assert effective.milestones == (150000, 220000)
    assert effective.max_iter == 250000
    assert effective.scheduler_interval == "step"
    assert effective.num_gpus == 1
    assert effective.world_size == 1
    assert effective.gradient_accumulation_steps == 1
    assert effective.eval_period == 5000
    assert effective.batch_size == 16
    assert effective.num_workers == 0
    assert effective.filter_empty_annotations is False
    assert effective.random_repeat_permutation is True
    assert effective.box_renewal is True
    assert effective.use_ensemble is True


def test_s0_package_topology_is_genuine_and_key_map_is_exhaustive() -> None:
    config = RADMConfig(
        num_classes=4,
        num_proposals=100,
        hidden_dim=256,
        text_feature_dim=768,
    )
    model = RADMDenoiser(config=config)

    assert_radm_package_topology(model, effective_radm_config())
    mapping = build_state_key_map(model)
    assert mapping
    assert set(mapping) == set(model.state_dict())
    assert mapping["backbone.body.body.layer1.0.conv1.weight"] == (
        "backbone.bottom_up.res2.0.conv1.weight"
    )
    assert mapping["head.blocks.0.self_attn.in_proj_weight"] == (
        "head.head_series.0.self_attn.in_proj_weight"
    )


def test_s0_topology_rejects_simplified_models() -> None:
    model = RADMDenoiser(
        config=RADMConfig(
            num_classes=4,
            num_proposals=2,
            hidden_dim=8,
            text_feature_dim=4,
            backbone_depth=18,
        )
    )

    with pytest.raises(AssertionError, match="num_proposals|hidden_dim"):
        assert_radm_package_topology(model, effective_radm_config())


def test_s0_optimizer_and_scheduler_use_step_cadence_defaults() -> None:
    effective = effective_radm_config()
    model = RADMDenoiser(
        config=RADMConfig(
            num_classes=4,
            num_proposals=2,
            hidden_dim=8,
            text_feature_dim=4,
            backbone_depth=18,
        )
    )
    optimizer = build_radm_optimizer(model, effective)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == effective.betas
    assert optimizer.defaults["eps"] == effective.eps
    assert all(
        group["lr"] == effective.learning_rate for group in optimizer.param_groups
    )
    assert all(
        group["weight_decay"] == effective.weight_decay
        for group in optimizer.param_groups
    )
    scheduler = build_radm_scheduler(optimizer, effective)
    assert scheduler.last_epoch == 0
    assert all(
        lr == effective.learning_rate * effective.warmup_factor
        for lr in scheduler.get_last_lr()
    )


def test_s0_backbone_freeze_matches_reference_optimizer_groups() -> None:
    effective = effective_radm_config()
    assert effective.backbone_freeze_at == 2
    model = RADMDenoiser(
        config=RADMConfig(
            num_classes=effective.num_classes,
            num_proposals=effective.num_proposals,
            hidden_dim=effective.hidden_dim,
            text_feature_dim=effective.text_feature_dim,
            backbone_depth=effective.backbone_depth,
            backbone_freeze_at=effective.backbone_freeze_at,
        )
    )

    assert not model.backbone.body.body.conv1.weight.requires_grad
    assert not model.backbone.body.body.layer1[0].conv1.weight.requires_grad
    assert model.backbone.body.body.layer2[0].conv1.weight.requires_grad
    optimizer = build_radm_optimizer(model, effective)
    assert len(optimizer.param_groups) == 434


def test_s0_topology_comparison_rejects_shape_mismatch() -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        compare_state_dict_topology(
            {"reference.weight": torch.zeros(2)},
            {"package.weight": torch.zeros(3)},
            {"package.weight": "reference.weight"},
        )


def test_s0_roi_pooler_uses_detectron2_canonical_fpn_levels() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 56.0, 56.0],
            [0.0, 0.0, 112.0, 112.0],
            [0.0, 0.0, 224.0, 224.0],
            [0.0, 0.0, 448.0, 448.0],
        ]
    ).reshape(1, 4, 4)

    levels = RADMProposalHead._assign_pooler_levels(boxes)

    assert levels.tolist() == [[0, 1, 2, 3]]


def test_s0_repeated_heads_detach_only_the_next_head_boxes() -> None:
    config = RADMConfig(
        num_classes=2,
        num_proposals=2,
        hidden_dim=8,
        text_feature_dim=4,
        num_heads=2,
        num_attention_heads=2,
        dim_feedforward=16,
        num_cls=1,
        num_reg=1,
        backbone_depth=18,
        with_vtram=False,
        with_gram=False,
    )
    model = RADMDenoiser(config=config).train()
    seen_next_input: list[bool] = []

    def record_next_input(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
        seen_next_input.append(cast(torch.Tensor, args[2]).requires_grad)

    hook = model.head.blocks[1].register_forward_pre_hook(record_next_input)
    try:
        output = model(
            boxes_xyxy=torch.tensor([[[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]]),
            timesteps=torch.tensor([3]),
            text_features=torch.zeros(1, 2, 4),
            text_mask=torch.ones(1, 2, 1, dtype=torch.bool),
            images=torch.zeros(1, 3, 64, 64),
        )
    finally:
        hook.remove()

    assert output.auxiliary_boxes_xyxy is not None
    assert output.auxiliary_boxes_xyxy[-1].requires_grad
    assert seen_next_input == [False]


def test_s0_loss_exposes_dynamic_k_focal_l1_giou_and_auxiliary_branches() -> None:
    lightning_module = pytest.importorskip("radm.training.lightning_module")
    radm_loss = lightning_module._radm_loss
    output = RADMDenoiserOutput(
        logits=torch.zeros(1, 4, 2, requires_grad=True),
        boxes_xyxy=torch.tensor(
            [
                [
                    [0.1, 0.1, 0.4, 0.4],
                    [0.2, 0.2, 0.5, 0.5],
                    [0.3, 0.3, 0.6, 0.6],
                    [0.4, 0.4, 0.7, 0.7],
                ]
            ],
            requires_grad=True,
        ),
        pred_original_sample=torch.zeros(1, 4, 4),
        pred_noise=torch.zeros(1, 4, 4),
        auxiliary_logits=torch.zeros(2, 1, 4, 2, requires_grad=True),
        auxiliary_boxes_xyxy=torch.tensor(
            [
                [
                    [
                        [0.1, 0.1, 0.4, 0.4],
                        [0.2, 0.2, 0.5, 0.5],
                        [0.3, 0.3, 0.6, 0.6],
                        [0.4, 0.4, 0.7, 0.7],
                    ]
                ],
                [
                    [
                        [0.1, 0.1, 0.4, 0.4],
                        [0.2, 0.2, 0.5, 0.5],
                        [0.3, 0.3, 0.6, 0.6],
                        [0.4, 0.4, 0.7, 0.7],
                    ]
                ],
            ],
            requires_grad=True,
        ),
    )
    target = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.25, 0.25, 0.3, 0.3]]),
        "boxes_xyxy": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
        "image_size_xyxy": torch.tensor([100.0, 100.0, 100.0, 100.0]),
        "image_size_xyxy_tgt": torch.tensor([[100.0, 100.0, 100.0, 100.0]]),
    }

    losses = radm_loss(
        output,
        [target],
        num_classes=2,
        alpha=0.25,
        gamma=2.0,
        ota_k=2,
        class_weight=5.0,
        l1_weight=1.0,
        giou_weight=1.0,
        no_object_weight=0.1,
    )

    assert set(losses) == {
        "loss_ce_0",
        "loss_bbox_0",
        "loss_giou_0",
        "loss_ce",
        "loss_bbox",
        "loss_giou",
    }
    sum(losses.values()).backward()


def test_s0_mapper_transform_and_collator_preserve_effective_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective = effective_radm_config()
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(np.random, "choice", lambda values: 480)
    image, boxes = _apply_training_transforms(
        torch.zeros(3, 100, 200),
        torch.tensor([[0.1, 0.2, 0.4, 0.8]]),
        effective=effective,
    )
    assert tuple(image.shape) == (3, 480, 960)
    torch.testing.assert_close(boxes, torch.tensor([[0.6, 0.2, 0.9, 0.8]]))

    examples = [
        {
            "image": torch.zeros(3, 32, 48),
            "boxes_xyxy": torch.zeros(1, 4),
            "labels": torch.tensor([0]),
            "text_features": torch.zeros(20, 768),
            "text_mask": torch.ones(20, 1, dtype=torch.bool),
            "image_size_xyxy": torch.tensor([48.0, 32.0, 48.0, 32.0]),
        },
        {
            "image": torch.zeros(3, 64, 32),
            "boxes_xyxy": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "text_features": torch.zeros(20, 768),
            "text_mask": torch.zeros(20, 1, dtype=torch.bool),
            "image_size_xyxy": torch.tensor([32.0, 64.0, 32.0, 64.0]),
        },
    ]
    batch = RADMDataCollator(effective=effective)(
        cast(list[RADMTrainingExample], examples)
    )
    assert tuple(batch["images"].shape) == (2, 3, 64, 64)
    assert batch["image_scales"].tolist() == [
        [48.0, 32.0, 48.0, 32.0],
        [32.0, 64.0, 32.0, 64.0],
    ]
