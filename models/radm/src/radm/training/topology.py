"""S0 topology guards for the RADM package model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Set
from typing import Final, Protocol, cast

import torch
from jaxtyping import Shaped
from torch import nn

from ..configuration_radm import RADMConfig, RADM_DIFFUSION_STATE_KEYS
from ..modeling_radm import RADMDenoiser, RADMDenoiserOutput
from .config import RADMEffectiveConfig
from .dataset import (
    RADM_CROP_TRANSFORM_NAMES,
    RADM_TEXT_ENCODING_SUMMARY,
    RADM_TRAIN_TRANSFORM_NAMES,
)


REVIEWED_REFERENCE_STATE_ALLOWLIST: Final[frozenset[str]] = frozenset()
"""Reference state keys intentionally absent from the package model.

This set is deliberately empty for the active recipe. Adding a key requires a
reviewed conversion rule and a test explaining why it is not represented in
the package topology.
"""

_FPN_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("backbone.body.fpn.inner_blocks.0.0.", "backbone.fpn_lateral2."),
    ("backbone.body.fpn.inner_blocks.1.0.", "backbone.fpn_lateral3."),
    ("backbone.body.fpn.inner_blocks.2.0.", "backbone.fpn_lateral4."),
    ("backbone.body.fpn.inner_blocks.3.0.", "backbone.fpn_lateral5."),
    ("backbone.body.fpn.layer_blocks.0.0.", "backbone.fpn_output2."),
    ("backbone.body.fpn.layer_blocks.1.0.", "backbone.fpn_output3."),
    ("backbone.body.fpn.layer_blocks.2.0.", "backbone.fpn_output4."),
    ("backbone.body.fpn.layer_blocks.3.0.", "backbone.fpn_output5."),
)


class _ReferenceState(Protocol):
    effective: RADMEffectiveConfig
    model: nn.Module

    @property
    def runtime_summary(
        self,
    ) -> Mapping[str, str | int | float | bool | tuple[str, ...] | tuple[int, ...]]: ...


def build_reviewed_state_key_map(
    reference_model: nn.Module,
    package_model: RADMDenoiser,
) -> dict[str, str]:
    """Build the explicit, reviewed package-to-reference state-key map.

    The only accepted namespace differences are the backbone FPN names and the
    six-head container name. No map is inferred from matching shapes or from
    the package model alone.
    """
    del reference_model
    result = {
        package_key: _package_key_to_reference_key(package_key)
        for package_key in package_model.state_dict()
    }
    if len(result) != len(package_model.state_dict()):
        raise AssertionError("Reviewed RADM key map is not exhaustive")
    if len(set(result.values())) != len(result):
        raise AssertionError("Reviewed RADM key map is not one-to-one")
    return result


def build_state_key_map(model: RADMDenoiser) -> dict[str, str]:
    """Compatibility wrapper kept for static map inspection.

    The real S0 guard calls :func:`build_reviewed_state_key_map` with both
    instantiated models and then validates every reference key.
    """
    return {key: _package_key_to_reference_key(key) for key in model.state_dict()}


def _package_key_to_reference_key(key: str) -> str:
    """Translate one explicitly reviewed package state namespace."""
    if key in RADM_DIFFUSION_STATE_KEYS:
        return key
    if key.startswith("head.time_mlp."):
        return key
    if key.startswith("head.blocks."):
        return key.replace("head.blocks.", "head.head_series.", 1)
    for package_prefix, reference_prefix in _FPN_RULES:
        if key.startswith(package_prefix):
            return reference_prefix + key[len(package_prefix) :]
    if key.startswith("backbone.body.body."):
        reference = key.replace("backbone.body.body.", "backbone.bottom_up.", 1)
        reference = reference.replace(
            "backbone.bottom_up.conv1.", "backbone.bottom_up.stem.conv1.", 1
        )
        reference = reference.replace(
            "backbone.bottom_up.bn1.", "backbone.bottom_up.stem.conv1.norm.", 1
        )
        for package_layer, reference_stage in (
            ("layer1", "res2"),
            ("layer2", "res3"),
            ("layer3", "res4"),
            ("layer4", "res5"),
        ):
            reference = reference.replace(
                f"backbone.bottom_up.{package_layer}.",
                f"backbone.bottom_up.{reference_stage}.",
                1,
            )
        reference = reference.replace(".downsample.0.", ".shortcut.")
        reference = reference.replace(".downsample.1.", ".shortcut.norm.")
        reference = reference.replace(".bn1.", ".conv1.norm.")
        reference = reference.replace(".bn2.", ".conv2.norm.")
        reference = reference.replace(".bn3.", ".conv3.norm.")
        return reference
    raise AssertionError(f"Unreviewed RADM topology key: {key}")


def assert_radm_package_topology(
    model: RADMDenoiser,
    effective: RADMEffectiveConfig,
) -> None:
    """Check package-side static configuration before reference comparison."""
    config = model.radm_config
    assert isinstance(config, RADMConfig), "model must carry RADMConfig"
    assert config.original_id2label == effective.class_id_to_label, (
        "class mapping mismatch: package config does not preserve the captured "
        "five-label vocabulary"
    )
    assert {
        index: config.original_id2label[index] for index in range(config.num_classes)
    } == effective.predicted_class_id_to_label, "predicted class mapping mismatch"
    assert len(config.original_id2label) == effective.vocabulary_size, (
        "class vocabulary size mismatch"
    )
    assert len(model.head.blocks) == effective.num_heads, "num_heads mismatch"
    assert config.num_proposals == effective.num_proposals, "num_proposals mismatch"
    assert config.num_classes == effective.num_classes, "num_classes mismatch"
    assert config.hidden_dim == effective.hidden_dim, "hidden_dim mismatch"
    assert config.text_feature_dim == effective.text_feature_dim, (
        "text_feature_dim mismatch"
    )
    assert config.max_text_num == effective.max_text_num, "max_text_num mismatch"
    assert config.num_attention_heads == effective.num_attention_heads, (
        "num_attention_heads mismatch"
    )
    assert config.dim_feedforward == effective.dim_feedforward, (
        "dim_feedforward mismatch"
    )
    assert config.num_dynamic == effective.num_dynamic, "num_dynamic mismatch"
    assert config.dim_dynamic == effective.dim_dynamic, "dim_dynamic mismatch"
    assert config.num_cls == effective.num_cls, "num_cls mismatch"
    assert config.num_reg == effective.num_reg, "num_reg mismatch"
    assert config.roi_resolution == effective.roi_resolution, "roi_resolution mismatch"
    assert config.roi_sampling_ratio == effective.roi_sampling_ratio, (
        "roi_sampling_ratio mismatch"
    )
    assert config.backbone_depth == effective.backbone_depth, "backbone_depth mismatch"
    assert config.backbone_freeze_at == effective.backbone_freeze_at, (
        "backbone_freeze_at mismatch"
    )
    assert config.with_vtram is effective.with_vtram, "with_vtram mismatch"
    assert config.with_gram is effective.with_gram, "with_gram mismatch"
    assert config.deep_supervision is effective.deep_supervision, (
        "deep_supervision mismatch"
    )
    assert config.num_train_timesteps == effective.num_train_timesteps, (
        "num_train_timesteps mismatch"
    )
    assert config.snr_scale == effective.snr_scale, "snr_scale mismatch"


def compare_state_dict_topology(
    reference: Mapping[str, Shaped[torch.Tensor, "..."]],
    package: Mapping[str, Shaped[torch.Tensor, "..."]],
    key_map: Mapping[str, str],
    *,
    allowlist: Set[str] = REVIEWED_REFERENCE_STATE_ALLOWLIST,
) -> None:
    """Compare mapped keys/shapes and reject every unreviewed reference key."""
    package_keys = set(package)
    mapped_package_keys = set(key_map)
    if package_keys != mapped_package_keys:
        raise AssertionError(
            f"Package keys missing from reviewed map: {sorted(package_keys - mapped_package_keys)[:8]}"
        )
    reference_keys = set(reference)
    mapped_reference_keys = set(key_map.values())
    missing = sorted(mapped_reference_keys - reference_keys)
    if missing:
        raise AssertionError(f"Missing reference topology keys: {missing[:8]}")
    invalid_allowlist = sorted(set(allowlist) & mapped_reference_keys)
    if invalid_allowlist:
        raise AssertionError(
            f"Allowlisted mapped keys are not extras: {invalid_allowlist}"
        )
    unexpected = sorted(reference_keys - mapped_reference_keys - set(allowlist))
    if unexpected:
        raise AssertionError(f"Unexpected reference topology keys: {unexpected[:8]}")
    for package_key, reference_key in key_map.items():
        if tuple(package[package_key].shape) != tuple(reference[reference_key].shape):
            raise AssertionError(
                f"Topology shape mismatch for {package_key} -> {reference_key}: "
                f"{tuple(package[package_key].shape)} != {tuple(reference[reference_key].shape)}"
            )


def assert_radm_topology_parity(
    reference_model: nn.Module,
    package_model: RADMDenoiser,
    key_map: Mapping[str, str],
    *,
    allowlist: Set[str] = REVIEWED_REFERENCE_STATE_ALLOWLIST,
) -> None:
    """Compare real-model parameter counts and exhaustive state topology."""
    reference_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    package_parameters = sum(
        parameter.numel() for parameter in package_model.parameters()
    )
    if reference_parameters != package_parameters:
        raise AssertionError(
            "RADM parameter-count mismatch: "
            f"{package_parameters} != {reference_parameters}"
        )
    compare_state_dict_topology(
        reference_model.state_dict(),
        package_model.state_dict(),
        key_map,
        allowlist=allowlist,
    )


def copy_reviewed_state_dict(
    reference_model: nn.Module,
    package_model: RADMDenoiser,
    key_map: Mapping[str, str],
    *,
    allowlist: Set[str] = REVIEWED_REFERENCE_STATE_ALLOWLIST,
) -> None:
    """Copy every mapped reference tensor into the package model."""
    compare_state_dict_topology(
        reference_model.state_dict(),
        package_model.state_dict(),
        key_map,
        allowlist=allowlist,
    )
    reference_state = reference_model.state_dict()
    package_state = package_model.state_dict()
    with torch.no_grad():
        for package_key, reference_key in key_map.items():
            package_state[package_key].copy_(reference_state[reference_key])


def assert_forward_parity(
    reference_output: Mapping[str, Shaped[torch.Tensor, "..."]],
    package_output: RADMDenoiserOutput,
) -> None:
    """Compare the copied-weight same-input head outputs."""
    reference_logits = reference_output["auxiliary_logits"]
    reference_boxes = reference_output["auxiliary_boxes_xyxy"]
    if (
        package_output.auxiliary_logits is None
        or package_output.auxiliary_boxes_xyxy is None
    ):
        raise AssertionError("package forward did not return all head outputs")
    torch.testing.assert_close(
        package_output.auxiliary_logits,
        reference_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        package_output.auxiliary_boxes_xyxy,
        reference_boxes,
        rtol=0,
        atol=0,
    )
    for output_name in ("pred_original_sample", "pred_noise"):
        reference_value = reference_output[output_name]
        package_value = getattr(package_output, output_name)
        try:
            torch.testing.assert_close(
                package_value,
                reference_value,
                rtol=0,
                atol=0,
            )
        except AssertionError as error:
            raise AssertionError(f"{output_name} mismatch: {error}") from error


def assert_effective_runtime_state(
    state: _ReferenceState, package_model: RADMDenoiser
) -> None:
    """Check runtime branches, diffusion buffers, and package static values."""
    effective = state.effective
    config = package_model.radm_config
    assert config.num_train_timesteps == effective.num_train_timesteps
    assert config.snr_scale == effective.snr_scale
    assert config.sample_step == effective.sample_step
    assert config.with_vtram is effective.with_vtram
    assert config.with_gram is effective.with_gram
    assert config.deep_supervision is effective.deep_supervision
    assert config.backbone_freeze_at == effective.backbone_freeze_at
    assert effective.optimizer == state.runtime_summary["optimizer"]
    assert effective.scheduler_interval == state.runtime_summary["scheduler_interval"]
    assert effective.box_renewal is state.runtime_summary["box_renewal"]
    assert effective.use_ensemble is state.runtime_summary["use_ensemble"]
    assert effective.ema_enabled is state.runtime_summary["ema_enabled"]
    assert effective.amp_enabled is state.runtime_summary["amp_enabled"]
    assert effective.ddp_enabled is state.runtime_summary["ddp_enabled"]
    assert effective.simple_trainer is state.runtime_summary["simple_trainer"]
    assert effective.transform_names == cast(
        tuple[str, ...], state.runtime_summary["transform_names"]
    )
    assert effective.transform_names == RADM_TRAIN_TRANSFORM_NAMES
    assert effective.crop_transform_names == tuple(
        cast(tuple[str, ...], state.runtime_summary["crop_transform_names"])
    )
    assert effective.min_size_train == cast(
        tuple[int, ...], state.runtime_summary["min_size_train"]
    )
    assert effective.max_size_train == cast(
        int, state.runtime_summary["max_size_train"]
    )
    assert effective.min_size_train_sampling == cast(
        str, state.runtime_summary["min_size_train_sampling"]
    )
    assert effective.crop_transform_names == RADM_CROP_TRANSFORM_NAMES
    assert state.runtime_summary["text_feature_dim"] == 768
    assert state.runtime_summary["max_text_num"] == 20
    assert {
        "mask_semantics": state.runtime_summary["text_mask_semantics"],
        "missing_fallback": state.runtime_summary["missing_text_fallback"],
    } == RADM_TEXT_ENCODING_SUMMARY
    for key in RADM_DIFFUSION_STATE_KEYS:
        reference_buffer = state.model.state_dict()[key]
        package_buffer = package_model.state_dict()[key]
        torch.testing.assert_close(package_buffer, reference_buffer, rtol=0, atol=0)


def assert_optimizer_scheduler_parity(
    reference_optimizer: torch.optim.Optimizer,
    package_optimizer: torch.optim.Optimizer,
    reference_scheduler: torch.optim.lr_scheduler.LRScheduler,
    package_scheduler: torch.optim.lr_scheduler.LRScheduler,
    effective: RADMEffectiveConfig,
) -> None:
    """Compare live optimizer and scheduler state without replacing the model."""
    assert isinstance(reference_optimizer, torch.optim.AdamW)
    assert isinstance(package_optimizer, torch.optim.AdamW)
    for name in ("betas", "eps"):
        assert reference_optimizer.defaults[name] == package_optimizer.defaults[name]
    assert len(reference_optimizer.param_groups) == len(package_optimizer.param_groups)
    for reference_group, package_group in zip(
        reference_optimizer.param_groups,
        package_optimizer.param_groups,
        strict=True,
    ):
        assert len(reference_group["params"]) == len(package_group["params"])
        assert reference_group["lr"] == package_group["lr"]
        assert reference_group["weight_decay"] == package_group["weight_decay"]

    reference_state = reference_scheduler.state_dict()
    package_state = package_scheduler.state_dict()
    assert reference_state["last_epoch"] == package_state["last_epoch"]
    torch.testing.assert_close(
        torch.tensor(reference_state["base_lrs"]),
        torch.tensor(package_state["base_lrs"]),
        rtol=0,
        atol=0,
    )
    assert package_state["milestones"] == Counter(effective.milestones)
    assert package_state["gamma"] == effective.scheduler_gamma
    assert package_state["warmup_factor"] == effective.warmup_factor
    assert package_state["warmup_iters"] == effective.warmup_iters
    assert package_state["warmup_method"] == "linear"
    torch.testing.assert_close(
        torch.tensor(reference_scheduler.get_last_lr()),
        torch.tensor(package_scheduler.get_last_lr()),
        rtol=1e-12,
        atol=1e-15,
    )
    reference_last_epoch = reference_scheduler.last_epoch
    package_last_epoch = package_scheduler.last_epoch
    try:
        for step in (
            0,
            max(effective.warmup_iters - 1, 0),
            effective.warmup_iters,
            *(milestone for milestone in effective.milestones),
        ):
            reference_scheduler.last_epoch = step
            package_scheduler.last_epoch = step
            torch.testing.assert_close(
                torch.tensor(reference_scheduler.get_lr()),
                torch.tensor(package_scheduler.get_lr()),
                rtol=1e-12,
                atol=1e-15,
            )
    finally:
        reference_scheduler.last_epoch = reference_last_epoch
        package_scheduler.last_epoch = package_last_epoch
