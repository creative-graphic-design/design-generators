"""Vendor checkpoint extraction and state-dict conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from typing import TypedDict, cast

import torch

from .configuration_layout_detr import LayoutDetrConfig
from .modeling_layout_detr import LayoutDetrForConditionalGeneration


class LayoutDetrConversionReport(TypedDict):
    """Structured conversion metadata persisted with converted checkpoints."""

    source_key_count: int
    target_key_count: int
    loaded_key_count: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    mismatched_shapes: list[tuple[str, tuple[int, ...], tuple[int, ...]]]
    custom_op_import_required: bool


def remap_generator_key(source_key: str) -> str:
    """Map a vendor ``G_ema`` key to the local model key when possible."""
    if source_key.startswith("module."):
        source_key = source_key.removeprefix("module.")
    exact = {
        "emb_label.weight": "emb_label.weight",
        "fc_z.weight": "fc_z.weight",
        "fc_z.bias": "fc_z.bias",
        "bbox_embed.layers.0.weight": "bbox_embed.0.weight",
        "bbox_embed.layers.0.bias": "bbox_embed.0.bias",
        "bbox_embed.layers.2.weight": "bbox_embed.2.weight",
        "bbox_embed.layers.2.bias": "bbox_embed.2.bias",
    }
    return exact.get(source_key, source_key)


def build_conversion_report(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    remapped_state: Mapping[str, torch.Tensor],
    *,
    custom_op_import_required: bool,
) -> LayoutDetrConversionReport:
    """Build strict-load diagnostics for a remapped state dict."""
    missing = sorted(set(target_state).difference(remapped_state))
    unexpected = sorted(set(remapped_state).difference(target_state))
    mismatched = []
    loaded = 0
    for key, tensor in remapped_state.items():
        if key not in target_state:
            continue
        target_shape = tuple(target_state[key].shape)
        source_shape = tuple(tensor.shape)
        if source_shape != target_shape:
            mismatched.append((key, source_shape, target_shape))
        else:
            loaded += 1
    return {
        "source_key_count": len(source_state),
        "target_key_count": len(target_state),
        "loaded_key_count": loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_shapes": mismatched,
        "custom_op_import_required": custom_op_import_required,
    }


def extract_generator_state(
    pickle_path: str | Path,
    *,
    vendor_root: str | Path,
    device: str = "cpu",
) -> tuple[
    dict[str, torch.Tensor], LayoutDetrConfig, LayoutDetrConversionReport
]:  # pragma: no cover
    """Extract ``G_ema`` from the original LayoutDETR pickle.

    The import is isolated to the conversion path. Normal converted
    ``from_pretrained`` inference never imports ``vendor/layout-detr`` or
    ``torch_utils.ops``.
    """
    vendor_path = Path(vendor_root).resolve()
    custom_op_import_required = False
    before_modules = set(sys.modules)
    with temporary_sys_path(vendor_path):
        import dnnlib  # type: ignore[import-not-found]
        import legacy  # type: ignore[import-not-found]

        with dnnlib.util.open_url(str(pickle_path)) as handle:
            generator = legacy.load_network_pkl(handle)["G_ema"].to(device)
    custom_op_import_required = any(
        name.startswith("torch_utils.ops")
        for name in set(sys.modules).difference(before_modules)
    )
    source_state = {
        key: value.detach().cpu()
        for key, value in cast(
            Mapping[str, torch.Tensor], generator.state_dict()
        ).items()
    }
    init_kwargs = dict(getattr(generator, "init_kwargs", {}) or {})
    config = LayoutDetrConfig(
        z_dim=int(getattr(generator, "z_dim", init_kwargs.get("z_dim", 4))),
        max_text_length=int(
            getattr(
                generator, "max_text_length", init_kwargs.get("max_text_length", 256)
            )
        ),
        original_training_options=init_kwargs,
    )
    target_state = LayoutDetrForConditionalGeneration(config).state_dict()
    remapped = {remap_generator_key(key): value for key, value in source_state.items()}
    report = build_conversion_report(
        source_state,
        target_state,
        remapped,
        custom_op_import_required=custom_op_import_required,
    )
    config.conversion_report = dict(report)
    return remapped, config, report


def strict_load_converted_state(
    model: LayoutDetrForConditionalGeneration,
    state: Mapping[str, torch.Tensor],
) -> LayoutDetrConversionReport:
    """Strict-load a remapped state dict and return diagnostics."""
    report = build_conversion_report(
        state,
        model.state_dict(),
        state,
        custom_op_import_required=False,
    )
    if (
        report["missing_keys"]
        or report["unexpected_keys"]
        or report["mismatched_shapes"]
    ):
        raise RuntimeError(f"LayoutDETR state dict is not strict-loadable: {report}")
    model.load_state_dict(dict(state), strict=True)
    return report


@contextmanager
def temporary_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend a vendor path during conversion-only imports."""
    raw = str(path)
    sys.path.insert(0, raw)
    try:
        yield
    finally:
        try:
            sys.path.remove(raw)
        except ValueError:
            pass
