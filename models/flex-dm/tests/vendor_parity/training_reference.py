"""Build the pinned Flex-DM training reference for live comparisons."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal, cast

from laygen.common.testing import skip_or_fail_vendor_parity

DatasetName = Literal["crello", "rico"]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "flex-dm"


def cache_root() -> Path:
    """Return the repository-relative parity cache root."""
    return _REPO_ROOT / ".cache" / "flex-dm"


def original_asset_root() -> Path:
    """Return the extracted original weights and data root."""
    return Path(os.environ.get("FLEX_DM_ORIGINAL_ASSET_DIR", cache_root() / "original"))


def original_variant_root(dataset: DatasetName, variant: str = "ours-exp-ft") -> Path:
    """Resolve one released run directory."""
    root = original_asset_root() / "weights" / dataset
    nested = root / dataset / variant
    return nested if nested.is_dir() else root / variant


def original_args(
    dataset: DatasetName, variant: str = "ours-exp-ft"
) -> dict[str, object]:
    """Load the released arguments for one dataset and run variant."""
    path = original_variant_root(dataset, variant) / "args.json"
    if not path.is_file():
        skip_or_fail_vendor_parity(
            "missing released Flex-DM training args",
            missing_paths=[path],
            regeneration_hint="set FLEX_DM_ORIGINAL_ASSET_DIR to the extracted original assets",
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(dict[str, object], value)


def original_source(relative_path: str) -> str:
    """Read a source file from the initialized pinned Flex-DM submodule."""
    path = _VENDOR_ROOT / "src" / "mfp" / "mfp" / relative_path
    if not path.is_file():
        skip_or_fail_vendor_parity(
            "missing pinned Flex-DM training source",
            missing_paths=[path],
            regeneration_hint="run git submodule update --init vendor/flex-dm",
        )
    return path.read_text(encoding="utf-8")


def build_original(
    dataset: DatasetName,
    *,
    masking_method: str | None = None,
    batch_size: int = 1,
    restore_checkpoint: bool = True,
):
    """Construct and restore the pinned vendor MFP and its data specification."""
    if not (_VENDOR_ROOT / "src" / "mfp").is_dir():
        skip_or_fail_vendor_parity(
            "missing pinned Flex-DM source tree",
            missing_paths=[_VENDOR_ROOT / "src" / "mfp"],
            regeneration_hint="run git submodule update --init vendor/flex-dm",
        )
    for path in (str(_VENDOR_ROOT), str(_VENDOR_ROOT / "src" / "mfp")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from traingen_parity.tensorflow_compat import install_keras_preprocessing_compat

    install_keras_preprocessing_compat()

    from mfp.data import DataSpec
    from mfp.models.mfp import MFP

    args = original_args(dataset)
    data_root = original_asset_root() / "data" / dataset
    if (data_root / dataset).is_dir():
        data_root /= dataset
    if not data_root.is_dir():
        skip_or_fail_vendor_parity(
            "missing original Flex-DM preprocessed data",
            missing_paths=[data_root],
            regeneration_hint="set FLEX_DM_ORIGINAL_ASSET_DIR to the extracted original assets",
        )

    dataspec = DataSpec(dataset, str(data_root), batch_size=batch_size)
    input_columns = dataspec.make_input_columns()
    model = MFP(
        input_columns,
        num_blocks=int(cast(int, args["num_blocks"])),
        block_type=str(args["block_type"]),
        masking_method=masking_method or str(args["masking_method"]),
        seq_type=str(args["seq_type"]),
        arch_type=str(args["arch_type"]),
        context=cast(str | None, args["context"]),
        latent_dim=int(cast(int, args["latent_dim"])),
        dropout=float(cast(float, args.get("dropout", 0.0))),
        l2=float(cast(float, args.get("l2", 0.0))),
        input_dtype=str(args["input_dtype"]),
    )
    if restore_checkpoint:
        checkpoint = original_variant_root(dataset) / "checkpoints" / "best.ckpt"
        if not checkpoint.with_suffix(".ckpt.index").is_file():
            skip_or_fail_vendor_parity(
                "missing released Flex-DM checkpoint",
                missing_paths=[checkpoint.with_suffix(".ckpt.index")],
                regeneration_hint="set FLEX_DM_ORIGINAL_ASSET_DIR to the extracted original assets",
            )
        model.load_weights(str(checkpoint))
    return dataspec, model, args
