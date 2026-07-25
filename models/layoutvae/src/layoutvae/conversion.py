"""Conversion helpers for LayoutVAE checkpoint artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import torch

from .configuration_layoutvae import LayoutVAEConfig
from .modeling_layoutvae import LayoutVAEModel
from .processing_layoutvae import LayoutVAEProcessor

DISALLOWED_OUTPUT_NAMES = {
    "countvae.h5",
    "bboxvae.h5",
    "pytorch_model.bin",
}


def load_original_state_dicts(
    source_root: str | Path,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Load original checkpoint state dictionaries through a pickle shim."""
    count, bbox = load_original_modules(source_root)
    return count.state_dict(), bbox.state_dict()


def load_original_modules(
    source_root: str | Path,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load original checkpoint modules through a pickle shim."""
    root = Path(source_root)
    trained = root / "TrainedModel"
    source_dir = root / "Source"
    count = _load_pickled_module(trained / "countvae.h5", source_dir)
    bbox = _load_pickled_module(trained / "bboxvae.h5", source_dir)
    return count, bbox


def _load_pickled_module(path: Path, source_dir: Path) -> torch.nn.Module:
    sys.path.insert(0, str(source_dir))
    try:
        from bboxvae import BboxVAE  # pyright: ignore[reportMissingImports]
        from countvae import CountVAE  # pyright: ignore[reportMissingImports]
        from modelblocks import Decoder, ELBOLoss, ELBOLoss_Bbox, Embeder, Encoder
        from modelblocks import EmbedBbox, Prior, Reparamatrize_cvae
        from modelblocks import ReparamatrizeMulti, Sampling, fcblock

        main = sys.modules["__main__"]
        shims = {
            "BboxVAE": BboxVAE,
            "CountVAE": CountVAE,
            "Decoder": Decoder,
            "ELBOLoss": ELBOLoss,
            "ELBOLoss_Bbox": ELBOLoss_Bbox,
            "Embeder": Embeder,
            "Encoder": Encoder,
            "EmbedBbox": EmbedBbox,
            "Prior": Prior,
            "Reparamatrize": Reparamatrize_cvae,
            "Reparamatrize_cvae": Reparamatrize_cvae,
            "ReparamatrizeMulti": ReparamatrizeMulti,
            "Sampling": Sampling,
            "fcblock": fcblock,
        }
        for name, value in shims.items():
            setattr(main, name, value)
        module = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{path} did not load as a torch.nn.Module")
        return module
    finally:
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass


def build_default_config() -> LayoutVAEConfig:
    """Return the fixed PubLayNet LayoutVAE configuration.

    Returns:
        PubLayNet LayoutVAE configuration.

    Examples:
        >>> build_default_config().dataset_name
        'publaynet'
    """
    return LayoutVAEConfig()


def convert_state_dicts(
    *,
    count_state_dict: dict[str, torch.Tensor],
    bbox_state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
) -> Path:
    """Convert count and box state dictionaries into HF files.

    Args:
        count_state_dict: State dictionary for `LayoutVAEModel.countvae`.
        bbox_state_dict: State dictionary for `LayoutVAEModel.bboxvae`.
        output_dir: Directory where converted files are written.

    Returns:
        The output directory.

    Raises:
        RuntimeError: If a state dictionary is incompatible.
        ValueError: If output filenames would violate the HF artifact contract.
    """
    target = Path(output_dir)
    if any((target / name).exists() for name in DISALLOWED_OUTPUT_NAMES):
        raise ValueError("output_dir contains non-HF checkpoint filenames")
    config = build_default_config()
    model = LayoutVAEModel(config)
    model.countvae.load_state_dict(count_state_dict, strict=True)
    model.bboxvae.load_state_dict(bbox_state_dict, strict=True)
    model.save_pretrained(target, safe_serialization=True)
    LayoutVAEProcessor(
        dataset_name=config.dataset_name,
        id2label=config.id2label,
    ).save_pretrained(str(target))
    processor_config = target / "processor_config.json"
    preprocessor_config = target / "preprocessor_config.json"
    if processor_config.exists():
        shutil.copyfile(processor_config, preprocessor_config)
    return target
