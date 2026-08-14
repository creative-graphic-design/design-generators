"""Independent reference adapters for the RALF training stages.

This module is only loaded by the gated vendor-parity training probes. The
package model and package data path remain independent of these adapters.
"""

from __future__ import annotations

import importlib
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Sequence as DatasetSequence, Value
from jaxtyping import Shaped
from torch import Tensor

from ralf import RalfConfig
from ralf.retrieval import RalfRetrievedBatch
from ralf.training.datamodule import collate_training_batch


def require_vendor(cache_dir: Path) -> None:
    """Configure and validate the independent reference dependencies."""
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("PARITY_REQUIRE=1 is required for training parity")
    vendor_root = Path(__file__).parents[4] / "vendor" / "ralf"
    if not vendor_root.exists():
        raise FileNotFoundError(
            f"pinned RALF vendor checkout is missing: {vendor_root}"
        )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    os.environ["RALF_CACHE_DIR"] = str(cache_dir)
    precomputed = cache_dir / "PRECOMPUTED_WEIGHT_DIR"
    for module_name in (
        "image2layout.train.fid.model",
        "image2layout.train.models.common.image",
        "image2layout.train.helpers.layout_tokenizer",
    ):
        module = importlib.import_module(module_name)
        module.PRECOMPUTED_WEIGHT_DIR = str(precomputed)


def _vendor_features(config: RalfConfig) -> Features:
    labels = [str(label) for _, label in sorted(config.id2label.items())]
    return Features(
        {
            "id": Value("string"),
            "label": DatasetSequence(ClassLabel(names=labels)),
        }
    )


def build_vendor_model(config: RalfConfig, *, cache_dir: Path) -> torch.nn.Module:
    """Construct the pinned original model without injecting it into package code."""
    require_vendor(cache_dir)
    from image2layout.train.models.retrieval_augmented_autoreg import (
        ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
    )
    from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer

    features = _vendor_features(config)
    tokenizer = LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=config.max_seq_length,
        num_bin=config.num_bin,
        var_order=list(config.var_order),
        special_tokens=list(config.special_tokens),
        is_loc_vocab_shared=config.is_loc_vocab_shared,
        geo_quantization=config.geo_quantization,
    )
    dataset_name = "pku" if config.dataset_name.startswith("pku") else "cgl"
    return ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
        features=features,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_seq_length=config.max_seq_length,
        db_dataset=Dataset.from_dict({"id": []}),
        d_model=config.d_model,
        decoder_d_model=config.decoder_d_model,
        top_k=config.top_k,
        layout_backbone=config.layout_backbone,
        use_reference_image=config.use_reference_image,
        freeze_layout_encoder=config.freeze_layout_encoder,
        retrieval_backbone=config.retrieval_backbone,
        random_retrieval=False,
        saliency_k="None",
        auxilary_task="uncond",
        use_flag_embedding=config.use_flag_embedding,
        use_multitask=config.use_multitask,
        RELATION_SIZE=config.relation_size,
        global_task_embedding=config.global_task_embedding,
    )


def move_retrieved(
    retrieved: RalfRetrievedBatch, device: torch.device
) -> RalfRetrievedBatch:
    """Move an explicit retrieval batch to a device."""
    return RalfRetrievedBatch(
        image=retrieved.image.to(device),
        saliency=retrieved.saliency.to(device),
        bbox=retrieved.bbox.to(device),
        labels=retrieved.labels.to(device),
        mask=retrieved.mask.to(device),
        indexes=None if retrieved.indexes is None else retrieved.indexes.to(device),
    )


def vendor_raw_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Convert package batch fields to the original model's raw batch schema."""
    retrieved = cast(RalfRetrievedBatch, batch["retrieved"])
    retrieved_image = torch.cat([retrieved.image, retrieved.saliency], dim=2)
    bbox = retrieved.bbox
    retrieved_data = {
        "image": retrieved_image,
        "saliency": retrieved.saliency,
        "center_x": bbox[..., 0],
        "center_y": bbox[..., 1],
        "width": bbox[..., 2],
        "height": bbox[..., 3],
        "label": retrieved.labels,
        "mask": retrieved.mask,
    }
    layout_bbox = cast(Tensor, batch["layout_bbox"])
    return {
        "id": list(range(cast(Tensor, batch["layout_labels"]).size(0))),
        "image": cast(Tensor, batch["pixel_values"]),
        "saliency": cast(Tensor, batch["saliency"]),
        "label": cast(Tensor, batch["layout_labels"]),
        "center_x": layout_bbox[..., 0],
        "center_y": layout_bbox[..., 1],
        "width": layout_bbox[..., 2],
        "height": layout_bbox[..., 3],
        "mask": cast(Tensor, batch["layout_mask"]),
        "retrieved": retrieved_data,
    }


def vendor_preprocess(
    model: torch.nn.Module, batch: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Prepare a package batch through the original model's preprocessing."""
    raw = vendor_raw_batch(batch)
    inputs, targets = model.preprocess(raw)
    return cast(dict[str, object], inputs), cast(dict[str, object], targets)


def package_batch_from_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    config: RalfConfig,
    table: Mapping[int | str, Sequence[int]],
    retrieval_samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a package batch from explicit rows for stream checks."""
    from ralf.training.datamodule import RalfTrainingDataset

    dataset = RalfTrainingDataset(
        samples=samples,
        config=config,
        retrieval_table=table,
        retrieval_samples=retrieval_samples,
    )
    return collate_training_batch([dataset[index] for index in range(len(dataset))])


def reseed(seed: int) -> None:
    """Reset all RNGs used by the two single-process reference loops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_sha256(state: Mapping[str, Shaped[torch.Tensor, "..."]]) -> str:
    """Hash a state mapping with names and tensor bytes."""
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def named_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> dict[str, dict[str, Tensor]]:
    """Return optimizer state keyed by model parameter names."""
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    result: dict[str, dict[str, Tensor]] = {}
    for parameter, state in optimizer.state.items():
        name = by_id[id(parameter)]
        result[name] = {
            key: value.detach().clone()
            for key, value in state.items()
            if isinstance(value, Tensor)
        }
    return result
