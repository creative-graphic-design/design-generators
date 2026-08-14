"""Gated adapter for the pinned original RADM training runtime."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast  # noqa: TID251 - Detectron2 CfgNode is runtime-dynamic.

import torch

from radm.training.config import (
    RADMEffectiveConfig,
)
from radm.training.topology import REVIEWED_REFERENCE_STATE_ALLOWLIST


class ReferenceUnavailable(RuntimeError):
    """Raised when the optional original runtime is not installed."""


class _VendorModel(Protocol):
    """Typed view of the registered Detectron2 meta-architecture."""

    num_timesteps: int
    box_renewal: bool
    use_ensemble: bool
    backbone: Callable[[torch.Tensor], Mapping[str, torch.Tensor]]
    head: Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class ReferenceProbe:
    """Fixed CPU probe shared by the original and package head paths."""

    package_inputs: dict[str, object]
    absolute_boxes: torch.Tensor
    normalized_boxes: torch.Tensor
    images: torch.Tensor
    text_features: torch.Tensor
    text_mask: torch.Tensor
    timesteps: torch.Tensor


@dataclass(frozen=True)
class RuntimeTextEncoding:
    """Text-feature dimensions and fallback semantics observed from the mapper."""

    feature_dim: int
    max_text_num: int
    mask_semantics: str
    missing_fallback: str


@dataclass
class ReferenceTrainingState:
    """Real initialized graph and static runtime state exposed to S0."""

    config: Any
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    effective: RADMEffectiveConfig
    runtime_summary: dict[str, Any]
    reviewed_state_allowlist: frozenset[str] = REVIEWED_REFERENCE_STATE_ALLOWLIST

    def package_model_kwargs(self) -> dict[str, object]:
        """Return package constructor values extracted from the runtime config."""
        cfg = self.config
        model = cast(_VendorModel, self.model)
        runtime_labels = cast(
            tuple[str, ...], self.runtime_summary["class_id_to_label"]
        )
        return {
            "num_classes": int(cfg.MODEL.RADM.NUM_CLASSES),
            "original_id2label": dict(enumerate(runtime_labels)),
            "num_proposals": int(cfg.MODEL.RADM.NUM_PROPOSALS),
            "hidden_dim": int(cfg.MODEL.RADM.HIDDEN_DIM),
            "text_feature_dim": self.effective.text_feature_dim,
            "max_text_num": self.effective.max_text_num,
            "num_heads": int(cfg.MODEL.RADM.NUM_HEADS),
            "num_attention_heads": int(cfg.MODEL.RADM.NHEADS),
            "dim_feedforward": int(cfg.MODEL.RADM.DIM_FEEDFORWARD),
            "num_dynamic": int(cfg.MODEL.RADM.NUM_DYNAMIC),
            "dim_dynamic": int(cfg.MODEL.RADM.DIM_DYNAMIC),
            "num_cls": int(cfg.MODEL.RADM.NUM_CLS),
            "num_reg": int(cfg.MODEL.RADM.NUM_REG),
            "roi_resolution": int(cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION),
            "roi_sampling_ratio": int(cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO),
            "with_vtram": bool(cfg.MODEL.RADM.withVTRAM),
            "with_gram": bool(cfg.MODEL.RADM.withGRAM),
            "deep_supervision": bool(cfg.MODEL.RADM.DEEP_SUPERVISION),
            "backbone_depth": int(cfg.MODEL.RESNETS.DEPTH),
            "backbone_freeze_at": int(cfg.MODEL.BACKBONE.FREEZE_AT),
            "num_train_timesteps": model.num_timesteps,
            "snr_scale": float(cfg.MODEL.RADM.SNR_SCALE),
        }

    def build_probe(self) -> ReferenceProbe:
        """Build a small deterministic input without data or checkpoint assets."""
        torch.manual_seed(261)
        proposals = int(self.config.MODEL.RADM.NUM_PROPOSALS)
        image_size = 64
        images = torch.linspace(
            -1.0, 1.0, 3 * image_size * image_size, dtype=torch.float32
        ).reshape(1, 3, image_size, image_size)
        one_box = torch.tensor([[0.125, 0.125, 0.375, 0.375]], dtype=torch.float32)
        normalized_boxes = one_box.repeat(1, proposals, 1)
        scale = normalized_boxes.new_tensor((image_size,) * 4)
        absolute_boxes = normalized_boxes * scale
        text_features = torch.linspace(
            -0.5,
            0.5,
            self.effective.max_text_num * self.effective.text_feature_dim,
            dtype=torch.float32,
        ).reshape(1, self.effective.max_text_num, self.effective.text_feature_dim)
        text_mask = torch.zeros(1, self.effective.max_text_num, 1, dtype=torch.bool)
        text_mask[:, : min(12, self.effective.max_text_num)] = True
        timesteps = torch.tensor([123], dtype=torch.long)
        package_inputs: dict[str, object] = {
            "boxes_xyxy": normalized_boxes,
            "timesteps": timesteps,
            "text_features": text_features,
            "text_mask": text_mask,
            "images": images,
        }
        return ReferenceProbe(
            package_inputs=package_inputs,
            absolute_boxes=absolute_boxes,
            normalized_boxes=normalized_boxes,
            images=images,
            text_features=text_features,
            text_mask=text_mask,
            timesteps=timesteps,
        )

    def forward_probe(self, probe: ReferenceProbe) -> dict[str, torch.Tensor]:
        """Run only the initialized original backbone/head on the fixed probe."""
        model = cast(_VendorModel, self.model)
        self.model.eval()
        with torch.no_grad():
            features = model.backbone(probe.images)
            class_outputs, box_outputs = model.head(
                [features[name] for name in self.config.MODEL.ROI_HEADS.IN_FEATURES],
                probe.absolute_boxes,
                probe.normalized_boxes,
                probe.text_features,
                probe.text_mask,
                probe.timesteps,
                None,
            )
        scale = probe.normalized_boxes.new_tensor((64.0,) * 4)
        normalized_boxes = box_outputs / scale
        return {
            "auxiliary_logits": class_outputs,
            "auxiliary_boxes_xyxy": normalized_boxes,
            "pred_original_sample": normalized_boxes[-1],
            "pred_noise": probe.normalized_boxes - normalized_boxes[-1],
        }

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe initialized-state metadata."""
        return {
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "state_dict_keys": sorted(self.model.state_dict()),
            "optimizer": {
                "class": self.optimizer.__class__.__name__,
                "defaults": _jsonable(self.optimizer.defaults),
                "param_groups": [
                    {
                        "lr": group["lr"],
                        "weight_decay": group["weight_decay"],
                        "parameter_count": len(group["params"]),
                    }
                    for group in self.optimizer.param_groups
                ],
            },
            "scheduler": {
                "class": self.scheduler.__class__.__name__,
                "state_dict": _jsonable(self.scheduler.state_dict()),
            },
            "runtime_summary": _jsonable(self.runtime_summary),
            "effective_config": self.effective.as_dict(),
        }


class RADMReferenceAdapter:
    """Construct the original registered model and training branches lazily."""

    def __init__(
        self,
        *,
        vendor_root: str | Path,
        dataset_root: str | Path | None = None,
        text_feature_root: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.vendor_root = Path(vendor_root)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.text_feature_root = Path(text_feature_root) if text_feature_root else None
        self.device = device

    def build_initialized_state(self) -> ReferenceTrainingState:
        """Build the actual registered graph, optimizer, and scheduler."""
        try:
            with _vendor_import_root(self.vendor_root), _legacy_pillow_compat():
                detectron2_config = importlib.import_module("detectron2.config")
                detectron2_modeling = importlib.import_module("detectron2.modeling")
                detectron2_data = importlib.import_module("detectron2.data")
                radm_config = importlib.import_module("RADM.config")
                train_net = importlib.import_module("train_net")
                config = detectron2_config.get_cfg()
                radm_config.add_radm_config(config)
                train_net.add_model_ema_configs(config)
                config.merge_from_file(
                    str(self.vendor_root / "configs" / "Base-RADM.yaml")
                )
                config.merge_from_file(str(self.vendor_root / "configs" / "radm.yaml"))
                config.defrost()
                config.MODEL.DEVICE = self.device
                if self.dataset_root is not None:
                    config.DATASETS.DATASET_PATH = str(self.dataset_root)
                if self.text_feature_root is not None:
                    config.DATASETS.TEXT_FEATURE_PATH = str(self.text_feature_root)
                config.freeze()
                for dataset_name in ("layout_train", "layout_val"):
                    if dataset_name in detectron2_data.DatasetCatalog:
                        detectron2_data.DatasetCatalog.remove(dataset_name)
                train_net.register_layout(config)
                model = detectron2_modeling.build_model(config)
                model.eval()
                optimizer = train_net.Trainer.build_optimizer(config, model)
                scheduler = train_net.Trainer.build_lr_scheduler(config, optimizer)
                mapper = train_net.RADMDatasetMapper(config, is_train=True)
                metadata = detectron2_data.MetadataCatalog.get("layout_train")
                world_size = int(
                    importlib.import_module("detectron2.utils.comm").get_world_size()
                )
                text_encoding = _runtime_text_encoding(mapper)
                effective = _effective_from_runtime(
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    mapper=mapper,
                    labels=tuple(metadata.thing_classes),
                    world_size=world_size,
                    text_encoding=text_encoding,
                )
                runtime_summary = _runtime_summary(
                    config=config,
                    model=model,
                    scheduler=scheduler,
                    mapper=mapper,
                    world_size=world_size,
                    labels=tuple(metadata.thing_classes),
                    text_encoding=text_encoding,
                )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ReferenceUnavailable(
                "The reference adapter requires the optional Detectron2/fvcore "
                "stack; no reference model was injected into the package model."
            ) from exc
        return ReferenceTrainingState(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            effective=effective,
            runtime_summary=runtime_summary,
        )


def _effective_from_runtime(
    *,
    config: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    mapper: Any,
    labels: tuple[str, ...],
    world_size: int,
    text_encoding: RuntimeTextEncoding,
) -> RADMEffectiveConfig:
    """Materialize the evidence schema from live runtime objects."""
    solver = config.SOLVER
    radm = config.MODEL.RADM
    vendor_model = cast(_VendorModel, model)
    pixel_mean = tuple(float(value) for value in config.MODEL.PIXEL_MEAN)
    pixel_std = tuple(float(value) for value in config.MODEL.PIXEL_STD)
    betas = tuple(float(value) for value in optimizer.defaults["betas"])
    milestones = tuple(int(value) for value in solver.STEPS)
    crop_size = tuple(int(value) for value in config.INPUT.CROP.SIZE)
    min_size_train = tuple(int(value) for value in config.INPUT.MIN_SIZE_TRAIN)
    model_ema = bool(config.MODEL_EMA.ENABLED)
    return RADMEffectiveConfig(
        num_classes=int(radm.NUM_CLASSES),
        vocabulary_size=len(labels),
        class_id_to_label=dict(enumerate(labels)),
        predicted_class_id_to_label=dict(enumerate(labels[: int(radm.NUM_CLASSES)])),
        num_proposals=int(radm.NUM_PROPOSALS),
        hidden_dim=int(radm.HIDDEN_DIM),
        text_feature_dim=text_encoding.feature_dim,
        max_text_num=text_encoding.max_text_num,
        num_heads=int(radm.NUM_HEADS),
        num_attention_heads=int(radm.NHEADS),
        dim_feedforward=int(radm.DIM_FEEDFORWARD),
        num_dynamic=int(radm.NUM_DYNAMIC),
        dim_dynamic=int(radm.DIM_DYNAMIC),
        num_cls=int(radm.NUM_CLS),
        num_reg=int(radm.NUM_REG),
        roi_resolution=int(config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION),
        roi_sampling_ratio=int(config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO),
        backbone_depth=int(config.MODEL.RESNETS.DEPTH),
        backbone_freeze_at=int(config.MODEL.BACKBONE.FREEZE_AT),
        pixel_mean=cast(tuple[float, float, float], pixel_mean),
        pixel_std=cast(tuple[float, float, float], pixel_std),
        with_vtram=bool(radm.withVTRAM),
        with_gram=bool(radm.withGRAM),
        deep_supervision=bool(radm.DEEP_SUPERVISION),
        use_focal=bool(radm.USE_FOCAL),
        use_fed_loss=bool(radm.USE_FED_LOSS),
        class_weight=float(radm.CLASS_WEIGHT),
        giou_weight=float(radm.GIOU_WEIGHT),
        l1_weight=float(radm.L1_WEIGHT),
        no_object_weight=float(radm.NO_OBJECT_WEIGHT),
        prior_prob=float(radm.PRIOR_PROB),
        alpha=float(radm.ALPHA),
        gamma=float(radm.GAMMA),
        ota_k=int(radm.OTA_K),
        num_train_timesteps=vendor_model.num_timesteps,
        snr_scale=float(radm.SNR_SCALE),
        sample_step=int(radm.SAMPLE_STEP),
        optimizer=str(solver.OPTIMIZER),
        learning_rate=float(solver.BASE_LR),
        weight_decay=float(solver.WEIGHT_DECAY),
        betas=cast(tuple[float, float], betas),
        eps=float(optimizer.defaults["eps"]),
        batch_size=int(solver.IMS_PER_BATCH),
        backbone_multiplier=float(solver.BACKBONE_MULTIPLIER),
        gradient_clip_norm=float(solver.CLIP_GRADIENTS.CLIP_VALUE),
        warmup_factor=float(solver.WARMUP_FACTOR),
        warmup_iters=int(solver.WARMUP_ITERS),
        milestones=cast(tuple[int, int], milestones),
        max_iter=int(solver.MAX_ITER),
        scheduler_gamma=float(getattr(scheduler, "gamma", 0.1)),
        scheduler_interval="step",
        num_gpus=world_size,
        world_size=world_size,
        gradient_accumulation_steps=1,
        eval_period=int(config.TEST.EVAL_PERIOD),
        num_workers=int(config.DATALOADER.NUM_WORKERS),
        filter_empty_annotations=bool(config.DATALOADER.FILTER_EMPTY_ANNOTATIONS),
        seed=int(config.SEED),
        min_size_train=min_size_train,
        max_size_train=int(config.INPUT.MAX_SIZE_TRAIN),
        min_size_train_sampling=str(config.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        crop_enabled=bool(config.INPUT.CROP.ENABLED),
        crop_size=cast(tuple[int, int], crop_size),
        crop_type=str(config.INPUT.CROP.TYPE),
        box_renewal=vendor_model.box_renewal,
        use_ensemble=vendor_model.use_ensemble,
        random_repeat_permutation=True,
        ema_enabled=model_ema,
        amp_enabled=bool(config.SOLVER.AMP.ENABLED),
        ddp_enabled=world_size > 1,
        simple_trainer=not bool(config.SOLVER.AMP.ENABLED),
        transform_names=tuple(type(value).__name__ for value in mapper.tfm_gens),
        crop_transform_names=tuple(
            type(value).__name__ for value in (mapper.crop_gen or [])
        ),
    )


def _runtime_summary(
    *,
    config: Any,
    model: torch.nn.Module,
    scheduler: Any,
    mapper: Any,
    world_size: int,
    labels: tuple[str, ...],
    text_encoding: RuntimeTextEncoding,
) -> dict[str, Any]:
    """Capture branch decisions from the initialized runtime."""
    vendor_model = cast(_VendorModel, model)
    amp_enabled = bool(config.SOLVER.AMP.ENABLED)
    return {
        "optimizer": str(config.SOLVER.OPTIMIZER),
        "scheduler_interval": "step",
        "scheduler_class": scheduler.__class__.__name__,
        "ema_enabled": bool(config.MODEL_EMA.ENABLED),
        "amp_enabled": amp_enabled,
        "ddp_enabled": world_size > 1,
        "simple_trainer": not amp_enabled,
        "box_renewal": vendor_model.box_renewal,
        "use_ensemble": vendor_model.use_ensemble,
        "transform_names": tuple(type(value).__name__ for value in mapper.tfm_gens),
        "crop_transform_names": tuple(
            type(value).__name__ for value in (mapper.crop_gen or [])
        ),
        "min_size_train": tuple(int(value) for value in config.INPUT.MIN_SIZE_TRAIN),
        "max_size_train": int(config.INPUT.MAX_SIZE_TRAIN),
        "min_size_train_sampling": str(config.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        "text_feature_dim": text_encoding.feature_dim,
        "max_text_num": text_encoding.max_text_num,
        "text_mask_semantics": text_encoding.mask_semantics,
        "missing_text_fallback": text_encoding.missing_fallback,
        "class_vocabulary_size": len(labels),
        "predicted_class_count": int(config.MODEL.RADM.NUM_CLASSES),
        "class_id_to_label": labels,
        "predicted_class_id_to_label": labels[: int(config.MODEL.RADM.NUM_CLASSES)],
    }


def _runtime_text_encoding(mapper: Any) -> RuntimeTextEncoding:
    """Extract text shape and fallback behavior from the selected mapper."""
    parameter = inspect.signature(mapper.load_text).parameters.get("max_text_num")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        raise RuntimeError("the selected mapper must expose a max_text_num default")
    max_text_num = int(parameter.default)

    text_root = Path(mapper.text_feature_dir)
    for index in range(100):
        text_name = f"__radm_s0_missing_{id(mapper)}_{index}.png"
        feature_path = text_root / f"{Path(text_name).stem}_feats.pth"
        if not feature_path.exists():
            break
    else:
        raise RuntimeError("could not select a missing S0 text-feature path")

    payload, valid_mask = mapper.load_text(text_name)
    features = payload.get("feats")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise RuntimeError("the mapper fallback must return a rank-2 feature tensor")
    if features.shape[0] != max_text_num:
        raise RuntimeError(
            "the mapper fallback changed its declared text padding length"
        )
    valid_mask = valid_mask.to(dtype=torch.bool).reshape(-1)
    if valid_mask.numel() != max_text_num:
        raise RuntimeError(
            "the mapper fallback mask length does not match text padding"
        )
    if bool(valid_mask.any()) or bool(torch.count_nonzero(features)):
        raise RuntimeError("the mapper missing-feature branch is not all-padding zeros")
    return RuntimeTextEncoding(
        feature_dim=int(features.shape[1]),
        max_text_num=max_text_num,
        mask_semantics="true_valid_false_padding",
        missing_fallback="zero_features_all_padding",
    )


def _jsonable(value: object) -> object:
    """Convert tensors and nested runtime values to metadata-safe values."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@contextlib.contextmanager
def _vendor_import_root(root: Path) -> Iterator[None]:
    """Temporarily expose the explicitly selected source checkout."""
    if not (root / "train_net.py").is_file():
        raise FileNotFoundError(f"missing reference launcher: {root / 'train_net.py'}")
    root_text = str(root.resolve())
    sys.path.insert(0, root_text)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


@contextlib.contextmanager
def _legacy_pillow_compat() -> Iterator[None]:
    """Bridge the checked Detectron2 v0.6 Pillow symbol during import only."""
    from PIL import Image

    had_linear = hasattr(Image, "LINEAR")
    if not had_linear:
        resampling = getattr(Image, "Resampling", None)
        if resampling is None:
            raise RuntimeError("Pillow has neither Image.LINEAR nor Image.Resampling")
        setattr(Image, "LINEAR", resampling.BILINEAR)
    try:
        yield
    finally:
        if not had_linear:
            delattr(Image, "LINEAR")


def write_reference_metadata(
    state: ReferenceTrainingState, output_path: str | Path
) -> None:
    """Write initialized-state metadata outside the repository source tree."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.metadata(), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
