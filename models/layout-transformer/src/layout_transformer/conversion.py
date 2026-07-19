"""Conversion helpers for original LT-Net checkpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal, cast

import yaml

from .configuration_layout_transformer import BoxLossType, DecoderHeadType
from .configuration_layout_transformer import LayoutTransformerConfig
from .modeling_layout_transformer import LayoutTransformerForLayoutGeneration
from .processing_layout_transformer import LayoutTransformerProcessor
from .tokenization_layout_transformer import LayoutTransformerRelationTokenizer
from .vendor_state_dict import load_original_state_dict


SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}


def _load_vocab(vocab_path: str | Path) -> dict[int, str]:
    path = Path(vocab_path)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            raw_vocab = pickle.load(f)
    else:
        with path.open() as f:
            raw_vocab = json.load(f)
    return {int(key): str(value) for key, value in raw_vocab.items()}


def _split_vocab(
    id2token: dict[int, str],
) -> tuple[dict[int, str], dict[int, str], list[int], list[int]]:
    relation_start = next(
        (idx for idx, token in sorted(id2token.items()) if token == "__in_image__"),
        None,
    )
    if relation_start is None:
        object_items = [
            (idx, token)
            for idx, token in sorted(id2token.items())
            if token not in SPECIAL_TOKENS
        ]
        relation_items: list[tuple[int, str]] = []
    else:
        object_items = [
            (idx, token)
            for idx, token in sorted(id2token.items())
            if idx < relation_start and token not in SPECIAL_TOKENS
        ]
        relation_items = [
            (idx, token)
            for idx, token in sorted(id2token.items())
            if idx >= relation_start
        ]
    id2label = dict(object_items)
    relation_id2label = dict(relation_items)
    return id2label, relation_id2label, list(id2label), list(relation_id2label)


def _load_config(
    *,
    cfg_path: str | Path,
    dataset_name: str,
    vocab_size: int,
    id2label: dict[int, str],
    relation_id2label: dict[int, str],
) -> LayoutTransformerConfig:
    with Path(cfg_path).open() as f:
        raw_cfg = yaml.safe_load(f) or {}
    model_cfg = raw_cfg.get("MODEL", {})
    encoder_cfg = model_cfg.get("ENCODER", {})
    decoder_cfg = model_cfg.get("DECODER", {})
    refine_cfg = model_cfg.get("REFINE", {})
    return LayoutTransformerConfig(
        dataset_name=dataset_name,
        vocab_size=int(encoder_cfg.get("VOCAB_SIZE", vocab_size)),
        obj_classes_size=int(encoder_cfg.get("OBJ_CLASSES_SIZE", 155)),
        hidden_size=int(encoder_cfg.get("HIDDEN_SIZE", 256)),
        num_hidden_layers=int(encoder_cfg.get("NUM_LAYERS", 4)),
        num_attention_heads=int(encoder_cfg.get("ATTN_HEADS", 4)),
        dropout=float(encoder_cfg.get("DROPOUT", 0.1)),
        enable_noise=bool(encoder_cfg.get("ENABLE_NOISE", False)),
        noise_size=int(encoder_cfg.get("NOISE_SIZE", 64)),
        decoder_head_type=DecoderHeadType(
            str(decoder_cfg.get("HEAD_TYPE", "GMM")).lower()
        ),
        decoder_box_loss=BoxLossType(str(decoder_cfg.get("BOX_LOSS", "PDF")).lower()),
        decoder_schedule_sample=bool(decoder_cfg.get("SCHEDULE_SAMPLE", False)),
        decoder_two_path=bool(decoder_cfg.get("TWO_PATH", False)),
        decoder_global_feature=bool(decoder_cfg.get("GLOBAL_FEATURE", True)),
        decoder_greedy=bool(decoder_cfg.get("GREEDY", True)),
        xy_temperature=float(decoder_cfg.get("XY_TEMP", 1.0)),
        wh_temperature=float(decoder_cfg.get("WH_TEMP", 1.0)),
        refine=bool(refine_cfg.get("REFINE", True)),
        refine_head_type=DecoderHeadType(
            str(refine_cfg.get("HEAD_TYPE", "Linear")).lower()
        ),
        refine_box_loss=BoxLossType(str(refine_cfg.get("BOX_LOSS", "Reg")).lower()),
        refine_x_softmax=bool(refine_cfg.get("X_Softmax", True)),
        id2label=id2label,
        relation_id2label=relation_id2label,
    )


def convert_original_checkpoint(
    *,
    checkpoint_path: str | Path,
    cfg_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    dataset_name: Literal["coco", "vg_msdn"],
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    strict: bool = True,
) -> None:
    """Convert an original LT-Net checkpoint into local HF-style files.

    Args:
        checkpoint_path: Vendor ``.pth`` checkpoint containing ``state_dict`` or
            a raw state dict.
        cfg_path: Vendor YAML config path. Stored in metadata for traceability.
        vocab_path: JSON id-to-token vocabulary exported from
            ``object_pred_idx_to_name.pkl``.
        output_dir: Directory to write converted model/processor files.
        dataset_name: Dataset identifier for the converted checkpoint.
        push_to_hub: Reserved publish flag; implementation PRs keep it false.
        hub_model_id: Optional Hub repo id used only when publishing is enabled.
        strict: Whether checkpoint keys must exactly match the converted model.

    Raises:
        NotImplementedError: If Hub upload is requested from this helper.

    Examples:
        >>> convert_original_checkpoint  # doctest: +ELLIPSIS
        <function convert_original_checkpoint at ...>
    """
    if push_to_hub or hub_model_id is not None:
        raise NotImplementedError(
            "Hub upload is intentionally not part of PR conversion"
        )
    if not strict:
        raise ValueError("LayoutTransformer conversion requires strict=True")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = _load_vocab(vocab_path)
    id2label, relation_id2label, object_token_ids, relation_token_ids = _split_vocab(
        vocab
    )
    config = _load_config(
        cfg_path=cfg_path,
        dataset_name=dataset_name,
        vocab_size=len(vocab),
        id2label=id2label,
        relation_id2label=relation_id2label,
    )
    model = LayoutTransformerForLayoutGeneration(config)
    state_dict = load_original_state_dict(checkpoint_path)
    incompatible = model.load_state_dict(state_dict, strict=True)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "cfg_path": str(cfg_path),
        "vocab_path": str(vocab_path),
        "dataset_name": dataset_name,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_vendor_key_mapping": True,
    }
    model.save_pretrained(out)
    tokenizer = LayoutTransformerRelationTokenizer(
        tokens=[vocab[idx] for idx in sorted(vocab)],
        object_token_ids=object_token_ids,
        relation_token_ids=relation_token_ids,
    )
    processor = LayoutTransformerProcessor(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_sequence_length=config.max_sequence_length,
        id2label=cast(dict[int, str], config.id2label),
        relation_id2label=config.relation_id2label,
    )
    processor.save_pretrained(out)
    with (out / "conversion_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
