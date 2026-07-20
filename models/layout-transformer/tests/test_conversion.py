import json
import pickle

import torch
from transformers import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig

from layout_transformer import (
    LayoutTransformerConfig,
    LayoutTransformerForLayoutGeneration,
    LayoutTransformerProcessor,
    LayoutTransformerRelationTokenizer,
)
from layout_transformer.conversion import (
    _load_config,
    _load_vocab,
    _split_vocab,
    convert_original_checkpoint,
)
from layout_transformer.vendor_state_dict import (
    load_original_state_dict,
    load_strict_mapped_state_dict,
)


def test_load_original_state_dict_strips_module_prefix(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, checkpoint)

    state_dict = load_original_state_dict(checkpoint)

    assert sorted(state_dict) == ["weight"]


def test_load_strict_mapped_state_dict_detects_mismatch():
    model = torch.nn.Linear(1, 1)

    try:
        load_strict_mapped_state_dict(model, {"other.weight": torch.ones(1, 1)})
    except RuntimeError as exc:
        assert "State dict mismatch" in str(exc)
    else:
        raise AssertionError("expected strict mismatch")


def test_convert_original_checkpoint_rejects_non_strict_conversion(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    vocab = tmp_path / "object_pred_id2name.json"
    cfg = tmp_path / "config.yaml"
    output = tmp_path / "converted"
    torch.save({"state_dict": {}}, checkpoint)
    vocab.write_text(
        json.dumps(
            {"0": "[PAD]", "1": "[CLS]", "2": "[SEP]", "3": "[MASK]", "4": "person"}
        )
    )
    cfg.write_text("MODEL: {}\n")

    try:
        convert_original_checkpoint(
            checkpoint_path=checkpoint,
            cfg_path=cfg,
            vocab_path=vocab,
            output_dir=output,
            dataset_name="coco",
            strict=False,
        )
    except ValueError as exc:
        assert "strict=True" in str(exc)
    else:
        raise AssertionError("expected non-strict conversion rejection")


def test_convert_original_checkpoint_rejects_hub_upload(tmp_path):
    try:
        convert_original_checkpoint(
            checkpoint_path=tmp_path / "missing.pth",
            cfg_path=tmp_path / "missing.yaml",
            vocab_path=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
            dataset_name="coco",
            push_to_hub=True,
        )
    except NotImplementedError as exc:
        assert "Hub upload" in str(exc)
    else:
        raise AssertionError("expected Hub upload rejection")


def test_load_vocab_accepts_json_and_pickle(tmp_path):
    raw_vocab = {"0": "[PAD]", "1": "[CLS]", "4": "person"}
    json_path = tmp_path / "vocab.json"
    pkl_path = tmp_path / "vocab.pkl"
    json_path.write_text(json.dumps(raw_vocab))
    with pkl_path.open("wb") as f:
        pickle.dump(raw_vocab, f)

    assert _load_vocab(json_path) == {0: "[PAD]", 1: "[CLS]", 4: "person"}
    assert _load_vocab(pkl_path) == {0: "[PAD]", 1: "[CLS]", 4: "person"}


def test_split_vocab_separates_objects_and_relations():
    id2label, relation_id2label, object_ids, relation_ids = _split_vocab(
        {
            0: "[PAD]",
            1: "[CLS]",
            2: "[SEP]",
            3: "[MASK]",
            4: "__image__",
            5: "person",
            6: "__in_image__",
            7: "left of",
        }
    )

    assert id2label == {4: "__image__", 5: "person"}
    assert relation_id2label == {6: "__in_image__", 7: "left of"}
    assert object_ids == [4, 5]
    assert relation_ids == [6, 7]


def test_load_config_maps_vendor_yaml_fields(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
MODEL:
  ENCODER:
    VOCAB_SIZE: 8
    OBJ_CLASSES_SIZE: 9
    HIDDEN_SIZE: 32
    NUM_LAYERS: 1
    ATTN_HEADS: 4
    DROPOUT: 0.0
    ENABLE_NOISE: true
    NOISE_SIZE: 11
  DECODER:
    HEAD_TYPE: GMM
    BOX_LOSS: PDF
    SCHEDULE_SAMPLE: true
    TWO_PATH: true
    GLOBAL_FEATURE: false
    GREEDY: false
    XY_TEMP: 0.7
    WH_TEMP: 0.8
  REFINE:
    REFINE: false
    HEAD_TYPE: Linear
    BOX_LOSS: Reg
    X_Softmax: false
"""
    )

    config = _load_config(
        cfg_path=cfg,
        dataset_name="coco",
        vocab_size=99,
        id2label={4: "__image__", 5: "person"},
        relation_id2label={6: "__in_image__"},
    )

    assert config.vocab_size == 8
    assert config.obj_classes_size == 9
    assert config.hidden_size == 32
    assert config.enable_noise is True
    assert config.decoder_greedy is False
    assert config.decoder_global_feature is False
    assert config.refine is False
    assert config.refine_x_softmax is False


def test_convert_original_checkpoint_writes_loadable_local_package(tmp_path):
    vocab = {
        "0": "[PAD]",
        "1": "[CLS]",
        "2": "[SEP]",
        "3": "[MASK]",
        "4": "__image__",
        "5": "person",
        "6": "__in_image__",
        "7": "left of",
    }
    vocab_path = tmp_path / "object_pred_id2name.json"
    cfg_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_dir = tmp_path / "converted"
    vocab_path.write_text(json.dumps(vocab))
    cfg_path.write_text(
        """
MODEL:
  ENCODER:
    VOCAB_SIZE: 8
    OBJ_CLASSES_SIZE: 9
    HIDDEN_SIZE: 32
    NUM_LAYERS: 1
    ATTN_HEADS: 4
    DROPOUT: 0.0
  DECODER:
    GREEDY: true
  REFINE:
    REFINE: false
"""
    )
    model = LayoutTransformerForLayoutGeneration(
        LayoutTransformerConfig(
            dataset_name="coco",
            vocab_size=8,
            obj_classes_size=9,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            refine=False,
            id2label={4: "__image__", 5: "person"},
            relation_id2label={6: "__in_image__", 7: "left of"},
        )
    )
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    convert_original_checkpoint(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        vocab_path=vocab_path,
        output_dir=output_dir,
        dataset_name="coco",
    )

    metadata = json.loads((output_dir / "conversion_metadata.json").read_text())
    config = LayoutTransformerConfig.from_pretrained(output_dir)
    tokenizer = LayoutTransformerRelationTokenizer.from_pretrained(output_dir)
    reloaded_model = LayoutTransformerForLayoutGeneration.from_pretrained(output_dir)
    AutoConfig.register(
        LayoutTransformerConfig.model_type, LayoutTransformerConfig, exist_ok=True
    )
    AutoProcessor.register(
        LayoutTransformerConfig, LayoutTransformerProcessor, exist_ok=True
    )
    processor = AutoProcessor.from_pretrained(output_dir, local_files_only=True)

    assert metadata["strict_vendor_key_mapping"] is True
    assert metadata["missing_keys"] == []
    assert metadata["unexpected_keys"] == []
    assert config.dataset_name == "coco"
    assert config.hidden_size == 32
    assert "use_vendor_modules" not in config.to_dict()
    assert tokenizer.object_token_ids == [4, 5]
    assert tokenizer.relation_token_ids == [6, 7]
    assert isinstance(reloaded_model, LayoutTransformerForLayoutGeneration)
    assert isinstance(processor, LayoutTransformerProcessor)
    assert processor.dataset_name == "coco"
