import json
from pathlib import Path
from typing import get_args, get_type_hints, cast

import torch
import pytest

from laygen.common.bbox import BoxFormat
from laygen.common.labels import DatasetName

from layoutformerpp import (
    ConditionType,
    LayoutFormerPPConfig,
    LayoutFormerPPProcessor,
    LayoutFormerPPTask,
    LayoutGenerationOutput,
    OutputType,
)


def test_processor_label_condition_and_postprocess() -> None:
    processor = LayoutFormerPPProcessor.from_config(
        dataset=DatasetName.rico25, task=LayoutFormerPPTask.gen_t
    )
    batch = processor(
        condition_type="label_size",
        labels=[["Text"]],
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
    )
    assert batch["input_ids"].shape[0] == 1
    ids = processor.tokenizer.encode_text("label_1 0 0 10 10 |")["input_ids"]
    out = processor.post_process_layouts(ids)
    assert isinstance(out, LayoutGenerationOutput)
    assert out.labels.tolist() == [[0]]
    assert out.mask.tolist() == [[True]]
    assert out.id2label[0] == "Text"


def test_processor_masks_padded_condition_rows_and_normalizes_pixels() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="completion")

    masked = processor(
        condition_type="completion",
        labels=[["Text", "Image"]],
        bbox=[[[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]],
        mask=torch.tensor([[True, False]]),
    )
    decoded = processor.tokenizer.batch_decode(
        masked["input_ids"], skip_special_tokens=True
    )

    assert decoded == ["label_1 0 0 25 25"]

    pixel = processor(
        condition_type="label_size",
        labels=[["Text"]],
        bbox=torch.tensor([[[50.0, 50.0, 20.0, 20.0]]]),
        normalized=False,
        canvas_size=(100, 100),
    )
    decoded_pixel = processor.tokenizer.batch_decode(
        pixel["input_ids"], skip_special_tokens=True
    )

    assert decoded_pixel == ["label_1 25 25"]

    with pytest.raises(ValueError, match="canvas_size is required"):
        processor(
            condition_type="label_size",
            labels=[["Text"]],
            bbox=torch.tensor([[[50.0, 50.0, 20.0, 20.0]]]),
            normalized=False,
        )


def test_processor_condition_aliases_and_error_paths() -> None:
    processor = LayoutFormerPPProcessor.from_config(
        dataset="rico", task=ConditionType.relation
    )

    assert processor.normalize_condition_type("gen_t") is ConditionType.label
    assert (
        processor.normalize_condition_type(ConditionType.relation)
        is ConditionType.relation
    )
    with pytest.raises(ValueError, match="Unsupported condition_type"):
        processor.normalize_condition_type("bad")
    with pytest.raises(ValueError, match="Unknown label"):
        processor(condition_type="label", labels=[["missing"]])
    call_hints = get_type_hints(processor.__call__)
    assert get_args(call_hints["return_tensors"]) == ("pt",)

    relation = processor(
        condition_type=ConditionType.relation,
        labels=[["Text", "Image"]],
        relations=[[(2, 1, 1, 0, 3)]],
    )
    assert relation["input_ids"].shape[0] == 1
    assert processor.dataset == "rico25"
    assert processor.task == "gen_r"


def test_processor_postprocess_padding_dict_and_errors() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
    sequences = processor.tokenizer.encode_text(
        ["label_1 0 0 10 10 | label_2 1 1 2 2 |", ""],
        add_eos=True,
    )["input_ids"]

    out = processor.post_process_layouts(
        sequences, box_format=BoxFormat.ltwh, output_type=OutputType.dict
    )
    assert isinstance(out, dict)
    labels = out["labels"]
    mask = out["mask"]
    intermediates = cast(dict[str, object], out["intermediates"])
    assert isinstance(labels, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(intermediates, dict)
    assert labels.shape == (2, 2)
    assert mask.tolist() == [[True, True], [False, False]]
    assert intermediates["box_format"] is BoxFormat.ltwh

    with pytest.raises(ValueError, match="Unsupported output_type"):
        processor.post_process_layouts(sequences, output_type="bad")
    postprocess_hints = get_type_hints(processor.post_process_layouts)
    assert get_args(postprocess_hints["return_tensors"]) == ("pt",)


@pytest.mark.parametrize(
    ("public_label", "sequence_id"),
    [
        (3, 5),
        (4, 4),
        (18, 6),
        (20, 15),
        (21, 25),
        ("  TEXT\t Button ", 5),
    ],
)
def test_s0_rico25_public_labels_join_sequence_ids_by_name(
    public_label: int | str, sequence_id: int
) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_t")
    assert processor._label_to_internal_id(public_label) == sequence_id
    with pytest.raises(ValueError, match="internal-only"):
        processor._label_to_internal_id(f"label_{sequence_id}")


def test_s0_rico25_sequence_labels_round_trip_to_public_ids() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_t")
    ids = processor.tokenizer.encode_text(
        [
            "label_5 0 0 10 10 |",
            "label_4 0 0 10 10 |",
            "label_6 0 0 10 10 |",
        ]
    )["input_ids"]
    output = processor.post_process_layouts(ids)
    assert isinstance(output, LayoutGenerationOutput)
    assert output.labels.tolist() == [[3], [4], [18]]
    assert processor.dataset == "rico25"
    assert processor.label_translation_metadata["public_to_sequence"][3] == 5
    assert len(processor.label_translation_sha256) == 64


def test_s0_rico_alias_is_canonicalized_at_public_config_boundary() -> None:
    config = LayoutFormerPPConfig(dataset="rico")
    assert config.dataset == "rico25"
    assert config.label_translation_metadata["sha256"] == (
        LayoutFormerPPProcessor.from_config(
            dataset="rico25",
            task="gen_t",
        ).label_translation_sha256
    )


def test_s0_rico25_dual_map_metadata_round_trips_and_fails_closed(
    tmp_path: Path,
) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_t")
    processor.save_pretrained(tmp_path)
    processor_path = tmp_path / "processor_config.json"
    saved = json.loads(processor_path.read_text(encoding="utf-8"))
    assert saved["label_translation_metadata"] == json.loads(
        json.dumps(processor.label_translation_metadata)
    )

    loaded = LayoutFormerPPProcessor.from_pretrained(tmp_path)
    assert loaded.label_translation_metadata == processor.label_translation_metadata
    assert loaded.label_translation_sha256 == processor.label_translation_sha256

    saved["label_translation_metadata"]["sha256"] = "0" * 64
    processor_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="label translation metadata"):
        LayoutFormerPPProcessor.from_pretrained(tmp_path)
