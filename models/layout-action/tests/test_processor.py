import pytest
import torch
from typing import cast

import layout_action.tokenization_layout_action as tokenization_layout_action
from layout_action import (
    LayoutActionConfig,
    LayoutActionProcessor,
    LayoutActionTokenizer,
)


def processor() -> LayoutActionProcessor:
    return LayoutActionProcessor(
        LayoutActionTokenizer(
            LayoutActionConfig(dataset_name="publaynet", max_elements=2)
        )
    )


def test_processor_normalizes_vendor_aliases() -> None:
    proc = processor()

    unconditional = proc(condition_type="random_generate")
    label = proc(condition_type="category_generate", labels=torch.tensor([[1, 2]]))
    completion = proc(
        condition_type="completion_generate",
        bbox=torch.zeros(1, 1, 4),
        labels=torch.zeros(1, 1, dtype=torch.long),
        mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert unconditional["input_ids"].shape == (1, 1)
    assert label["forced_token_ids"][0, 0].item() == proc.config.label_token_id(1)
    assert completion["input_ids"].shape[1] == 14


def test_processor_rejects_unsupported_conditions() -> None:
    proc = processor()

    with pytest.raises(NotImplementedError):
        proc(condition_type="label_size")
    with pytest.raises(NotImplementedError):
        proc(condition_type="retrieval")


def test_processor_post_process_dict() -> None:
    proc = processor()
    sequences = proc.tokenizer.encode_layout(
        bbox=torch.tensor([[[0.2, 0.3, 0.1, 0.1]]]),
        labels=torch.tensor([[2]]),
        mask=torch.tensor([[True]]),
    )

    output = proc.post_process_layouts(
        sequences,
        output_type="dict",
        return_intermediates=True,
    )

    assert list(output) == [
        "bbox",
        "labels",
        "mask",
        "id2label",
        "sequences",
        "intermediates",
    ]
    assert cast(torch.Tensor, output["mask"]).tolist() == [[True, False]]


def test_processor_payload_validation_and_serialization(tmp_path) -> None:
    proc = processor()

    with pytest.raises(ValueError):
        proc(return_tensors="np")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        proc(condition_type="label")

    encoded = proc(
        condition_type="completion",
        bbox=[[[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]]],
        labels=[["text", "figure"]],
        mask=[[True, True]],
        box_format="ltrb",
        normalized=False,
        canvas_size=(100, 100),
        num_elements=[1],
    )
    assert encoded["input_ids"].shape[1] == 14

    proc.save_pretrained(tmp_path)
    restored = LayoutActionProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert restored.config.dataset_name == proc.config.dataset_name


def test_processor_completion_prompt_preserves_copy_margin_actions() -> None:
    proc = LayoutActionProcessor(
        LayoutActionTokenizer(
            LayoutActionConfig(dataset_name="publaynet", max_elements=2)
        )
    )
    cfg = proc.config
    qbox = torch.tensor([[[40, 80, 20, 20], [70, 80, 20, 20]]])
    bbox = proc.tokenizer.continuize_bbox(qbox)

    encoded = proc(
        condition_type="completion",
        bbox=bbox,
        labels=torch.tensor([[0, 4]]),
        mask=torch.tensor([[True, True]]),
    )

    assert encoded["input_ids"].shape[1] == 1 + 2 * cfg.element_token_width
    assert encoded["input_ids"][0, 15].item() == cfg.margin_token_id
    assert encoded["input_ids"][0, 16].item() == cfg.object_token_id(1)
    assert encoded["input_ids"][0, 17].item() == 10
    assert encoded["input_ids"][0, 18].item() == cfg.copy_token_id


def test_processor_from_pretrained_forwards_hub_style_subfolder(
    tmp_path, monkeypatch
) -> None:
    proc = processor()
    (metadata_path,) = proc.tokenizer.save_vocabulary(tmp_path)
    calls: dict[str, object] = {}

    def fake_cached_file(path_or_repo_id, filename, **kwargs):  # type: ignore[no-untyped-def]
        calls["path_or_repo_id"] = path_or_repo_id
        calls["filename"] = filename
        calls.update(kwargs)
        return metadata_path

    monkeypatch.setattr(tokenization_layout_action, "cached_file", fake_cached_file)

    restored = LayoutActionProcessor.from_pretrained(
        "creative-graphic-design/layout-action-publaynet",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        subfolder="processor",
    )

    assert restored.config.dataset_name == proc.config.dataset_name
    assert calls["path_or_repo_id"] == "creative-graphic-design/layout-action-publaynet"
    assert calls["filename"] == tokenization_layout_action.TOKENIZER_CONFIG_FILE
    assert calls["subfolder"] == "processor"
