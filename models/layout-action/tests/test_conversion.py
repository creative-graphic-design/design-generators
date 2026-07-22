import torch

from layout_action import (
    LayoutActionConfig,
    LayoutActionForCausalLM,
    remap_layout_action_key,
    remap_state_dict,
    convert_layout_action_checkpoint,
)


def test_conversion_key_report_covers_synthetic_vendor_keys() -> None:
    config = LayoutActionConfig(
        dataset_name="publaynet",
        max_elements=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = LayoutActionForCausalLM(config)
    state_dict = {
        "tok_emb.weight": torch.zeros_like(model.tok_emb.weight),
        "missing.weight": torch.zeros(1),
    }

    remapped, report = remap_state_dict(state_dict, model)

    assert remap_layout_action_key("tok_emb.weight") == "tok_emb.weight"
    assert "tok_emb.weight" in remapped
    assert [row.loaded for row in report] == [True, False]


def test_convert_layout_action_checkpoint_with_synthetic_state(tmp_path) -> None:
    config = LayoutActionConfig(
        dataset_name="publaynet",
        max_elements=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = LayoutActionForCausalLM(config)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)

    report = convert_layout_action_checkpoint(
        checkpoint=checkpoint,
        output_dir=tmp_path / "converted",
        config=config,
        strict=True,
    )

    assert report["missing_keys"] == []
    assert (tmp_path / "converted" / "config.json").exists()
