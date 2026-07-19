from pathlib import Path
from typing import cast

from ralf import RalfConfig


def test_config_derives_token_ids_and_round_trips(tmp_path: Path) -> None:
    config = RalfConfig(
        dataset_name="cgl",
        retrieval_metadata={"table": "cache/retrieval.json"},
        original_hydra_config={"generator": {"_target_": "vendor"}},
    )

    assert config.num_labels == 5
    assert config.vocab_size == 5 + 128 * 4 + 3
    assert config.pad_token_id == 5 + 128 * 4
    assert config.bbox_token_offset("center_x") == 5 + 128 * 2

    config.save_pretrained(tmp_path)
    loaded = RalfConfig.from_pretrained(tmp_path)

    assert loaded.retrieval_metadata == {"table": "cache/retrieval.json"}
    assert loaded.original_hydra_config == {"generator": {"_target_": "vendor"}}


def test_pku_config_keeps_invalid_label_for_source_vocabulary() -> None:
    config = RalfConfig(dataset_name="pku_posterlayout")

    assert cast(dict[int, str], config.id2label)[3] == "INVALID"
    assert cast(dict[str, int], config.label2id)["text"] == 0
