from typing import cast

from housegan import HouseGanConfig


def test_config_defaults_roundtrip(tmp_path):
    config = HouseGanConfig()
    assert cast(dict[int, str], config.id2label)[0] == "living_room"
    assert config.relation_id2label[-1] == "not_adjacent"
    assert config.num_labels == 10
    assert "GPL-3.0" in config.license_note
    config.save_pretrained(tmp_path)
    loaded = HouseGanConfig.from_pretrained(tmp_path)
    assert cast(dict[int, str], loaded.id2label)[9] == "laundry_room"
