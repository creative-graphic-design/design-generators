from pathlib import Path

from layout_corrector.conversion import discover_seed_dirs, remap_corrector_key


def test_remap_corrector_key():
    assert remap_corrector_key("model.module.model.cat_emb.weight") == "cat_emb.weight"


def test_discover_seed_dirs(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (tmp_path / "notes").mkdir()

    assert discover_seed_dirs(tmp_path) == [Path(tmp_path / "42")]
