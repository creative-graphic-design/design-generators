from tempfile import TemporaryDirectory

from radm.model_card import write_local_model_card


def test_write_local_model_card() -> None:
    with TemporaryDirectory() as tmp:
        path = write_local_model_card(tmp, dataset_name="cgl")
        text = path.read_text(encoding="utf-8")
    assert "RADM local converted artifact" in text
