from ds_gan.model_card import write_dsgan_model_card


def test_write_model_card(tmp_path):
    path = write_dsgan_model_card(tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "DS-GAN PosterLayout" in text
