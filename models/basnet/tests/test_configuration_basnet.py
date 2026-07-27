from basnet import BASNetConfig


def test_config_round_trip(tmp_path):
    config = BASNetConfig(input_size=128, id2label={0: "foreground"})
    config.save_pretrained(tmp_path)

    loaded = BASNetConfig.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.input_size == 128
    assert loaded.id2label == {0: "foreground"}


def test_config_rejects_invalid_input_size():
    try:
        BASNetConfig(input_size=0)
    except ValueError as exc:
        assert "input_size" in str(exc)
    else:
        raise AssertionError("invalid input_size should fail")
