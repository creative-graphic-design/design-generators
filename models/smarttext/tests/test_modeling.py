import torch

from smarttext import SmartTextBASNet, SmartTextConfig, SmartTextScorer


def test_scorer_forward_and_save_load(tmp_path):
    config = SmartTextConfig()
    model = SmartTextScorer(config).eval()

    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, 256, 256),
            torch.tensor([[0, 0, 0, 64, 64]], dtype=torch.float32),
        )

    assert output.scores.shape == torch.Size([1])
    model.save_pretrained(tmp_path)
    loaded = SmartTextScorer.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, SmartTextScorer)


def test_basnet_forward_and_tuple_output():
    model = SmartTextBASNet(SmartTextConfig()).eval()

    with torch.no_grad():
        output = model(torch.rand(1, 3, 256, 256), return_dict=False)

    assert output[0].shape == torch.Size([1, 256, 256])
