import torch

from basnet import BASNetConfig, BASNetModel, normalize_saliency


def test_normalize_saliency_scales_per_row():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])

    normalized = normalize_saliency(pred)

    assert torch.equal(normalized, torch.tensor([[[0.0, 0.25], [0.5, 1.0]]]))


def test_basnet_forward_and_tuple_output():
    model = BASNetModel(BASNetConfig()).eval()

    with torch.no_grad():
        output = model(torch.rand(1, 3, 256, 256), return_dict=False)

    assert output[0].shape == torch.Size([1, 256, 256])


def test_basnet_save_pretrained_round_trip(tmp_path):
    model = BASNetModel(BASNetConfig()).eval()
    model.save_pretrained(tmp_path)

    loaded = BASNetModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, BASNetModel)
