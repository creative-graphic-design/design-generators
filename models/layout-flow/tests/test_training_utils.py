import pytest
import torch

from layout_flow.training.dataset import LayoutFlowH5Dataset, collate_layout_flow_batch
from layout_flow.training.losses import layout_flow_losses


def test_collate_layout_flow_batch_matches_vendor_xywh_padding() -> None:
    sample = {
        "id": "a",
        "bbox": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "type": torch.tensor([2]),
        "length": torch.tensor(1),
    }
    batch = collate_layout_flow_batch([sample], max_length=2)
    bbox = batch["bbox"]
    mask = batch["mask"]
    assert isinstance(bbox, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert bbox.shape == (1, 2, 4)
    assert bbox[0, 0].tolist() == pytest.approx([0.25, 0.4, 0.3, 0.4])
    assert mask[0, :, 0].tolist() == [True, False]
    assert batch["id"] == ["a"]


def test_layout_flow_losses_use_flow_plus_weighted_geom_l1() -> None:
    cond = torch.ones(1, 1, 5)
    ut = torch.ones(1, 1, 5)
    vt = torch.zeros(1, 1, 5)
    losses = layout_flow_losses(cond, ut, vt, geom_dim=4, geom_l1_weight=0.2)
    assert torch.isclose(losses["flow_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["geom_l1_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["train_loss"], torch.tensor(1.2))


def test_h5_dataset_and_datamodule_read_tiny_vendor_file(tmp_path) -> None:
    pytest.importorskip("h5pickle")
    pytest.importorskip("lightning")

    import h5py

    from layout_flow.training.datamodule import LayoutFlowDataModule

    path = tmp_path / "publaynet_train.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("sample-0")
        group.create_dataset("bbox", data=[[0.0, 0.0, 0.2, 0.4]])
        group.create_dataset("categories", data=[1])
        group.create_dataset("length", data=1)
    for split_name in ["publaynet_val.h5", "publaynet_test.h5"]:
        (tmp_path / split_name).write_bytes(path.read_bytes())

    dataset = LayoutFlowH5Dataset(data_path=tmp_path, dataset_name="publaynet")
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["id"] == "sample-0"
    labels = sample["type"]
    assert isinstance(labels, torch.Tensor)
    assert torch.equal(labels, torch.tensor([1]))

    datamodule = LayoutFlowDataModule(
        data_path=tmp_path,
        dataset_name="publaynet",
        batch_size=1,
        max_length=2,
        num_workers=0,
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["bbox"].shape == (1, 2, 4)
