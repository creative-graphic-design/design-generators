from torch.utils.data import Dataset
import pytest
import torch

import radm.training.datamodule as datamodule
from radm.training.dataset import RADMTrainingExample


def _example() -> RADMTrainingExample:
    return RADMTrainingExample(
        image=torch.zeros(3, 8, 8),
        boxes_xyxy=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([1]),
        text_features=torch.zeros(2, 4),
        text_mask=torch.tensor([[True], [False]]),
        canvas_size=torch.tensor([8, 8]),
    )


class TinyDataset(Dataset[RADMTrainingExample]):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> RADMTrainingExample:
        if index != 0:
            raise IndexError(index)
        return _example()


def test_datamodule_builds_train_and_validation_loaders(monkeypatch) -> None:
    monkeypatch.setattr(datamodule, "CGLV2ParquetDataset", TinyDataset)
    module = datamodule.RADMDataModule(
        data_root="cache",
        batch_size=1,
        image_size=8,
        max_elements=3,
        max_text_num=2,
        text_feature_dim=4,
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=2,
        max_train_samples=1,
        max_val_samples=1,
    )

    train_batch = next(iter(module.train_dataloader()))
    module.setup("validate")
    val_batch = next(iter(module.val_dataloader()))

    assert train_batch.boxes_xyxy.shape == (1, 3, 4)
    assert val_batch.mask.tolist() == [[True, False, False]]
    assert module.train_dataset is not None
    assert module.val_dataset is not None


def test_datamodule_loader_requires_initialized_dataset() -> None:
    module = datamodule.RADMDataModule(num_workers=0)
    with pytest.raises(RuntimeError, match="Dataset has not been initialized"):
        module._loader(None, shuffle=False)
