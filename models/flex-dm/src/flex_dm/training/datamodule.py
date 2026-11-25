"""Lightning data module for original Flex-DM TFRecords."""

from __future__ import annotations

from pathlib import Path

from jaxtyping import Shaped
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from ..configuration_flex_dm import FlexDmDatasetName
from .dataset import FlexDmTfrecordDataset


class FlexDmDataModule(LightningDataModule):
    """Load package-readable Flex-DM TFRecords for Lightning."""

    def __init__(
        self,
        *,
        data_dir: str,
        dataset_name: FlexDmDatasetName | str,
        batch_size: int = 256,
        num_workers: int = 0,
        shuffle_train: bool = True,
    ) -> None:
        """Store the data location and loader settings."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.datasets: dict[str, FlexDmTfrecordDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        """Open datasets for the requested trainer stage."""
        if stage in {None, "fit"}:
            self.datasets["train"] = FlexDmTfrecordDataset(
                self.data_dir, self.dataset_name, "train"
            )
            self.datasets["val"] = FlexDmTfrecordDataset(
                self.data_dir, self.dataset_name, "val"
            )

        if stage in {None, "test"}:
            self.datasets["test"] = FlexDmTfrecordDataset(
                self.data_dir, self.dataset_name, "test"
            )

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[Tensor, "..."]]]:
        """Return the training loader."""
        if "train" not in self.datasets:
            self.setup("fit")

        return self._loader("train", shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader[dict[str, Shaped[Tensor, "..."]]]:
        """Return the ordered validation loader."""
        if "val" not in self.datasets:
            self.setup("fit")

        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Shaped[Tensor, "..."]]]:
        """Return the ordered test loader."""
        if "test" not in self.datasets:
            self.setup("test")

        return self._loader("test", shuffle=False)

    def _loader(
        self, split: str, *, shuffle: bool
    ) -> DataLoader[dict[str, Shaped[Tensor, "..."]]]:
        dataset = self.datasets[split]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
        )
