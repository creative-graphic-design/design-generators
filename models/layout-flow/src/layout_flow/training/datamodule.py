"""LightningDataModule for LayoutFlow training."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from laygen.common.bbox import BoxFormat

from .config import LayoutFlowTrainingDatasetName, LayoutFlowTrainingSplit
from .dataset import LayoutFlowH5Dataset, collate_layout_flow_batch


class LayoutFlowDataModule(LightningDataModule):
    """Package-local LightningDataModule for LayoutFlow HDF5 data."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        dataset_name: LayoutFlowTrainingDatasetName = "publaynet",
        batch_size: int = 256,
        max_length: int = 20,
        num_workers: int = 4,
        box_format: BoxFormat | str = BoxFormat.xywh,
        lex_order: bool = False,
        permute_elements: bool = False,
        inoue_split: bool = False,
    ) -> None:
        """Initialize datamodule settings."""
        super().__init__()
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.box_format = box_format
        self.lex_order = lex_order
        self.permute_elements = permute_elements
        self.inoue_split = inoue_split
        self.train_dataset: LayoutFlowH5Dataset | None = None
        self.val_dataset: LayoutFlowH5Dataset | None = None
        self.test_dataset: LayoutFlowH5Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open datasets for the requested stage."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("validation")
        if stage in {None, "test"}:
            self.test_dataset = self._dataset("test")

    def train_dataloader(self) -> DataLoader[object]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[object]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[object]:
        """Return the test dataloader."""
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(self.test_dataset, shuffle=False)

    def _dataset(self, split: LayoutFlowTrainingSplit) -> LayoutFlowH5Dataset:
        return LayoutFlowH5Dataset(
            data_path=self.data_path,
            dataset_name=self.dataset_name,
            split=split,
            lex_order=self.lex_order,
            permute_elements=self.permute_elements,
            inoue_split=self.inoue_split,
        )

    def _loader(
        self, dataset: LayoutFlowH5Dataset | None, *, shuffle: bool
    ) -> DataLoader[object]:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_layout_flow_batch,
                max_length=self.max_length,
                box_format=self.box_format,
            ),
        )
