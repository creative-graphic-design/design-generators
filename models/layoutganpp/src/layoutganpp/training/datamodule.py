"""Lightning data module for LayoutGAN++ training."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import LayoutGANPPDataset, LayoutRow, collate_layoutganpp
from laygen.common.randomness import resolve_torch_generator


class LayoutGANPPDataModule(LightningDataModule):
    """Load local reference-shaped rows or deterministic smoke-test rows."""

    def __init__(
        self,
        *,
        dataset_name: str,
        data_root: str | None = None,
        batch_size: int,
        num_workers: int,
        synthetic_size: int = 64,
        seed: int = 0,
        shuffle_train: bool = True,
    ) -> None:
        """Initialize the package-local training data module."""
        super().__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root) if data_root is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.synthetic_size = synthetic_size
        self.seed = seed
        self.shuffle_train = shuffle_train

    def setup(self, stage: str | None = None) -> None:
        """Create deterministic train, validation, and test datasets."""
        del stage
        self.train_dataset = LayoutGANPPDataset(
            dataset_name=self.dataset_name,
            split="train",
            data_root=self.data_root,
            synthetic_size=self.synthetic_size,
            seed=self.seed,
        )
        self.val_dataset = LayoutGANPPDataset(
            dataset_name=self.dataset_name,
            split="val",
            data_root=self.data_root,
            synthetic_size=max(2, self.synthetic_size // 4),
            seed=self.seed + 10_000,
        )
        self.test_dataset = LayoutGANPPDataset(
            dataset_name=self.dataset_name,
            split="test",
            data_root=self.data_root,
            synthetic_size=max(2, self.synthetic_size // 4),
            seed=self.seed + 20_000,
        )

    def train_dataloader(self) -> DataLoader[LayoutRow]:
        """Return the training data loader."""
        self.setup("fit")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=collate_layoutganpp,
            generator=resolve_torch_generator(seed=self.seed),
        )

    def val_dataloader(self) -> DataLoader[LayoutRow]:
        """Return the validation data loader."""
        self.setup("validate")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_layoutganpp,
        )

    def test_dataloader(self) -> DataLoader[LayoutRow]:
        """Return the test data loader."""
        self.setup("test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_layoutganpp,
        )
