"""Lightning data module for LayoutGAN++ training."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .config import LayoutGANPPSeedMode
from .dataset import LayoutGANPPDataset, LayoutRow, collate_layoutganpp


class LayoutGANPPDataModule(LightningDataModule):
    """Load local reference-shaped rows or deterministic synthetic rows."""

    def __init__(
        self,
        *,
        dataset_name: str,
        data_root: str | None = None,
        batch_size: int,
        num_workers: int,
        synthetic_size: int = 64,
        seed: int = 0,
        seed_mode: LayoutGANPPSeedMode | str = LayoutGANPPSeedMode.default,
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
        self.seed_mode = LayoutGANPPSeedMode(seed_mode)
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
        if not hasattr(self, "train_dataset"):
            self.setup("fit")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=collate_layoutganpp,
        )

    def val_dataloader(self) -> DataLoader[LayoutRow]:
        """Return the validation data loader."""
        if not hasattr(self, "val_dataset"):
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
        if not hasattr(self, "test_dataset"):
            self.setup("test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_layoutganpp,
        )
