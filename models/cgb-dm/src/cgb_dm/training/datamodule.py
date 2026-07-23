"""PyTorch Lightning data module for CGB-DM training."""

from __future__ import annotations

from torch.utils.data import DataLoader

from cgb_dm.configuration_cgb_dm import CGBDMConfig

from .dataset import CGBDMOriginalDataset, CGBDMSyntheticDataset

try:
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - exercised only without training extra

    class LightningDataModule:  # type: ignore[no-redef]
        """Fallback base when Lightning is not installed."""


class CGBDMDataModule(LightningDataModule):
    """Data module for original-zip or synthetic CGB-DM rows."""

    def __init__(
        self,
        *,
        config: CGBDMConfig | dict[str, object],
        source: str = "synthetic",
        data_root: str | None = None,
        batch_size: int = 2,
        num_workers: int = 0,
    ) -> None:
        """Initialize data module options."""
        super().__init__()
        self.config = (
            config if isinstance(config, CGBDMConfig) else CGBDMConfig(**config)
        )
        self.source = source
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Create train and validation datasets."""
        del stage
        if self.source == "original_zip":
            if self.data_root is None:
                raise ValueError("data_root is required for source='original_zip'")
            self.train_dataset = CGBDMOriginalDataset(self.data_root, split="train")
            self.val_dataset = CGBDMOriginalDataset(self.data_root, split="val")
            return
        if self.source != "synthetic":
            raise ValueError(f"Unsupported CGB-DM data source: {self.source}")
        kwargs = {
            "max_seq_length": self.config.max_seq_length,
            "seq_dim": self.config.seq_dim,
            "image_size": self.config.image_size,
        }
        self.train_dataset = CGBDMSyntheticDataset(**kwargs)
        self.val_dataset = CGBDMSyntheticDataset(length=2, **kwargs)

    def train_dataloader(self) -> DataLoader[dict[str, object]]:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, object]]:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
