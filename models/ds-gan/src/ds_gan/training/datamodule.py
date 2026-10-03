"""Lightning data module for the approved PKU PosterLayout source."""

from __future__ import annotations

from datasets import DatasetDict
from lightning.pytorch import LightningDataModule
from jaxtyping import Shaped
import torch
from torch.utils.data import DataLoader

from ..processing_ds_gan import DSGANProcessor
from .dataset import DSGANDataset, load_cached_dataset


class DSGANDataModule(LightningDataModule):
    """Prepare training and test streams with the reference transforms."""

    def __init__(
        self,
        *,
        dataset_name: str = "pku_posterlayout",
        cache_dir: str | None = None,
        batch_size: int = 128,
        test_batch_size: int = 4,
        max_elem: int = 32,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        """Initialize the approved-source loader settings."""
        super().__init__()
        if dataset_name not in {"pku_posterlayout", "PKU-PosterLayout"}:
            raise ValueError(f"unsupported DS-GAN dataset: {dataset_name}")

        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_elem = max_elem
        self.num_workers = num_workers
        self.seed = seed
        self.processor = DSGANProcessor()
        self.dataset: DatasetDict | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open cached source rows for fitting or evaluation."""
        if stage in {None, "fit", "test"}:
            self.dataset = load_cached_dataset(self.cache_dir)

    def train_dataloader(self) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."]]]:
        """Return the reference-compatible shuffled training stream."""
        if self.dataset is None:
            self.setup("fit")

        assert self.dataset is not None
        return DataLoader(
            DSGANDataset(
                self.dataset["train"],
                split="train",
                processor=self.processor,
                max_elem=self.max_elem,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=self._generator(),
        )

    def test_dataloader(self) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."]]]:
        """Return the complete unshuffled test stream."""
        if self.dataset is None:
            self.setup("test")

        assert self.dataset is not None
        return DataLoader(
            DSGANDataset(
                self.dataset["test"],
                split="test",
                processor=self.processor,
                max_elem=self.max_elem,
            ),
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator
