"""Lightning data module for the approved PKU PosterLayout source."""

from __future__ import annotations

from collections.abc import Iterator, Sized

from datasets import DatasetDict
from lightning.pytorch import LightningDataModule
from jaxtyping import Shaped
from laygen.common.randomness import randperm, random_seed
import torch
from torch.utils.data import DataLoader, Sampler

from ..processing_ds_gan import DSGANProcessor
from .dataset import DSGANDataset, load_cached_dataset


class _VendorRandomSampler(Sampler[int]):
    """Reproduce the reference loader's default RandomSampler RNG sequence."""

    def __init__(self, data_source: Sized, generator: torch.Generator) -> None:
        self.data_source = data_source
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        seed = random_seed(self.generator)
        permutation_generator = torch.Generator(device="cpu")
        permutation_generator.manual_seed(seed)
        yield from randperm(
            len(self.data_source), generator=permutation_generator, device="cpu"
        ).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


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
        dataset = DSGANDataset(
            self.dataset["train"],
            split="train",
            processor=self.processor,
            max_elem=self.max_elem,
        )
        generator = self._generator()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=_VendorRandomSampler(dataset, generator),
            num_workers=self.num_workers,
            generator=generator,
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
        """Return the seeded CPU sampler stream used by the reference loader."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        return generator
