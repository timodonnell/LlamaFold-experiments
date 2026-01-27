"""Data generation for distance matrix completion experiment.

Generates random 3D point clouds and their distance matrices with masking
for training the model to predict missing distances.
"""

from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset


class DistanceMatrixBatch(NamedTuple):
    """Batch of distance matrix completion examples."""

    distances: torch.Tensor  # (batch, n_points, n_points) full distance matrices
    mask: torch.Tensor  # (batch, n_points, n_points) 1 = observed, 0 = to predict
    points: torch.Tensor  # (batch, n_points, 3) original coordinates (for evaluation)


class DistanceMatrixDataset(Dataset):
    """Dataset that generates random 3D point clouds and distance matrices on-the-fly."""

    def __init__(
        self,
        n_samples: int,
        n_points: int = 20,
        mask_ratio: float = 0.3,
        coord_scale: float = 10.0,
        seed: int | None = None,
    ):
        """
        Args:
            n_samples: Number of samples in the dataset
            n_points: Number of points per point cloud
            mask_ratio: Fraction of distances to mask (predict)
            coord_scale: Scale of random coordinates
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.mask_ratio = mask_ratio
        self.coord_scale = coord_scale
        self.seed = seed

        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        else:
            self.rng = torch.Generator()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate random 3D points
        if self.seed is not None:
            # Deterministic based on idx and seed
            rng = torch.Generator().manual_seed(self.seed + idx)
        else:
            rng = self.rng

        points = torch.randn(self.n_points, 3, generator=rng) * self.coord_scale

        # Compute pairwise distances
        distances = torch.cdist(points, points)

        # Create mask (1 = observed, 0 = to predict)
        # Only mask upper triangle (distance matrix is symmetric)
        n_pairs = self.n_points * (self.n_points - 1) // 2
        n_mask = int(n_pairs * self.mask_ratio)

        # Get upper triangle indices
        triu_i, triu_j = torch.triu_indices(self.n_points, self.n_points, offset=1)
        perm = torch.randperm(n_pairs, generator=rng)
        mask_indices = perm[:n_mask]

        mask = torch.ones(self.n_points, self.n_points)
        mask[triu_i[mask_indices], triu_j[mask_indices]] = 0
        mask[triu_j[mask_indices], triu_i[mask_indices]] = 0  # Symmetric

        return distances, mask, points


def create_dataloaders(
    train_samples: int = 10000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    n_points: int = 20,
    mask_ratio: float = 0.3,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_dataset = DistanceMatrixDataset(
        n_samples=train_samples,
        n_points=n_points,
        mask_ratio=mask_ratio,
        seed=seed,
    )
    val_dataset = DistanceMatrixDataset(
        n_samples=val_samples,
        n_points=n_points,
        mask_ratio=mask_ratio,
        seed=seed + 100000,
    )
    test_dataset = DistanceMatrixDataset(
        n_samples=test_samples,
        n_points=n_points,
        mask_ratio=mask_ratio,
        seed=seed + 200000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
