"""Tests for data generation."""

import torch

from experiments.exp1_distance_matrix.src.data import (
    DistanceMatrixDataset,
    create_dataloaders,
)


class TestDistanceMatrixDataset:
    def test_dataset_length(self):
        dataset = DistanceMatrixDataset(n_samples=100, n_points=10, seed=42)
        assert len(dataset) == 100

    def test_dataset_shapes(self):
        n_points = 15
        dataset = DistanceMatrixDataset(n_samples=10, n_points=n_points, seed=42)
        distances, mask, points = dataset[0]

        assert distances.shape == (n_points, n_points)
        assert mask.shape == (n_points, n_points)
        assert points.shape == (n_points, 3)

    def test_distance_matrix_symmetric(self):
        dataset = DistanceMatrixDataset(n_samples=10, n_points=20, seed=42)
        distances, _, _ = dataset[0]

        assert torch.allclose(distances, distances.T)

    def test_distance_matrix_diagonal_zero(self):
        dataset = DistanceMatrixDataset(n_samples=10, n_points=20, seed=42)
        distances, _, _ = dataset[0]

        assert torch.allclose(distances.diag(), torch.zeros(20))

    def test_mask_symmetric(self):
        dataset = DistanceMatrixDataset(n_samples=10, n_points=20, seed=42)
        _, mask, _ = dataset[0]

        assert torch.allclose(mask, mask.T)

    def test_mask_ratio(self):
        n_points = 20
        mask_ratio = 0.3
        dataset = DistanceMatrixDataset(
            n_samples=100, n_points=n_points, mask_ratio=mask_ratio, seed=42
        )

        total_masked = 0
        total_pairs = n_points * (n_points - 1)  # Exclude diagonal

        for i in range(len(dataset)):
            _, mask, _ = dataset[i]
            # Count unmasked off-diagonal entries
            unmasked = (mask.sum() - n_points).item()  # Subtract diagonal
            masked = total_pairs - unmasked
            total_masked += masked

        avg_mask_ratio = total_masked / (len(dataset) * total_pairs)
        # Should be close to mask_ratio (within some tolerance)
        assert abs(avg_mask_ratio - mask_ratio) < 0.05

    def test_reproducibility_with_seed(self):
        dataset1 = DistanceMatrixDataset(n_samples=10, n_points=20, seed=42)
        dataset2 = DistanceMatrixDataset(n_samples=10, n_points=20, seed=42)

        d1, m1, p1 = dataset1[5]
        d2, m2, p2 = dataset2[5]

        assert torch.allclose(d1, d2)
        assert torch.allclose(m1, m2)
        assert torch.allclose(p1, p2)


class TestDataLoaders:
    def test_create_dataloaders(self):
        train_loader, val_loader, test_loader = create_dataloaders(
            train_samples=100,
            val_samples=20,
            test_samples=20,
            n_points=10,
            batch_size=16,
            seed=42,
        )

        assert len(train_loader.dataset) == 100
        assert len(val_loader.dataset) == 20
        assert len(test_loader.dataset) == 20

    def test_batch_shapes(self):
        train_loader, _, _ = create_dataloaders(
            train_samples=100,
            val_samples=20,
            test_samples=20,
            n_points=15,
            batch_size=8,
            seed=42,
        )

        distances, mask, points = next(iter(train_loader))

        assert distances.shape == (8, 15, 15)
        assert mask.shape == (8, 15, 15)
        assert points.shape == (8, 15, 3)
