"""Tests for coordinate reconstruction."""

import torch

from experiments.exp1_distance_matrix.src.reconstruct import (
    classical_mds,
    procrustes_align,
)


class TestClassicalMDS:
    def test_output_shape(self):
        batch_size = 4
        n_points = 20
        n_dims = 3

        # Generate random points and compute distances
        points = torch.randn(batch_size, n_points, n_dims) * 10
        distances = torch.cdist(points, points)

        reconstructed = classical_mds(distances, n_dims=n_dims)
        assert reconstructed.shape == (batch_size, n_points, n_dims)

    def test_reconstruction_preserves_distances(self):
        batch_size = 2
        n_points = 15
        n_dims = 3

        # Generate random points
        points = torch.randn(batch_size, n_points, n_dims) * 10
        distances = torch.cdist(points, points)

        # Reconstruct
        reconstructed = classical_mds(distances, n_dims=n_dims)

        # Compute distances from reconstructed points
        reconstructed_distances = torch.cdist(reconstructed, reconstructed)

        # Distances should be preserved (up to numerical precision)
        assert torch.allclose(distances, reconstructed_distances, atol=1e-3)


class TestProcrustesAlign:
    def test_output_shape(self):
        batch_size = 4
        n_points = 20

        pred = torch.randn(batch_size, n_points, 3)
        target = torch.randn(batch_size, n_points, 3)

        aligned, rmsd = procrustes_align(pred, target)
        assert aligned.shape == (batch_size, n_points, 3)
        assert isinstance(rmsd, float)

    def test_perfect_alignment(self):
        batch_size = 2
        n_points = 15

        target = torch.randn(batch_size, n_points, 3)

        # Create "predicted" that's just rotated and translated
        # Generate proper rotation matrices (det = 1)
        R = torch.randn(batch_size, 3, 3)
        U, _, Vh = torch.linalg.svd(R)
        R = torch.bmm(U, Vh)  # Orthogonal matrix

        # Ensure det = 1 (rotation, not reflection)
        det = torch.linalg.det(R)
        # Flip last column if det < 0
        correction = torch.ones(batch_size, 3, device=R.device)
        correction[:, -1] = torch.sign(det)
        R = R * correction.unsqueeze(1)

        # Apply rotation and translation
        pred = torch.bmm(target, R) + torch.randn(batch_size, 1, 3)

        aligned, rmsd = procrustes_align(pred, target)

        # After alignment, RMSD should be ~0
        assert rmsd < 1e-4

    def test_rmsd_non_negative(self):
        pred = torch.randn(4, 20, 3)
        target = torch.randn(4, 20, 3)

        _, rmsd = procrustes_align(pred, target)
        assert rmsd >= 0

    def test_identity_alignment(self):
        target = torch.randn(2, 10, 3)

        # Aligning to itself should give RMSD = 0
        aligned, rmsd = procrustes_align(target.clone(), target)
        assert rmsd < 1e-6
        assert torch.allclose(aligned, target, atol=1e-5)
