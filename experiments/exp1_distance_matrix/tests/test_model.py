"""Tests for the distance matrix completion model."""

import pytest
import torch

from experiments.exp1_distance_matrix.src.model import (
    DistanceMatrixLoss,
    DistanceMatrixTransformer,
    PositionalEncoding2D,
)


class TestPositionalEncoding2D:
    def test_output_shape(self):
        d_model = 64
        pe = PositionalEncoding2D(d_model=d_model, max_len=100)

        batch_size = 4
        n_pairs = 50
        i_indices = torch.randint(0, 20, (batch_size, n_pairs))
        j_indices = torch.randint(0, 20, (batch_size, n_pairs))

        output = pe(i_indices, j_indices)
        assert output.shape == (batch_size, n_pairs, d_model)

    def test_different_positions_different_encoding(self):
        pe = PositionalEncoding2D(d_model=64, max_len=100)

        i1 = torch.tensor([[0, 1]])
        j1 = torch.tensor([[1, 2]])

        output = pe(i1, j1)
        # Different positions should have different encodings
        assert not torch.allclose(output[0, 0], output[0, 1])


class TestDistanceMatrixTransformer:
    @pytest.fixture
    def model(self):
        return DistanceMatrixTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            max_points=50,
        )

    def test_output_shape(self, model):
        batch_size = 4
        n_points = 20
        distances = torch.randn(batch_size, n_points, n_points).abs()
        distances = (distances + distances.transpose(1, 2)) / 2  # Symmetric
        mask = torch.ones(batch_size, n_points, n_points)

        output = model(distances, mask)
        assert output.shape == (batch_size, n_points, n_points)

    def test_output_symmetric(self, model):
        batch_size = 2
        n_points = 15
        distances = torch.randn(batch_size, n_points, n_points).abs()
        distances = (distances + distances.transpose(1, 2)) / 2
        mask = torch.ones(batch_size, n_points, n_points)

        output = model(distances, mask)
        assert torch.allclose(output, output.transpose(1, 2), atol=1e-5)

    def test_gradient_flow(self, model):
        batch_size = 2
        n_points = 10
        distances = torch.randn(batch_size, n_points, n_points).abs()
        distances = (distances + distances.transpose(1, 2)) / 2
        mask = torch.ones(batch_size, n_points, n_points)
        mask[:, 0, 1] = 0
        mask[:, 1, 0] = 0

        output = model(distances, mask)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDistanceMatrixLoss:
    def test_loss_only_on_masked(self):
        loss_fn = DistanceMatrixLoss()

        batch_size = 2
        n_points = 10
        pred = torch.randn(batch_size, n_points, n_points)
        target = torch.randn(batch_size, n_points, n_points)

        # Mask everything except one pair
        mask = torch.ones(batch_size, n_points, n_points)
        mask[:, 0, 1] = 0
        mask[:, 1, 0] = 0

        loss = loss_fn(pred, target, mask)

        # Manual calculation: both (0,1) and (1,0) are masked
        sq_errors = (pred[:, 0, 1] - target[:, 0, 1]) ** 2 + (pred[:, 1, 0] - target[:, 1, 0]) ** 2
        expected = sq_errors.sum() / 4  # 2 positions * 2 batches
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_zero_loss_when_perfect(self):
        loss_fn = DistanceMatrixLoss()

        target = torch.randn(2, 10, 10)
        mask = torch.ones(2, 10, 10)
        mask[:, 0, 1] = 0
        mask[:, 1, 0] = 0

        # Predict exactly the target
        loss = loss_fn(target, target, mask)
        assert loss.item() == 0.0

    def test_loss_ignores_observed(self):
        loss_fn = DistanceMatrixLoss()

        pred = torch.zeros(2, 10, 10)
        target = torch.ones(2, 10, 10)

        # All observed (mask = 1), so loss should be 0
        mask = torch.ones(2, 10, 10)
        loss = loss_fn(pred, target, mask)
        assert loss.item() == 0.0
