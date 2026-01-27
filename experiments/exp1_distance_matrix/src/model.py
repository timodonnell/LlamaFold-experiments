"""Transformer model for distance matrix completion.

Uses a transformer architecture to predict missing distances from observed ones.
The model treats each pair of points as a token, similar to how proteins might
be represented in future experiments.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for matrix positions (i, j)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model

        # Create position encodings for row and column
        pe = torch.zeros(max_len, d_model // 2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, i_indices: torch.Tensor, j_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            i_indices: (batch, n_pairs) row indices
            j_indices: (batch, n_pairs) column indices

        Returns:
            (batch, n_pairs, d_model) positional encodings
        """
        pe_i = self.pe[i_indices]  # (batch, n_pairs, d_model//2)
        pe_j = self.pe[j_indices]  # (batch, n_pairs, d_model//2)
        return torch.cat([pe_i, pe_j], dim=-1)


class DistanceMatrixTransformer(nn.Module):
    """Transformer for distance matrix completion.

    Architecture:
    - Input: observed distances with positional encodings
    - Encoder: standard transformer encoder
    - Output: predicted distances for masked positions
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_points: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        # Input embedding: distance value -> d_model
        self.distance_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Mask token (learned embedding for masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, max_points)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        distances: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            distances: (batch, n_points, n_points) distance matrices
            mask: (batch, n_points, n_points) 1 = observed, 0 = to predict

        Returns:
            (batch, n_points, n_points) predicted distance matrices
        """
        batch_size, n_points, _ = distances.shape
        device = distances.device

        # Get upper triangle indices (avoid redundant symmetric entries)
        triu_i, triu_j = torch.triu_indices(n_points, n_points, offset=1, device=device)

        # Extract upper triangle values
        dist_flat = distances[:, triu_i, triu_j]  # (batch, n_pairs)
        mask_flat = mask[:, triu_i, triu_j]  # (batch, n_pairs)

        # Embed distances
        dist_embed = self.distance_embed(dist_flat.unsqueeze(-1))  # (batch, n_pairs, d_model)

        # Replace masked positions with mask token
        mask_expanded = mask_flat.unsqueeze(-1)  # (batch, n_pairs, 1)
        x = dist_embed * mask_expanded + self.mask_token * (1 - mask_expanded)

        # Add positional encoding
        i_indices = triu_i.unsqueeze(0).expand(batch_size, -1)
        j_indices = triu_j.unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoding(i_indices, j_indices)

        # Transform
        x = self.transformer(x)

        # Project to distance predictions
        pred_flat = self.output_proj(x).squeeze(-1)  # (batch, n_pairs)

        # Reconstruct full symmetric matrix
        pred = torch.zeros(batch_size, n_points, n_points, device=device)
        pred[:, triu_i, triu_j] = pred_flat
        pred[:, triu_j, triu_i] = pred_flat  # Symmetric

        return pred


class DistanceMatrixLoss(nn.Module):
    """Loss function for distance matrix completion.

    Computes MSE only on masked (predicted) positions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, n_points, n_points) predicted distances
            target: (batch, n_points, n_points) ground truth distances
            mask: (batch, n_points, n_points) 1 = observed, 0 = to predict

        Returns:
            Scalar loss value
        """
        # Compute loss only on masked positions
        pred_mask = 1 - mask  # Positions we predicted
        diff = (pred - target) ** 2
        masked_diff = diff * pred_mask

        # Average over masked positions
        n_masked = pred_mask.sum()
        if n_masked > 0:
            loss = masked_diff.sum() / n_masked
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss
