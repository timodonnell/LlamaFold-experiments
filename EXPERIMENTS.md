# Experiments

This document describes the derisking experiments for the LLM-based protein structure prediction project.

## Experiment 1: Distance Matrix Completion

**Goal:** Test geometric reasoning ability of transformer models on distance matrices.

**Task:** Given a partially observed distance matrix for random 3D point clouds, predict missing distances.

**Success Criteria:** Near-zero loss on held-out distances.

**Extension:** Reconstruct 3D coordinates up to rigid-body transforms.

### Running the Experiment

```bash
# Quick training run (for testing)
uv run python -m experiments.exp1_distance_matrix.src.train \
    --n-epochs 20 \
    --train-samples 2000 \
    --no-wandb \
    --output-dir outputs/exp1_quick

# Full training run with wandb logging
uv run python -m experiments.exp1_distance_matrix.src.train \
    --n-epochs 1000 \
    --train-samples 100000 \
    --batch-size 128 \
    --lr 3e-4 \
    --wandb-project distance-matrix-completion \
    --output-dir outputs/exp1_full
```

### Architecture Overview

The experiment uses a BERT-style masked prediction approach:

1. **Data Generation**: Random 3D point clouds → pairwise distance matrices → random masking
2. **Model**: Transformer encoder predicts masked distances from observed ones
3. **Evaluation**: MSE/MAE on masked positions + coordinate reconstruction via MDS

### Key Components

#### Data Generation ([`data.py`](experiments/exp1_distance_matrix/src/data.py))

The dataset generates random 3D point clouds on-the-fly and computes their distance matrices:

```python
# experiments/exp1_distance_matrix/src/data.py:54-81

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
```

#### Model Architecture ([`model.py`](experiments/exp1_distance_matrix/src/model.py))

The transformer operates on the upper triangle of the distance matrix (exploiting symmetry). Each point-pair becomes a token with 2D positional encoding:

```python
# experiments/exp1_distance_matrix/src/model.py:45-95

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
```

The forward pass extracts the upper triangle, applies masking, and reconstructs a symmetric output:

```python
# experiments/exp1_distance_matrix/src/model.py:113-143

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
```

#### Coordinate Reconstruction ([`reconstruct.py`](experiments/exp1_distance_matrix/src/reconstruct.py))

The extension task reconstructs 3D coordinates from predicted distance matrices using classical Multi-Dimensional Scaling (MDS):

```python
# experiments/exp1_distance_matrix/src/reconstruct.py:11-49

def classical_mds(distances: torch.Tensor, n_dims: int = 3) -> torch.Tensor:
    """Reconstruct coordinates from distance matrix using classical MDS."""
    batch_size, n_points, _ = distances.shape
    device = distances.device

    # Squared distances
    D_sq = distances**2

    # Centering matrix
    eye = torch.eye(n_points, device=device)
    ones = torch.ones(n_points, n_points, device=device) / n_points
    H = eye - ones

    # Gram matrix (inner products): B = -0.5 * H @ D_sq @ H
    H_batch = H.unsqueeze(0).expand(batch_size, -1, -1)
    B = -0.5 * torch.bmm(torch.bmm(H_batch, D_sq), H_batch)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(B)

    # Take top n_dims eigenvalues (they're sorted ascending, so take last ones)
    eigenvalues = eigenvalues[:, -n_dims:]
    eigenvectors = eigenvectors[:, :, -n_dims:]

    # Clamp negative eigenvalues (can happen due to numerical errors)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    # Reconstruct coordinates
    coords = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(1)

    return coords
```

Procrustes analysis aligns reconstructed coordinates to ground truth (handling rotation/translation ambiguity):

```python
# experiments/exp1_distance_matrix/src/reconstruct.py:52-100

def procrustes_align(
    coords_pred: torch.Tensor,
    coords_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Align predicted coordinates to target using Procrustes analysis."""
    # Center both sets
    pred_centered = coords_pred - coords_pred.mean(dim=1, keepdim=True)
    target_centered = coords_target - coords_target.mean(dim=1, keepdim=True)

    # Compute optimal rotation using SVD
    H = torch.bmm(pred_centered.transpose(1, 2), target_centered)
    U, S, Vh = torch.linalg.svd(H)

    # Rotation matrix R = V @ U^T
    # Handle reflection: if det(R) < 0, flip sign of last column
    V = Vh.transpose(1, 2)
    d = torch.linalg.det(torch.bmm(V, U.transpose(1, 2)))

    sign_matrix = torch.ones(V.shape[0], V.shape[2], device=V.device)
    sign_matrix[:, -1] = torch.sign(d)

    V_corrected = V * sign_matrix.unsqueeze(1)
    R = torch.bmm(V_corrected, U.transpose(1, 2))

    # Apply rotation and translate to match target center
    aligned = torch.bmm(pred_centered, R.transpose(1, 2))
    aligned = aligned + coords_target.mean(dim=1, keepdim=True)

    # Compute RMSD
    diff = aligned - coords_target
    rmsd = torch.sqrt((diff**2).sum(dim=-1).mean(dim=-1))

    return aligned, rmsd.mean().item()
```

### Results

**Full training run** (1000 epochs, 100k samples, 4.9M parameters):

| Metric | Value |
|--------|-------|
| **Test Loss (MSE)** | 0.0653 |
| **Test MAE** | 0.171 |
| **Test RMSE** | 0.256 |
| Best Epoch | 779 |
| Best Val Loss | 0.0621 |

Given coordinates scaled to ~10 units (distances ranging 0-30+), a **MAE of 0.17** is effectively **near-zero**, meeting the success criteria.

**wandb run:** https://wandb.ai/timodonnell/distance-matrix-completion/runs/mbk250c5

Training progression:

| Epoch | Train Loss | Val MAE |
|-------|------------|---------|
| 1 | 98.68 | 19.15 |
| 100 | 1.75 | 0.92 |
| 500 | 0.21 | 0.21 |
| 1000 | 0.13 | 0.18 |

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-points` | 20 | Points per point cloud |
| `--mask-ratio` | 0.3 | Fraction of distances to mask |
| `--d-model` | 256 | Transformer hidden dimension |
| `--n-heads` | 8 | Attention heads |
| `--n-layers` | 6 | Transformer layers |
| `--d-ff` | 1024 | Feed-forward dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--n-epochs` | 100 | Training epochs |
| `--train-samples` | 10000 | Training set size |

---

## Experiment 2: Infilling Individual Residues

**Status:** Not yet implemented

**Goal:** Test binned-coordinate tokenization.

**Task:** Given a protein structure with one missing residue, predict its CA coordinate.

**Success Criteria:** Error within a few Angstroms most of the time.

---

## Experiment 3: Predict Structures from Constraints

**Status:** Not yet implemented

**Goal:** Assess end-to-end structure generation.

**Task:** Given a sequence and sparse distance constraints, predict full CA coordinates.

**Success Criteria:** Median RMSD < 3Å on well-folded proteins.
