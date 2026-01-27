"""Coordinate reconstruction from distance matrices using MDS.

Extension task: reconstruct 3D coordinates from predicted distance matrices
up to rigid-body transforms (translation, rotation, reflection).
"""

import torch
import torch.nn as nn


def classical_mds(distances: torch.Tensor, n_dims: int = 3) -> torch.Tensor:
    """Reconstruct coordinates from distance matrix using classical MDS.

    Args:
        distances: (batch, n_points, n_points) distance matrices
        n_dims: Number of dimensions for reconstruction (default 3)

    Returns:
        (batch, n_points, n_dims) reconstructed coordinates
    """
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
    eigenvalues = eigenvalues[:, -n_dims:]  # (batch, n_dims)
    eigenvectors = eigenvectors[:, :, -n_dims:]  # (batch, n_points, n_dims)

    # Clamp negative eigenvalues (can happen due to numerical errors)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    # Reconstruct coordinates
    coords = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(1)

    return coords


def procrustes_align(
    coords_pred: torch.Tensor,
    coords_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Align predicted coordinates to target using Procrustes analysis.

    Finds optimal rotation/reflection and translation to align coords_pred to coords_target.

    Args:
        coords_pred: (batch, n_points, 3) predicted coordinates
        coords_target: (batch, n_points, 3) target coordinates

    Returns:
        aligned_coords: (batch, n_points, 3) aligned predicted coordinates
        rmsd: Root mean square deviation after alignment
    """
    # Center both sets
    pred_centered = coords_pred - coords_pred.mean(dim=1, keepdim=True)
    target_centered = coords_target - coords_target.mean(dim=1, keepdim=True)

    # Compute optimal rotation using SVD
    # H = pred^T @ target
    H = torch.bmm(pred_centered.transpose(1, 2), target_centered)
    U, S, Vh = torch.linalg.svd(H)

    # Rotation matrix R = V @ U^T
    # Handle reflection: if det(R) < 0, flip sign of last column of U
    V = Vh.transpose(1, 2)
    d = torch.linalg.det(torch.bmm(V, U.transpose(1, 2)))

    # Create sign correction matrix
    sign_matrix = torch.ones(V.shape[0], V.shape[2], device=V.device)
    sign_matrix[:, -1] = torch.sign(d)

    # Apply correction to V
    V_corrected = V * sign_matrix.unsqueeze(1)
    R = torch.bmm(V_corrected, U.transpose(1, 2))

    # Apply rotation
    aligned = torch.bmm(pred_centered, R.transpose(1, 2))

    # Translate to match target center
    aligned = aligned + coords_target.mean(dim=1, keepdim=True)

    # Compute RMSD
    diff = aligned - coords_target
    rmsd = torch.sqrt((diff**2).sum(dim=-1).mean(dim=-1))

    return aligned, rmsd.mean().item()


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate coordinate reconstruction from predicted distance matrices.

    Args:
        model: Trained distance matrix completion model
        dataloader: DataLoader providing (distances, mask, points) tuples
        device: Device to run evaluation on

    Returns:
        Dictionary with reconstruction metrics
    """
    model.eval()
    total_rmsd_from_pred = 0.0
    total_rmsd_from_gt = 0.0
    total_dist_error = 0.0
    n_batches = 0

    for distances, mask, points in dataloader:
        distances = distances.to(device)
        mask = mask.to(device)
        points = points.to(device)

        # Predict full distance matrix
        pred_distances = model(distances, mask)

        # Combine: use ground truth for observed, predictions for masked
        combined_distances = distances * mask + pred_distances * (1 - mask)

        # Reconstruct coordinates from predicted distances
        coords_from_pred = classical_mds(combined_distances)

        # Reconstruct coordinates from ground truth distances (sanity check)
        coords_from_gt = classical_mds(distances)

        # Align and compute RMSD
        _, rmsd_pred = procrustes_align(coords_from_pred, points)
        _, rmsd_gt = procrustes_align(coords_from_gt, points)

        # Distance matrix error (on masked positions)
        pred_mask = 1 - mask
        dist_error = (torch.abs(pred_distances - distances) * pred_mask).sum()
        n_masked = pred_mask.sum()
        if n_masked > 0:
            dist_error = dist_error / n_masked

        total_rmsd_from_pred += rmsd_pred
        total_rmsd_from_gt += rmsd_gt
        total_dist_error += dist_error.item()
        n_batches += 1

    return {
        "rmsd_from_predictions": total_rmsd_from_pred / n_batches,
        "rmsd_from_ground_truth": total_rmsd_from_gt / n_batches,
        "mean_distance_error": total_dist_error / n_batches,
    }
