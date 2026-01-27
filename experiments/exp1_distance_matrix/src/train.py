"""Training script for distance matrix completion experiment."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import create_dataloaders
from .model import DistanceMatrixLoss, DistanceMatrixTransformer


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for distances, mask, _ in dataloader:
        distances = distances.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred = model(distances, mask)
        loss = loss_fn(pred, distances, mask)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    n_masked = 0

    for distances, mask, _ in dataloader:
        distances = distances.to(device)
        mask = mask.to(device)

        pred = model(distances, mask)
        loss = loss_fn(pred, distances, mask)

        # Compute MAE on masked positions
        pred_mask = 1 - mask
        mae = (torch.abs(pred - distances) * pred_mask).sum()
        n_masked += pred_mask.sum().item()

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mae": total_mae / n_masked if n_masked > 0 else 0.0,
        "rmse": math.sqrt(total_loss / n_batches),
    }


def train(
    n_points: int = 20,
    mask_ratio: float = 0.3,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    dropout: float = 0.1,
    batch_size: int = 64,
    lr: float = 1e-4,
    n_epochs: int = 100,
    train_samples: int = 10000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    seed: int = 42,
    output_dir: str = "outputs",
    device: str | None = None,
) -> dict:
    """Train the distance matrix completion model."""
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(seed)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        n_points=n_points,
        mask_ratio=mask_ratio,
        batch_size=batch_size,
        seed=seed,
    )

    # Create model
    model = DistanceMatrixTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_points=n_points + 10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    loss_fn = DistanceMatrixLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_path / "best_model.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f}"
            )

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(
        f"\nTest Results: Loss: {test_metrics['loss']:.6f} | "
        f"MAE: {test_metrics['mae']:.4f} | RMSE: {test_metrics['rmse']:.4f}"
    )

    # Save results
    results = {
        "config": {
            "n_points": n_points,
            "mask_ratio": mask_ratio,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "seed": seed,
            "n_params": n_params,
        },
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train distance matrix completion model")
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/exp1")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
