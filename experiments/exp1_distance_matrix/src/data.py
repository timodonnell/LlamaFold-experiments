"""Data generation for distance matrix experiment.

Generates documents showing pairwise distances between random 3D points
in a text format suitable for LLM training.
"""

import random
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class DistanceDocument:
    """A document containing pairwise distances."""

    coordinates: np.ndarray  # (20, 3) coordinates
    pairs: list[tuple[int, int]]  # List of (point_i, point_j) pairs
    distances: list[int]  # Corresponding integer distances

    def to_text(self, include_end: bool = True) -> str:
        """Convert to text format for LLM training.

        Args:
            include_end: Whether to include <end> token. Set False for eval prompts.
        """
        lines = ["<start>"]
        for (i, j), dist in zip(self.pairs, self.distances):
            # Space-separate tokens so tokenizer can split them
            lines.append(f"<p{i}> <p{j}> <d{dist}>")
        if include_end:
            lines.append("<end>")
        return "\n".join(lines)


def generate_coordinates(n_points: int = 20, coord_range: float = 100.0) -> np.ndarray:
    """Generate random 3D coordinates from uniform distribution.

    Args:
        n_points: Number of points to generate.
        coord_range: Coordinates are sampled uniformly from [-coord_range, coord_range].

    Returns:
        Array of shape (n_points, 3) with coordinates.
    """
    return np.random.uniform(-coord_range, coord_range, size=(n_points, 3))


def compute_all_distances(coords: np.ndarray) -> dict[tuple[int, int], int]:
    """Compute all pairwise distances between points.

    Args:
        coords: Array of shape (n_points, 3).

    Returns:
        Dictionary mapping (i, j) pairs (i < j) to integer distances.
    """
    n_points = len(coords)
    distances = {}
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances[(i, j)] = int(round(dist))
    return distances


def create_document(
    coords: np.ndarray,
    pair_order: list[tuple[int, int]] | None = None,
    swap_prob: float = 0.5,
) -> DistanceDocument:
    """Create a distance document from coordinates.

    Args:
        coords: Array of shape (n_points, 3).
        pair_order: Optional specific ordering of pairs. If None, random order is used.
        swap_prob: Probability of swapping (i,j) to (j,i) for each pair.

    Returns:
        DistanceDocument with randomized pair ordering.
    """
    distances = compute_all_distances(coords)
    all_pairs = list(distances.keys())

    if pair_order is None:
        pair_order = all_pairs.copy()
        random.shuffle(pair_order)

    # Randomly swap order within pairs
    final_pairs = []
    final_distances = []
    for i, j in pair_order:
        if random.random() < swap_prob:
            final_pairs.append((j, i))
        else:
            final_pairs.append((i, j))
        # Distance is symmetric, so use the canonical (min, max) key
        key = (min(i, j), max(i, j))
        final_distances.append(distances[key])

    return DistanceDocument(
        coordinates=coords,
        pairs=final_pairs,
        distances=final_distances,
    )


def split_document_for_eval(
    doc: DistanceDocument,
    n_observed: int = 180,
) -> tuple[DistanceDocument, DistanceDocument]:
    """Split a document into observed and held-out portions.

    Args:
        doc: Full distance document.
        n_observed: Number of pairs to include in observed portion.

    Returns:
        Tuple of (observed_doc, held_out_doc).
    """
    n_total = len(doc.pairs)
    indices = list(range(n_total))
    random.shuffle(indices)

    observed_indices = indices[:n_observed]
    held_out_indices = indices[n_observed:]

    observed_pairs = [doc.pairs[i] for i in observed_indices]
    observed_distances = [doc.distances[i] for i in observed_indices]

    held_out_pairs = [doc.pairs[i] for i in held_out_indices]
    held_out_distances = [doc.distances[i] for i in held_out_indices]

    observed_doc = DistanceDocument(
        coordinates=doc.coordinates,
        pairs=observed_pairs,
        distances=observed_distances,
    )

    held_out_doc = DistanceDocument(
        coordinates=doc.coordinates,
        pairs=held_out_pairs,
        distances=held_out_distances,
    )

    return observed_doc, held_out_doc


class DistanceDataset(Dataset):
    """Dataset that generates distance documents on-the-fly."""

    def __init__(
        self,
        size: int,
        n_points: int = 20,
        coord_range: float = 100.0,
        seed: int | None = None,
    ):
        """Initialize dataset.

        Args:
            size: Number of documents in dataset.
            n_points: Number of points per document.
            coord_range: Coordinates sampled uniformly from [-coord_range, coord_range].
            seed: Random seed for reproducibility.
        """
        self.size = size
        self.n_points = n_points
        self.coord_range = coord_range
        self.seed = seed

        # Pre-generate all documents for consistency
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.documents = []
        for _ in range(size):
            coords = generate_coordinates(n_points, coord_range)
            doc = create_document(coords)
            self.documents.append(doc)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        doc = self.documents[idx]
        return {
            "text": doc.to_text(),
            "coordinates": doc.coordinates,
            "pairs": doc.pairs,
            "distances": doc.distances,
        }


class EvalDataset(Dataset):
    """Dataset for evaluation with observed/held-out splits."""

    def __init__(
        self,
        size: int,
        n_points: int = 20,
        coord_range: float = 100.0,
        n_observed: int = 180,
        seed: int | None = None,
    ):
        """Initialize evaluation dataset.

        Args:
            size: Number of evaluation examples.
            n_points: Number of points per document.
            coord_range: Coordinates sampled uniformly from [-coord_range, coord_range].
            n_observed: Number of observed pairs (rest are held out).
            seed: Random seed for reproducibility.
        """
        self.size = size
        self.n_points = n_points
        self.coord_range = coord_range
        self.n_observed = n_observed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.examples = []
        for _ in range(size):
            coords = generate_coordinates(n_points, coord_range)
            doc = create_document(coords)
            observed, held_out = split_document_for_eval(doc, n_observed)
            self.examples.append(
                {
                    "observed": observed,
                    "held_out": held_out,
                    "full": doc,
                }
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        return {
            "prompt": example["observed"].to_text(include_end=False),
            "observed_pairs": example["observed"].pairs,
            "observed_distances": example["observed"].distances,
            "held_out_pairs": example["held_out"].pairs,
            "held_out_distances": example["held_out"].distances,
            "coordinates": example["full"].coordinates,
        }


def get_special_tokens() -> list[str]:
    """Get list of special tokens needed for this task."""
    tokens = ["<start>", "<end>"]
    # Point tokens for 20 points (no space, so tokenizer can split)
    for i in range(20):
        tokens.append(f"<p{i}>")
    # Distance tokens - max distance with coord_range=100 in 3D is roughly 346
    # Use range 0-500 to be safe
    for d in range(501):
        tokens.append(f"<d{d}>")
    return tokens
