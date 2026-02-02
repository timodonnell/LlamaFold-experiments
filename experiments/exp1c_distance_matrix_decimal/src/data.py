"""Data generation for distance matrix experiment with decimal tokenization.

Generates documents showing pairwise distances between random 3D points
in a text format suitable for LLM training. Distances are represented
using three tokens in decimal format (hundreds, tens, ones).
"""

import random
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


def format_distance(dist: int) -> str:
    """Format a distance as three decimal tokens.

    Args:
        dist: Integer distance (0-500).

    Returns:
        String with three tokens, e.g., "315" -> "<d300> <d10> <d5>"
    """
    if dist < 0 or dist > 500:
        raise ValueError(f"Distance {dist} out of range [0, 500]")

    hundreds = (dist // 100) * 100  # 0, 100, 200, 300, 400, 500
    tens = ((dist % 100) // 10) * 10  # 0, 10, 20, ..., 90
    ones = dist % 10  # 0, 1, 2, ..., 9

    return f"<d{hundreds:03d}> <d{tens:02d}> <d{ones}>"


def parse_distance_tokens(tokens: list[str]) -> int:
    """Parse three decimal tokens back to an integer distance.

    Args:
        tokens: List of three tokens, e.g., ["<d300>", "<d10>", "<d5>"]

    Returns:
        Integer distance, e.g., 315
    """
    if len(tokens) != 3:
        raise ValueError(f"Expected 3 tokens, got {len(tokens)}")

    hundreds = int(tokens[0][2:-1])  # "<d300>" -> 300
    tens = int(tokens[1][2:-1])  # "<d10>" -> 10
    ones = int(tokens[2][2:-1])  # "<d5>" -> 5

    return hundreds + tens + ones


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
            # Format: <pX> <pY> <dHHH> <dTT> <dO>
            dist_tokens = format_distance(dist)
            lines.append(f"<p{i}> <p{j}> {dist_tokens}")
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
    """Dataset that generates distance documents on-the-fly (no repeats)."""

    def __init__(
        self,
        size: int,
        n_points: int = 20,
        coord_range: float = 100.0,
        seed: int | None = None,
    ):
        """Initialize dataset.

        Args:
            size: Number of documents per "epoch" (for __len__).
            n_points: Number of points per document.
            coord_range: Coordinates sampled uniformly from [-coord_range, coord_range].
            seed: Not used (kept for API compatibility).
        """
        self.size = size
        self.n_points = n_points
        self.coord_range = coord_range

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        # Generate fresh document each time (no caching)
        coords = generate_coordinates(self.n_points, self.coord_range)
        doc = create_document(coords)
        return {
            "text": doc.to_text(),
            "coordinates": doc.coordinates,
            "pairs": doc.pairs,
            "distances": doc.distances,
        }


class EvalDataset(Dataset):
    """Dataset for evaluation with observed/held-out splits (fresh each time)."""

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
            size: Number of evaluation examples per "epoch" (for __len__).
            n_points: Number of points per document.
            coord_range: Coordinates sampled uniformly from [-coord_range, coord_range].
            n_observed: Number of observed pairs (rest are held out).
            seed: Not used (kept for API compatibility).
        """
        self.size = size
        self.n_points = n_points
        self.coord_range = coord_range
        self.n_observed = n_observed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        # Generate fresh example each time (no caching)
        coords = generate_coordinates(self.n_points, self.coord_range)
        doc = create_document(coords)
        observed, held_out = split_document_for_eval(doc, self.n_observed)
        return {
            "prompt": observed.to_text(include_end=False),
            "observed_pairs": observed.pairs,
            "observed_distances": observed.distances,
            "held_out_pairs": held_out.pairs,
            "held_out_distances": held_out.distances,
            "coordinates": doc.coordinates,
        }


def get_special_tokens() -> list[str]:
    """Get list of special tokens needed for this task.

    Uses decimal tokenization for distances:
    - Hundreds: <d000>, <d100>, <d200>, <d300>, <d400>, <d500> (6 tokens)
    - Tens: <d00>, <d10>, <d20>, ..., <d90> (10 tokens)
    - Ones: <d0>, <d1>, ..., <d9> (10 tokens)

    Total: 26 distance tokens instead of 501 in exp1.
    """
    tokens = ["<start>", "<end>"]

    # Point tokens for 20 points
    for i in range(20):
        tokens.append(f"<p{i}>")

    # Distance tokens - decimal format
    # Hundreds place: 0, 100, 200, 300, 400, 500
    for h in range(0, 600, 100):
        tokens.append(f"<d{h:03d}>")

    # Tens place: 0, 10, 20, ..., 90
    for t in range(0, 100, 10):
        tokens.append(f"<d{t:02d}>")

    # Ones place: 0, 1, 2, ..., 9
    for o in range(10):
        tokens.append(f"<d{o}>")

    return tokens
