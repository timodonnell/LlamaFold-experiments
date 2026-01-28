"""Tests for data generation."""

import numpy as np

from experiments.exp1_distance_matrix.src.data import (
    DistanceDataset,
    EvalDataset,
    compute_all_distances,
    create_document,
    generate_coordinates,
    get_special_tokens,
    split_document_for_eval,
)


class TestGenerateCoordinates:
    def test_shape(self):
        coords = generate_coordinates(n_points=20, std=100.0)
        assert coords.shape == (20, 3)

    def test_different_n_points(self):
        coords = generate_coordinates(n_points=10, std=50.0)
        assert coords.shape == (10, 3)


class TestComputeAllDistances:
    def test_number_of_pairs(self):
        coords = generate_coordinates(n_points=20)
        distances = compute_all_distances(coords)
        # n*(n-1)/2 pairs for n=20
        assert len(distances) == 20 * 19 // 2

    def test_distances_are_positive(self):
        coords = generate_coordinates(n_points=10)
        distances = compute_all_distances(coords)
        for dist in distances.values():
            assert dist >= 0

    def test_distances_are_integers(self):
        coords = generate_coordinates(n_points=10)
        distances = compute_all_distances(coords)
        for dist in distances.values():
            assert isinstance(dist, int)


class TestCreateDocument:
    def test_document_has_all_pairs(self):
        coords = generate_coordinates(n_points=20)
        doc = create_document(coords)
        assert len(doc.pairs) == 20 * 19 // 2
        assert len(doc.distances) == 20 * 19 // 2

    def test_document_to_text_format(self):
        coords = generate_coordinates(n_points=5)
        doc = create_document(coords)
        text = doc.to_text()

        assert text.startswith("<start>")
        assert text.endswith("<end>")
        assert "<point" in text

    def test_pairs_cover_all_combinations(self):
        coords = generate_coordinates(n_points=5)
        doc = create_document(coords)

        # Get canonical pairs (smaller index first)
        canonical_pairs = set()
        for i, j in doc.pairs:
            canonical_pairs.add((min(i, j), max(i, j)))

        # Should have all 5*4/2 = 10 unique pairs
        assert len(canonical_pairs) == 10


class TestSplitDocumentForEval:
    def test_split_sizes(self):
        coords = generate_coordinates(n_points=20)
        doc = create_document(coords)
        observed, held_out = split_document_for_eval(doc, n_observed=180)

        assert len(observed.pairs) == 180
        assert len(held_out.pairs) == 10

    def test_split_covers_all_pairs(self):
        coords = generate_coordinates(n_points=20)
        doc = create_document(coords)
        observed, held_out = split_document_for_eval(doc, n_observed=180)

        all_pairs = set()
        for i, j in observed.pairs:
            all_pairs.add((min(i, j), max(i, j)))
        for i, j in held_out.pairs:
            all_pairs.add((min(i, j), max(i, j)))

        assert len(all_pairs) == 190


class TestDistanceDataset:
    def test_dataset_length(self):
        dataset = DistanceDataset(size=100, n_points=20, seed=42)
        assert len(dataset) == 100

    def test_dataset_item_keys(self):
        dataset = DistanceDataset(size=10, n_points=20, seed=42)
        item = dataset[0]

        assert "text" in item
        assert "coordinates" in item
        assert "pairs" in item
        assert "distances" in item

    def test_text_format(self):
        dataset = DistanceDataset(size=10, n_points=20, seed=42)
        item = dataset[0]

        assert item["text"].startswith("<start>")
        assert item["text"].endswith("<end>")

    def test_reproducibility_with_seed(self):
        dataset1 = DistanceDataset(size=10, n_points=20, seed=42)
        dataset2 = DistanceDataset(size=10, n_points=20, seed=42)

        item1 = dataset1[5]
        item2 = dataset2[5]

        assert item1["text"] == item2["text"]
        assert np.allclose(item1["coordinates"], item2["coordinates"])


class TestEvalDataset:
    def test_dataset_length(self):
        dataset = EvalDataset(size=50, n_points=20, seed=42)
        assert len(dataset) == 50

    def test_dataset_item_keys(self):
        dataset = EvalDataset(size=10, n_points=20, n_observed=180, seed=42)
        item = dataset[0]

        assert "prompt" in item
        assert "observed_pairs" in item
        assert "observed_distances" in item
        assert "held_out_pairs" in item
        assert "held_out_distances" in item
        assert "coordinates" in item

    def test_split_sizes(self):
        dataset = EvalDataset(size=10, n_points=20, n_observed=180, seed=42)
        item = dataset[0]

        assert len(item["observed_pairs"]) == 180
        assert len(item["held_out_pairs"]) == 10


class TestGetSpecialTokens:
    def test_includes_start_end(self):
        tokens = get_special_tokens()
        assert "<start>" in tokens
        assert "<end>" in tokens

    def test_includes_point_tokens(self):
        tokens = get_special_tokens()
        for i in range(20):
            assert f"<point {i}>" in tokens

    def test_includes_distance_tokens(self):
        tokens = get_special_tokens()
        # Check some distance tokens
        assert "<0>" in tokens
        assert "<100>" in tokens
        assert "<500>" in tokens
        assert "<1000>" in tokens

    def test_total_token_count(self):
        tokens = get_special_tokens()
        # 2 (start/end) + 20 (points) + 1001 (distances 0-1000)
        assert len(tokens) == 2 + 20 + 1001
