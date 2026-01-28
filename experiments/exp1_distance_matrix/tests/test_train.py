"""Tests for training utilities."""

from experiments.exp1_distance_matrix.src.train import parse_model_output


class TestParseModelOutput:
    def test_parse_single_pair(self):
        output = "<point 0><point 1><123>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 123}

    def test_parse_multiple_pairs(self):
        output = "<point 0><point 1><100>\n<point 2><point 5><200>\n<point 10><point 15><300>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 100, (2, 5): 200, (10, 15): 300}

    def test_canonical_ordering(self):
        # Even if output has reversed order, key should be canonical
        output = "<point 5><point 2><150>"
        predictions = parse_model_output(output)
        assert predictions == {(2, 5): 150}

    def test_parse_with_surrounding_text(self):
        output = "some text <point 3><point 7><42> more text"
        predictions = parse_model_output(output)
        assert predictions == {(3, 7): 42}

    def test_parse_empty_output(self):
        output = "no valid pairs here"
        predictions = parse_model_output(output)
        assert predictions == {}

    def test_parse_full_document(self):
        output = """<start>
<point 0><point 1><100>
<point 0><point 2><150>
<point 1><point 2><200>
<end>"""
        predictions = parse_model_output(output)
        assert len(predictions) == 3
        assert predictions[(0, 1)] == 100
        assert predictions[(0, 2)] == 150
        assert predictions[(1, 2)] == 200

    def test_last_value_wins_for_duplicate_pairs(self):
        # If same pair appears twice, last value wins
        output = "<point 0><point 1><100>\n<point 0><point 1><200>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 200}
