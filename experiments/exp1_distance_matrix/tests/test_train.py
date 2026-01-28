"""Tests for training utilities."""

from experiments.exp1_distance_matrix.src.train import parse_model_output


class TestParseModelOutput:
    def test_parse_single_pair(self):
        output = "<p0> <p1> <d123>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 123}

    def test_parse_multiple_pairs(self):
        output = "<p0> <p1> <d100>\n<p2> <p5> <d200>\n<p10> <p15> <d300>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 100, (2, 5): 200, (10, 15): 300}

    def test_canonical_ordering(self):
        # Even if output has reversed order, key should be canonical
        output = "<p5> <p2> <d150>"
        predictions = parse_model_output(output)
        assert predictions == {(2, 5): 150}

    def test_parse_with_surrounding_text(self):
        output = "some text <p3> <p7> <d42> more text"
        predictions = parse_model_output(output)
        assert predictions == {(3, 7): 42}

    def test_parse_empty_output(self):
        output = "no valid pairs here"
        predictions = parse_model_output(output)
        assert predictions == {}

    def test_parse_full_document(self):
        output = """<start>
<p0> <p1> <d100>
<p0> <p2> <d150>
<p1> <p2> <d200>
<end>"""
        predictions = parse_model_output(output)
        assert len(predictions) == 3
        assert predictions[(0, 1)] == 100
        assert predictions[(0, 2)] == 150
        assert predictions[(1, 2)] == 200

    def test_last_value_wins_for_duplicate_pairs(self):
        # If same pair appears twice, last value wins
        output = "<p0> <p1> <d100>\n<p0> <p1> <d200>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 200}

    def test_parse_without_spaces(self):
        # Should also work without spaces between tokens
        output = "<p0><p1><d123>"
        predictions = parse_model_output(output)
        assert predictions == {(0, 1): 123}
