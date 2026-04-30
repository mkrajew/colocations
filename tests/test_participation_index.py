from src.colocation import _participation_index


def test_participation_index_returns_zeros_for_empty_table() -> None:
    cand = ("A", "B", "C")
    feature_counts = {"A": 10, "B": 20, "C": 30}

    pi, pr = _participation_index(cand, table=[], feature_counts=feature_counts)

    assert pi == 0.0
    assert pr == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_participation_index_uses_unique_instances_per_column() -> None:
    cand = ("A", "B", "C")
    table = [
        (0, 10, 20),
        (0, 11, 20),
        (1, 10, 21),
        (1, 12, 21),
    ]
    feature_counts = {"A": 4, "B": 6, "C": 3}

    pi, pr = _participation_index(cand, table=table, feature_counts=feature_counts)

    assert pr == {
        "A": 2.0 / 4.0,  # unique A instances: {0, 1}
        "B": 3.0 / 6.0,  # unique B instances: {10, 11, 12}
        "C": 2.0 / 3.0,  # unique C instances: {20, 21}
    }
    assert pi == min(pr.values())
    assert pi == 0.5


def test_participation_index_handles_zero_feature_count_with_max_guard() -> None:
    cand = ("A", "B")
    table = [(0, 10), (1, 10)]
    feature_counts = {"A": 0, "B": 5}

    pi, pr = _participation_index(cand, table=table, feature_counts=feature_counts)

    assert pr["A"] == 2.0  # denominator max(0, 1) == 1
    assert pr["B"] == 1.0 / 5.0
    assert pi == 1.0 / 5.0

