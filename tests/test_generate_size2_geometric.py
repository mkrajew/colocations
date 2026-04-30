import numpy as np

from src.colocation import _generate_size2_geometric


def test_generate_size2_geometric_groups_pairs_by_sorted_feature_key() -> None:
    feat_arr = np.asarray(["C", "A", "B", "A"])
    pairs = np.asarray(
        [
            [2, 3],  # (B, A) -> key ("A", "B"), row (3, 2)
            [0, 1],  # (C, A) -> key ("A", "C"), row (1, 0)
            [1, 2],  # (A, B) -> key ("A", "B"), row (1, 2)
            [0, 3],  # (C, A) -> key ("A", "C"), row (3, 0)
        ],
        dtype=np.int64,
    )

    tables = _generate_size2_geometric(pairs, feat_arr)

    assert tables == {
        ("A", "B"): [(1, 2), (3, 2)],
        ("A", "C"): [(1, 0), (3, 0)],
    }


def test_generate_size2_geometric_skips_same_feature_pairs() -> None:
    feat_arr = np.asarray(["A", "A", "B"])
    pairs = np.asarray(
        [
            [0, 1],  # same feature ("A", "A"), should be ignored
            [1, 2],
            [0, 2],
        ],
        dtype=np.int64,
    )

    tables = _generate_size2_geometric(pairs, feat_arr)

    assert tables == {("A", "B"): [(0, 2), (1, 2)]}


def test_generate_size2_geometric_returns_empty_when_no_pairs() -> None:
    feat_arr = np.asarray(["A", "B", "C"])
    pairs = np.empty((0, 2), dtype=np.int64)

    tables = _generate_size2_geometric(pairs, feat_arr)

    assert tables == {}

