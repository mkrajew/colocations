import numpy as np

from src.colocation import _build_neighbors


def _pairs_as_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, row)) for row in pairs.tolist()}


def test_build_neighbors_returns_expected_pairs_and_adjacency() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    pairs, adjacency = _build_neighbors(coords, distance=1.0)

    assert _pairs_as_set(pairs) == {(0, 1), (0, 2)}
    assert adjacency == [{1, 2}, {0}, {0}, set()]

    for i, nbrs in enumerate(adjacency):
        assert i not in nbrs
        for j in nbrs:
            assert i in adjacency[j]


def test_build_neighbors_includes_pairs_on_distance_boundary() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [3.0, 4.0],  # exactly distance 5 from index 0
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    pairs, adjacency = _build_neighbors(coords, distance=5.0)

    assert _pairs_as_set(pairs) == {(0, 1)}
    assert adjacency == [{1}, {0}, set()]


def test_build_neighbors_with_zero_distance_keeps_only_duplicate_points() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    pairs, adjacency = _build_neighbors(coords, distance=0.0)

    assert _pairs_as_set(pairs) == {(0, 1)}
    assert adjacency == [{1}, {0}, set()]


def test_build_neighbors_empty_input() -> None:
    coords = np.empty((0, 2), dtype=np.float64)

    pairs, adjacency = _build_neighbors(coords, distance=1.0)

    assert pairs.shape == (0, 2)
    assert adjacency == []

