from src.colocation import _join_combinatorial


def test_join_combinatorial_joins_on_prefix_and_filters_with_adjacency() -> None:
    parent_a = [(11, 21, 31), (10, 20, 30)]
    parent_b = [(10, 20, 41), (11, 21, 42), (10, 20, 40)]

    adjacency: list[set[int]] = [set() for _ in range(43)]
    adjacency[30] = {40}
    adjacency[31] = {42}

    out = _join_combinatorial(parent_a, parent_b, adjacency)

    assert out == [(10, 20, 30, 40), (11, 21, 31, 42)]


def test_join_combinatorial_skips_rows_when_prefix_missing_or_not_adjacent() -> None:
    parent_a = [(10, 20, 30), (12, 22, 32)]
    parent_b = [(10, 20, 40), (13, 23, 43)]  # no prefix (12, 22) bucket

    adjacency: list[set[int]] = [set() for _ in range(44)]
    adjacency[30] = {41}  # does not contain 40, so prefix match fails adjacency
    adjacency[32] = {43}

    out = _join_combinatorial(parent_a, parent_b, adjacency)

    assert out == []


def test_join_combinatorial_returns_empty_for_empty_inputs() -> None:
    adjacency: list[set[int]] = [set() for _ in range(50)]

    assert _join_combinatorial([], [(10, 20, 30)], adjacency) == []
    assert _join_combinatorial([(10, 20, 30)], [], adjacency) == []


def test_join_combinatorial_supports_parent_size_four_and_up() -> None:
    parent_a = [(1, 2, 3, 7)]
    parent_b = [(1, 2, 3, 8), (1, 2, 3, 9)]

    adjacency: list[set[int]] = [set() for _ in range(10)]
    adjacency[7] = {9}

    out = _join_combinatorial(parent_a, parent_b, adjacency)

    assert out == [(1, 2, 3, 7, 9)]
