from src.colocation import apriori_gen


def test_apriori_gen_returns_empty_for_empty_input() -> None:
    assert apriori_gen([]) == []


def test_apriori_gen_joins_by_prefix_and_prunes_non_prevalent_subsets() -> None:
    prev_size_2 = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
    ]

    assert apriori_gen(prev_size_2) == [("A", "B", "C")]


def test_apriori_gen_returns_sorted_candidates_across_prefix_groups() -> None:
    prev_size_3 = [
        ("A", "B", "C"),
        ("A", "B", "D"),
        ("A", "B", "E"),
        ("A", "C", "D"),
        ("A", "C", "E"),
        ("A", "D", "E"),
        ("B", "C", "D"),
        ("B", "C", "E"),
        ("B", "D", "E"),
        ("C", "D", "E"),
    ]

    assert apriori_gen(prev_size_3) == [
        ("A", "B", "C", "D"),
        ("A", "B", "C", "E"),
        ("A", "B", "D", "E"),
        ("A", "C", "D", "E"),
        ("B", "C", "D", "E"),
    ]
