"""Smoke-level regression checks for the core colocation mining pipeline."""

import pandas as pd
import pytest

from src.colocation import discover_colocations


def _events() -> pd.DataFrame:
    """Coordinates matching Fig. 2 row-instance counts from the paper."""
    rows = [
        # feature  inst  x      y
        ("A", 1, 0.0, 0.0),
        ("A", 2, 10.0, 12.0),
        ("A", 3, 10.0, 10.0),
        ("A", 4, 100.0, 100.0),
        ("B", 1, 1.4, 0.0),
        ("B", 2, 100.0, 0.0),
        ("B", 3, 101.0, 0.0),
        ("B", 4, 10.0, 11.0),
        ("B", 5, 200.0, 200.0),
        ("C", 1, 11.0, 10.0),
        ("C", 2, 0.0, 1.4),
    ]
    df = pd.DataFrame(rows, columns=["feature", "label", "x", "y"])
    df["feature_type"] = df["feature"]
    df["instance_id"] = df.index
    return df[["instance_id", "feature_type", "x", "y"]]


def test_smoketest_fig2_and_multiresolution_equivalence() -> None:
    """Reproduce Fig. 2 PI values and ensure multiresolution matches fine results."""
    events = _events()

    res = discover_colocations(
        events,
        distance=1.5,
        min_prevalence=0.0,
        min_conditional_prob=0.0,
    )

    assert res.prevalent[("A", "B")] == pytest.approx(0.4)
    assert res.prevalent[("A", "C")] == pytest.approx(0.5)
    assert res.prevalent[("B", "C")] == pytest.approx(0.2)
    assert res.prevalent[("A", "B", "C")] == pytest.approx(0.2)
    assert len(res.table_instances[("A", "B", "C")]) == 1

    res_mr = discover_colocations(
        events,
        distance=1.5,
        min_prevalence=0.0,
        min_conditional_prob=0.0,
        use_multiresolution=True,
    )

    assert set(res_mr.prevalent) == set(res.prevalent)
    for colocation, pi in res.prevalent.items():
        assert res_mr.prevalent[colocation] == pytest.approx(pi)
